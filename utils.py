"""
CyberWatch AI V3 — shared model loading + feature extraction.
All models load lazily (only when their module is first used) and
strictly follow the feature order saved in models/*_feature_columns.json
so the app can never drift from the training pipeline.
"""

import json
import math
import pickle
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

BASE_DIR   = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _p(name: str) -> Path:
    path = MODELS_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"Missing model file: {path}\n"
            "Run `python download_models.py` first (or extract "
            "CyberWatch_All_Models_V3.zip into the models/ folder)."
        )
    return path


def _pickle(name: str):
    with open(_p(name), "rb") as f:
        return pickle.load(f)


def _json(name: str):
    with open(_p(name)) as f:
        return json.load(f)


# ============================================================
# MODULE 1 — Network Intrusion (XGBoost + LSTM)
# ============================================================

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=0.3)
        self.fc1     = nn.Linear(hidden_size, 64)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2     = nn.Linear(64, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc2(self.dropout(self.relu(self.fc1(out))))


@lru_cache(maxsize=1)
def load_network():
    from xgboost import XGBClassifier
    model = XGBClassifier()
    model.load_model(str(_p("xgb_network_model.json")))  # version-safe format
    bundle = {
        "model":   model,
        "le":      _pickle("le_network.pkl"),
        "scaler":  _pickle("scaler_network.pkl"),
        "columns": _json("network_feature_columns.json"),
    }
    return bundle


@lru_cache(maxsize=1)
def load_network_lstm():
    net = load_network()
    lstm = LSTMClassifier(input_size=len(net["columns"]), hidden_size=128,
                          num_layers=2, num_classes=len(net["le"].classes_))
    lstm.load_state_dict(torch.load(_p("lstm_network_model.pt"),
                                    map_location=DEVICE))
    lstm.to(DEVICE).eval()
    return lstm


def predict_network(features):
    """features: dict {col: value} or array-like in saved column order."""
    net = load_network()
    if isinstance(features, dict):
        row = np.array([float(features.get(c, 0.0)) for c in net["columns"]])
    else:
        row = np.asarray(features, dtype="float64")
        if row.shape[-1] != len(net["columns"]):
            raise ValueError(f"Expected {len(net['columns'])} features, "
                             f"got {row.shape[-1]}")
    row = np.clip(row.reshape(1, -1), -1e9, 1e9).astype("float32")
    row = net["scaler"].transform(row)
    proba = net["model"].predict_proba(row)[0]
    idx   = int(np.argmax(proba))
    return net["le"].classes_[idx], round(float(proba[idx]) * 100, 2)


def predict_network_burst(window_rows):
    """window_rows: array (10, n_features) of consecutive flows (raw scale)."""
    net  = load_network()
    lstm = load_network_lstm()
    X = np.clip(np.asarray(window_rows, dtype="float64"), -1e9, 1e9)
    X = net["scaler"].transform(X.astype("float32"))
    with torch.no_grad():
        out   = lstm(torch.FloatTensor(X).unsqueeze(0).to(DEVICE))
        proba = torch.softmax(out, dim=1)[0]
        idx   = int(torch.argmax(proba).item())
    return net["le"].classes_[idx], round(float(proba[idx]) * 100, 2)


# ============================================================
# MODULE 2 — Phishing URL (RF + XGBoost ensemble)
# ============================================================

TRUSTED_DOMAINS = [
    "google.com", "youtube.com", "facebook.com", "instagram.com",
    "twitter.com", "linkedin.com", "netflix.com", "amazon.com",
    "microsoft.com", "apple.com", "github.com", "stackoverflow.com",
    "wikipedia.org", "reddit.com", "sbi.co.in", "hdfcbank.com",
    "icicibank.com", "kaggle.com",
]


def extract_url_features(url):
    """v4 extractor — must stay byte-identical to the training notebook."""
    url = str(url).strip()
    url = re.sub(r"^https?://", "", url, flags=re.IGNORECASE)
    features = {}

    features["url_length"] = len(url)
    features["num_dots"] = url.count(".")
    features["num_hyphens"] = url.count("-")
    features["num_underscores"] = url.count("_")
    features["num_slashes"] = url.count("/")
    features["num_question_marks"] = url.count("?")
    features["num_equal_signs"] = url.count("=")
    features["num_at_signs"] = url.count("@")
    features["num_ampersands"] = url.count("&")
    features["num_hash"] = url.count("#")
    features["num_percent"] = url.count("%")
    features["num_digits"] = sum(c.isdigit() for c in url)
    features["num_letters"] = sum(c.isalpha() for c in url)
    features["num_special"] = sum(not c.isalnum() for c in url)

    l = len(url) if len(url) > 0 else 1
    features["digit_ratio"] = features["num_digits"] / l
    features["letter_ratio"] = features["num_letters"] / l
    features["special_ratio"] = features["num_special"] / l
    features["vowel_ratio"] = sum(c in "aeiou" for c in url.lower()) / l
    features["digit_letter_ratio"] = features["num_digits"] / (
        features["num_letters"] + 1)

    features["has_ip"] = 1 if re.search(
        r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", url) else 0
    features["has_port"] = 1 if re.search(r":\d+", url) else 0
    features["has_double_slash"] = url.count("//")
    features["has_triple_www"] = 1 if url.count("www") > 1 else 0
    features["has_shortener"] = 1 if any(
        s in url for s in ["bit.ly", "tinyurl", "goo.gl", "t.co",
                           "ow.ly", "is.gd", "buff.ly", "tiny.cc"]) else 0

    suspicious_words = [
        "login", "signin", "verify", "secure", "account", "update",
        "banking", "confirm", "password", "paypal", "ebay", "amazon",
        "apple", "microsoft", "google", "bank", "free", "winner",
        "prize", "click", "urgent", "suspended", "limited", "validate",
        "credential", "recover", "unusual", "suspend", "restricted",
        "billing", "payment", "invoice", "security",
    ]
    features["num_suspicious_words"] = sum(
        1 for w in suspicious_words if w in url.lower())
    features["has_suspicious_word"] = 1 if features["num_suspicious_words"] > 0 else 0

    brands = ["paypal", "ebay", "amazon", "apple", "microsoft", "google",
              "facebook", "netflix", "instagram", "twitter", "linkedin",
              "dropbox", "adobe", "yahoo", "outlook", "office365"]
    features["num_brand_names"] = sum(1 for b in brands if b in url.lower())
    features["has_brand_name"] = 1 if features["num_brand_names"] > 0 else 0

    try:
        domain = url.split("/")[0]
        domain = domain.split(":")[0]
        features["domain_length"] = len(domain)
        features["num_subdomains"] = max(0, domain.count(".") - 1)
        features["domain_has_digit"] = 1 if any(c.isdigit() for c in domain) else 0
        features["domain_has_hyphen"] = 1 if "-" in domain else 0
        features["domain_num_hyphens"] = domain.count("-")
        tld = domain.split(".")[-1] if "." in domain else ""
        features["tld_length"] = len(tld)
        suspicious_tlds = ["tk", "ml", "ga", "cf", "gq", "xyz", "top",
                           "click", "download", "stream", "gdn", "loan",
                           "men", "work", "party", "date", "racing",
                           "review", "accountant", "science", "faith"]
        features["has_suspicious_tld"] = 1 if tld in suspicious_tlds else 0
        trusted_tlds = ["com", "org", "edu", "gov", "net", "in", "co"]
        features["has_trusted_tld"] = 1 if tld in trusted_tlds else 0
    except Exception:
        for k in ["domain_length", "num_subdomains", "domain_has_digit",
                  "domain_has_hyphen", "domain_num_hyphens", "tld_length",
                  "has_suspicious_tld", "has_trusted_tld"]:
            features[k] = 0

    try:
        path = "/".join(url.split("/")[1:])
        features["path_length"] = len(path)
        features["path_depth"] = path.count("/")
        features["path_has_exe"] = 1 if any(
            ext in path.lower() for ext in [".exe", ".php", ".asp", ".jsp"]) else 0
    except Exception:
        features["path_length"] = 0
        features["path_depth"] = 0
        features["path_has_exe"] = 0

    features["has_query"] = 1 if "?" in url else 0
    features["query_length"] = len(url.split("?")[1]) if "?" in url else 0
    features["num_query_params"] = url.count("=")
    features["has_encoding"] = 1 if "%" in url else 0
    features["num_encoded_chars"] = url.count("%")

    try:
        features["has_consecutive_numbers"] = 1 if re.search(
            r"\d{4,}", url.split("/")[0]) else 0
    except Exception:
        features["has_consecutive_numbers"] = 0

    char_freq = {}
    for c in url:
        char_freq[c] = char_freq.get(c, 0) + 1
    entropy = 0
    for freq in char_freq.values():
        p = freq / len(url) if len(url) > 0 else 1
        if p > 0:
            entropy -= p * math.log2(p)
    features["url_entropy"] = round(entropy, 4)

    return features


@lru_cache(maxsize=1)
def load_phishing():
    from xgboost import XGBClassifier
    xgb = XGBClassifier()
    xgb.load_model(str(_p("xgb_phishing_model.json")))
    return {
        "rf":      _pickle("rf_phishing_model.pkl"),   # ~1 GB, lazy by design
        "xgb":     xgb,
        "le":      _pickle("le_phishing.pkl"),
        "scaler":  _pickle("scaler_phishing.pkl"),
        "columns": _json("phishing_feature_columns.json"),
        "w":       _json("phishing_ensemble_config.json")["xgb_weight"],
    }


def predict_url(url):
    url_str = str(url).strip()
    try:
        bare = re.sub(r"^https?://", "", url_str, flags=re.IGNORECASE)
        domain = bare.split("/")[0].replace("www.", "").split(":")[0]
        for trusted in TRUSTED_DOMAINS:
            if domain == trusted or domain.endswith("." + trusted):
                return "SAFE", 99.0, "trusted domain whitelist"
    except Exception:
        pass

    ph = load_phishing()
    feats = extract_url_features(url_str)
    row = np.array([feats[c] for c in ph["columns"]],
                   dtype="float32").reshape(1, -1)
    row = ph["scaler"].transform(row)
    w = ph["w"]
    proba = ph["rf"].predict_proba(row) * (1 - w) + ph["xgb"].predict_proba(row) * w
    idx  = int(np.argmax(proba))
    conf = round(float(np.max(proba)) * 100, 2)
    label = "SAFE" if ph["le"].classes_[idx] == "good" else "PHISHING"
    return label, conf, "ML ensemble (RF + XGBoost)"


# ============================================================
# MODULE 3 — Malware families (CNN on memory-analysis features)
# ============================================================

class MalwareCNN(nn.Module):
    def __init__(self, input_length, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool    = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)
        conv_out = input_length // 8
        self.fc1 = nn.Linear(128 * conv_out, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        return self.fc3(x)


@lru_cache(maxsize=1)
def load_malware():
    le      = _pickle("le_malware.pkl")
    columns = _json("malware_feature_columns.json")
    cnn = MalwareCNN(input_length=len(columns), num_classes=len(le.classes_))
    cnn.load_state_dict(torch.load(_p("cnn_malware_model.pt"),
                                   map_location=DEVICE))
    cnn.to(DEVICE).eval()
    return {"model": cnn, "le": le,
            "scaler": _pickle("scaler_malware.pkl"), "columns": columns}


def predict_malware(features):
    """features: dict {col: value} or array in saved column order (55)."""
    mal = load_malware()
    if isinstance(features, dict):
        row = np.array([float(features.get(c, 0.0)) for c in mal["columns"]])
    else:
        row = np.asarray(features, dtype="float64")
        if row.shape[-1] != len(mal["columns"]):
            raise ValueError(f"Expected {len(mal['columns'])} features, "
                             f"got {row.shape[-1]}")
    row = mal["scaler"].transform(
        row.reshape(1, -1).astype("float32")).astype("float32")
    with torch.no_grad():
        out   = mal["model"](torch.FloatTensor(row).unsqueeze(1).to(DEVICE))
        proba = torch.softmax(out, dim=1)[0]
        idx   = int(torch.argmax(proba).item())
    return mal["le"].classes_[idx], round(float(proba[idx]) * 100, 2)


# ============================================================
# MODULE 4 — Social media crime (BERT + insult guardrail)
# ============================================================

# DynaHate labels person-directed insults as "not hate", which taught
# BERT to call them Normal. This lexicon guardrail restores the
# Offensive Language flag for plain insults — hybrid rules+ML, the
# standard approach in production moderation systems.
INSULT_LEXICON = [
    "idiot", "stupid", "dumb", "moron", "loser", "pathetic", "fool",
    "jerk", "trash", "garbage", "crap", "shut up", "nobody asked",
    "clown", "worthless", "useless", "disgusting", "shit", "bullshit",
    "fuck", "fucking", "bitch", "bastard", "asshole", "ass", "damn",
    "hell", "scum", "filth", "retard", "creep",
]
_INSULT_RE = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in INSULT_LEXICON) + r")\b",
    re.IGNORECASE)


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"\brt\b", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@lru_cache(maxsize=1)
def load_social():
    from transformers import AutoTokenizer, BertConfig, BertForSequenceClassification
    le = _pickle("le_social.pkl")
    # AutoTokenizer reads tokenizer.json (the fast format) — works offline
    tokenizer = AutoTokenizer.from_pretrained(str(MODELS_DIR / "bert_tokenizer"))
    # Default BertConfig == bert-base-uncased architecture; building the
    # model from config avoids downloading 440 MB of pretrained weights
    # that our fine-tuned state dict fully replaces anyway.
    config = BertConfig(num_labels=len(le.classes_))
    model = BertForSequenceClassification(config)
    model.load_state_dict(torch.load(_p("bert_best_model.pt"),
                                     map_location=DEVICE))
    model.to(DEVICE).eval()
    return {"model": model, "tokenizer": tokenizer, "le": le}


def predict_social(text):
    soc = load_social()
    enc = soc["tokenizer"](clean_text(text), max_length=128,
                           padding="max_length", truncation=True,
                           return_tensors="pt")
    with torch.no_grad():
        out = soc["model"](input_ids=enc["input_ids"].to(DEVICE),
                           attention_mask=enc["attention_mask"].to(DEVICE))
        proba = torch.softmax(out.logits, dim=1)[0]
        idx   = int(torch.argmax(proba).item())
    label = soc["le"].classes_[idx]
    conf  = round(float(proba[idx]) * 100, 2)
    source = "BERT"
    if label == "Normal" and _INSULT_RE.search(str(text)):
        label, source = "Offensive Language", "BERT + insult guardrail"
    return label, conf, source


# ============================================================
# ANOMALY — Isolation Forest + Autoencoder (zero-day detection)
# ============================================================

class DeepAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.BatchNorm1d(64), nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.LeakyReLU(0.2),
            nn.Linear(32, 16), nn.BatchNorm1d(16), nn.LeakyReLU(0.2),
            nn.Linear(16, 8))
        self.decoder = nn.Sequential(
            nn.Linear(8, 16), nn.BatchNorm1d(16), nn.LeakyReLU(0.2),
            nn.Linear(16, 32), nn.BatchNorm1d(32), nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(32, 64), nn.BatchNorm1d(64), nn.LeakyReLU(0.2),
            nn.Linear(64, input_dim))

    def forward(self, x):
        return self.decoder(self.encoder(x))


@lru_cache(maxsize=1)
def load_anomaly():
    ae = DeepAutoencoder(input_dim=20)
    ae.load_state_dict(torch.load(_p("autoencoder_model.pt"),
                                  map_location=DEVICE))
    ae.to(DEVICE).eval()
    return {
        "iforest":   _pickle("isolation_forest.pkl"),
        "scaler":    _pickle("scaler_anomaly.pkl"),
        "pca":       _pickle("pca_anomaly.pkl"),
        "ae":        ae,
        "threshold": float(_pickle("ae_threshold.pkl")),
        "columns":   _json("network_feature_columns.json"),
    }


def predict_anomaly(features, sensitivity=1.0):
    """Returns (is_threat, detail). sensitivity scales the AE threshold:
    <1.0 = more alerts, >1.0 = fewer false alarms."""
    an = load_anomaly()
    if isinstance(features, dict):
        row = np.array([float(features.get(c, 0.0)) for c in an["columns"]])
    else:
        row = np.asarray(features, dtype="float64")
    row = np.clip(row.reshape(1, -1), -1e9, 1e9).astype("float32")
    row_pca = an["pca"].transform(an["scaler"].transform(row))

    if_flag = bool(an["iforest"].predict(row_pca)[0] == -1)
    with torch.no_grad():
        t = torch.FloatTensor(row_pca).to(DEVICE)
        err = float(torch.mean((an["ae"](t) - t) ** 2).item())
    ae_flag = err > an["threshold"] * sensitivity

    return (if_flag or ae_flag), {
        "isolation_forest": "ANOMALY" if if_flag else "normal",
        "autoencoder_error": round(err, 6),
        "threshold": round(an["threshold"] * sensitivity, 6),
        "autoencoder": "ANOMALY" if ae_flag else "normal",
    }


def get_metrics():
    return _json("metrics.json")
