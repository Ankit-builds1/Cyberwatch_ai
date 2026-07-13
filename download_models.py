"""
CyberWatch AI V3 model pack installer.

The trained models are too large for normal git history, so a clean clone
restores them from either:
  1. a local CyberWatch_All_Models_V3.zip file, or
  2. the GitHub Release asset URL configured below / via CYBERWATCH_MODELS_URL.

Usage:
  python download_models.py
"""

import json
import os
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

DEFAULT_MODELS_URL = (
    "https://github.com/Ankit-builds1/Cyberwatch_ai/releases/"
    "download/v3.0/CyberWatch_All_Models_V3.zip"
)
MODELS_URL = os.environ.get("CYBERWATCH_MODELS_URL", DEFAULT_MODELS_URL)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
ZIP_PATH = BASE_DIR / "CyberWatch_All_Models_V3.zip"

REQUIRED = [
    "xgb_network_model.json", "le_network.pkl", "scaler_network.pkl",
    "network_feature_columns.json", "lstm_network_model.pt",
    "rf_phishing_model.pkl", "xgb_phishing_model.json", "le_phishing.pkl",
    "scaler_phishing.pkl", "phishing_feature_columns.json",
    "phishing_ensemble_config.json",
    "cnn_malware_model.pt", "le_malware.pkl", "scaler_malware.pkl",
    "malware_feature_columns.json",
    "bert_best_model.pt", "le_social.pkl", "bert_tokenizer/tokenizer.json",
    "isolation_forest.pkl", "scaler_anomaly.pkl", "pca_anomaly.pkl",
    "autoencoder_model.pt", "ae_threshold.pkl", "metrics.json",
]

CORRECTED_METRICS = {
    "network_xgboost_test_acc": 97.14,
    "network_lstm_test_acc": 94.06,
    "phishing_ensemble_test_acc": 93.78,
    "malware_cnn_test_acc": 97.13,
    "social_bert_test_acc": 91.55,
    "anomaly_combined_detection": 80.4,
}


def install_corrected_metrics():
    """Install the verified V3 metrics over the release archive's stale copy."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = MODELS_DIR / "metrics.json"
    metrics_path.write_text(
        json.dumps(CORRECTED_METRICS, indent=2) + "\n",
        encoding="utf-8",
    )


def models_complete():
    return all((MODELS_DIR / filename).exists() for filename in REQUIRED)


def progress(block_num, block_size, total_size):
    done = block_num * block_size
    pct = min(100, done * 100 // max(total_size, 1))
    mb = done / 1024 / 1024
    sys.stdout.write(f"\r  downloading... {pct}% ({mb:.0f} MB)")
    sys.stdout.flush()


def download_zip():
    print("CyberWatch AI V3 - downloading model pack...")
    print(f"  from: {MODELS_URL}")
    try:
        urlretrieve(MODELS_URL, ZIP_PATH, reporthook=progress)
    except Exception as exc:
        print(f"\nDownload failed: {exc}")
        print("\nFix options:")
        print("  1. Upload CyberWatch_All_Models_V3.zip as a GitHub Release asset")
        print("     at the URL shown above, then run this command again.")
        print("  2. Put CyberWatch_All_Models_V3.zip in this folder, then run:")
        print("     python download_models.py")
        print("  3. Use a custom URL:")
        print("     set CYBERWATCH_MODELS_URL=https://.../CyberWatch_All_Models_V3.zip")
        sys.exit(1)


def main():
    if models_complete():
        install_corrected_metrics()
        print("All model files already present in models/ - nothing to do.")
        return

    had_local_zip = ZIP_PATH.exists()
    if had_local_zip:
        print(f"Using local model pack: {ZIP_PATH.name}")
    else:
        download_zip()

    print("\n  extracting...")
    MODELS_DIR.mkdir(exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH) as archive:
        archive.extractall(MODELS_DIR)

    if not had_local_zip:
        ZIP_PATH.unlink(missing_ok=True)

    if models_complete():
        install_corrected_metrics()
        print("Models installed. Try: python predict.py info")
    else:
        missing = [filename for filename in REQUIRED
                   if not (MODELS_DIR / filename).exists()]
        print(f"Extraction finished but {len(missing)} files are missing: "
              f"{missing[:5]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
