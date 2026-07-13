"""
CyberWatch AI V3 — command-line interface.

Usage examples:
  python predict.py url "http://paypal-login.tk/verify"
  python predict.py text "your social media post here"
  python predict.py network --csv flows.csv
  python predict.py malware --csv memory_features.csv
  python predict.py anomaly --csv flows.csv
  python predict.py live                       (real-time capture, needs admin + Npcap)
  python predict.py info                       (model metrics)
"""

import argparse
import sys

import numpy as np
import pandas as pd

import utils

BANNER = r"""
   ______      __              _       __      __       __
  / ____/_  __/ /_  ___  _____| |     / /___ _/ /______/ /_
 / /   / / / / __ \/ _ \/ ___/ | /| / / __ `/ __/ ___/ __ \
/ /___/ /_/ / /_/ /  __/ /   | |/ |/ / /_/ / /_/ /__/ / / /
\____/\__, /_.___/\___/_/    |__/|__/\__,_/\__/\___/_/ /_/
     /____/        AI V3 — Local Cybercrime Prevention
"""


def cmd_url(args):
    label, conf, source = utils.predict_url(args.value)
    icon = "✅" if label == "SAFE" else "🚨"
    print(f"\n{icon} {label} ({conf}% confidence)")
    print(f"   URL:    {args.value}")
    print(f"   Method: {source}")
    if label == "PHISHING":
        print("   ⚠️  Do NOT enter credentials or download files from this URL.")


def cmd_text(args):
    label, conf, source = utils.predict_social(args.value)
    icon = {"Normal": "✅", "Offensive Language": "⚠️", "Hate Speech": "🚨"}[label]
    print(f"\n{icon} {label} ({conf}% confidence)")
    print(f"   Text:   {args.value[:80]}")
    print(f"   Method: {source}")


def _load_rows(args, columns, kind):
    if args.csv:
        df = pd.read_csv(args.csv)
        missing = [c for c in columns if c not in df.columns]
        if missing:
            print(f"❌ CSV is missing {len(missing)} required {kind} columns, "
                  f"e.g. {missing[:5]}")
            print(f"   Required columns are listed in "
                  f"models/{kind}_feature_columns.json")
            sys.exit(1)
        return df[columns].apply(pd.to_numeric, errors="coerce").fillna(0).values
    if args.values:
        return np.array([[float(v) for v in args.values.split(",")]])
    print("❌ Provide --csv FILE or --values v1,v2,...")
    sys.exit(1)


def cmd_network(args):
    columns = utils.load_network()["columns"]
    rows = _load_rows(args, columns, "network")
    limit = args.limit or len(rows)
    print(f"\nClassifying {min(limit, len(rows))} flow(s)...\n")
    counts = {}
    for i, row in enumerate(rows[:limit]):
        label, conf = utils.predict_network(row)
        counts[label] = counts.get(label, 0) + 1
        icon = "✅" if label == "Benign" else "🚨"
        print(f"{icon} flow {i:4d} → {label:14s} ({conf}%)")
    print("\nSummary:", dict(sorted(counts.items(), key=lambda x: -x[1])))


def cmd_malware(args):
    columns = utils.load_malware()["columns"]
    rows = _load_rows(args, columns, "malware")
    limit = args.limit or len(rows)
    print(f"\nAnalysing {min(limit, len(rows))} sample(s)...\n")
    for i, row in enumerate(rows[:limit]):
        label, conf = utils.predict_malware(row)
        icon = "✅" if label == "Benign" else "🚨"
        extra = "" if label == "Benign" else "  ← MALWARE (family is an estimate)"
        print(f"{icon} sample {i:4d} → {label:11s} ({conf}%){extra}")


def cmd_anomaly(args):
    columns = utils.load_anomaly()["columns"]
    rows = _load_rows(args, columns, "network")
    limit = args.limit or len(rows)
    print(f"\nChecking {min(limit, len(rows))} flow(s) for zero-day anomalies "
          f"(sensitivity={args.sensitivity})...\n")
    threats = 0
    for i, row in enumerate(rows[:limit]):
        is_threat, detail = utils.predict_anomaly(row, args.sensitivity)
        threats += is_threat
        icon = "🚨 THREAT" if is_threat else "✅ normal"
        print(f"{icon} | flow {i:4d} | IF: {detail['isolation_forest']:7s} "
              f"| AE err: {detail['autoencoder_error']:.6f} "
              f"(thr {detail['threshold']:.6f})")
    print(f"\n{threats}/{min(limit, len(rows))} flows flagged as potential threats")


def cmd_live(args):
    import live_capture
    live_capture.run(interface=args.interface, window=args.window,
                     sensitivity=args.sensitivity)


def cmd_info(args):
    print("\nModel performance (honest held-out test results):")
    for k, v in utils.get_metrics().items():
        print(f"  {k:32s} {v}%")
    print("\nTraining notebook: "
          "kaggle.com/code/ankit20554/cyberwatch-ai-v3-complete")


def main():
    print(BANNER)
    parser = argparse.ArgumentParser(
        prog="predict.py",
        description="CyberWatch AI V3 — local cybercrime prevention CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("url", help="scan a URL for phishing")
    p.add_argument("value", help="the URL to scan")
    p.set_defaults(func=cmd_url)

    p = sub.add_parser("text", help="analyse social media text")
    p.add_argument("value", help="the text to analyse")
    p.set_defaults(func=cmd_text)

    p = sub.add_parser("network", help="classify network flows (CSV)")
    p.add_argument("--csv", help="CSV containing the 46 flow features")
    p.add_argument("--values", help="comma-separated 46 feature values")
    p.add_argument("--limit", type=int, help="max rows to classify")
    p.set_defaults(func=cmd_network)

    p = sub.add_parser("malware", help="classify memory-analysis features (CSV)")
    p.add_argument("--csv", help="CSV containing the 55 MalMem features")
    p.add_argument("--values", help="comma-separated 55 feature values")
    p.add_argument("--limit", type=int, help="max rows to classify")
    p.set_defaults(func=cmd_malware)

    p = sub.add_parser("anomaly", help="zero-day anomaly check on flows (CSV)")
    p.add_argument("--csv", help="CSV containing the 46 flow features")
    p.add_argument("--values", help="comma-separated 46 feature values")
    p.add_argument("--limit", type=int, help="max rows to check")
    p.add_argument("--sensitivity", type=float, default=1.0,
                   help="<1.0 = more alerts, >1.0 = fewer false alarms")
    p.set_defaults(func=cmd_anomaly)

    p = sub.add_parser("live", help="live network monitoring (admin + Npcap)")
    p.add_argument("--interface", default=None, help="network interface name")
    p.add_argument("--window", type=float, default=2.0,
                   help="aggregation window in seconds")
    p.add_argument("--sensitivity", type=float, default=1.0)
    p.set_defaults(func=cmd_live)

    p = sub.add_parser("info", help="show model metrics")
    p.set_defaults(func=cmd_info)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
