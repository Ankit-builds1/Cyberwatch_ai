"""
CyberWatch AI V3 — live network capture layer.

Sniffs traffic on the local machine, aggregates packets into time
windows, computes CICIoT2023-style features for each window, and runs
two detectors on every window:
  1. XGBoost attack-type classifier (Module 1)
  2. Isolation Forest + Autoencoder zero-day detector (Anomaly module)

Requirements:
  * Windows: install Npcap (https://npcap.com) with WinPcap-compatible mode
  * Run the terminal as Administrator
  * pip install scapy

HONEST NOTE: the 46 training features come from the CICIoT2023 toolchain.
This module reproduces their definitions as faithfully as scapy allows,
but live values are an approximation of the training distribution. The
anomaly detector (trained on "what normal looks like") transfers best;
treat the XGBoost attack-type label on live traffic as indicative.
"""

import math
import time
from collections import Counter

import numpy as np

import utils

try:
    from scapy.all import sniff, IP, TCP, UDP, ICMP, ARP, conf
except ImportError:
    sniff = None

PORT_PROTOCOLS = {
    "HTTP": {80, 8080}, "HTTPS": {443, 8443}, "DNS": {53},
    "Telnet": {23}, "SMTP": {25, 465, 587}, "SSH": {22},
    "IRC": {6667, 6697}, "DHCP": {67, 68},
}


def _local_ip():
    try:
        return conf.route.route("8.8.8.8")[1]
    except Exception:
        return None


def window_features(packets, window_seconds, local_ip):
    """Build the 46 CICIoT2023-style features from a packet window."""
    f = {c: 0.0 for c in utils.load_network()["columns"]}
    n = len(packets)
    if n == 0:
        return None

    sizes, ttls, protos, headers = [], [], [], []
    in_sizes, out_sizes, times = [], [], []
    flag_counts = Counter()
    seen_ports = set()
    layer_seen = Counter()

    for pkt in packets:
        size = len(pkt)
        sizes.append(size)
        times.append(float(pkt.time))
        headers.append(14)  # ethernet
        if pkt.haslayer(ARP):
            layer_seen["ARP"] += 1
        if pkt.haslayer(IP):
            ip = pkt[IP]
            layer_seen["IPv"] += 1
            ttls.append(ip.ttl)
            protos.append(ip.proto)
            headers[-1] += ip.ihl * 4
            if local_ip and ip.src == local_ip:
                out_sizes.append(size)
            else:
                in_sizes.append(size)
        if pkt.haslayer(TCP):
            tcp = pkt[TCP]
            layer_seen["TCP"] += 1
            headers[-1] += tcp.dataofs * 4
            seen_ports.update({tcp.sport, tcp.dport})
            flags = str(tcp.flags)
            for ch, name in [("F", "fin"), ("S", "syn"), ("R", "rst"),
                             ("P", "psh"), ("A", "ack"), ("E", "ece"),
                             ("C", "cwr"), ("U", "urg")]:
                if ch in flags:
                    flag_counts[name] += 1
        elif pkt.haslayer(UDP):
            layer_seen["UDP"] += 1
            headers[-1] += 8
            seen_ports.update({pkt[UDP].sport, pkt[UDP].dport})
        elif pkt.haslayer(ICMP):
            layer_seen["ICMP"] += 1

    dur = max(window_seconds, 1e-6)
    sizes_arr = np.array(sizes, dtype="float64")

    f["flow_duration"] = dur
    f["Header_Length"] = float(sum(headers))
    f["Protocol Type"] = float(Counter(protos).most_common(1)[0][0]) if protos else 0.0
    f["Duration"]      = float(np.mean(ttls)) if ttls else 64.0
    f["Rate"]          = n / dur
    f["Srate"]         = len(out_sizes) / dur
    f["Drate"]         = len(in_sizes) / dur

    for name in ["fin", "syn", "rst", "psh", "ack", "ece", "cwr"]:
        f[f"{name}_flag_number"] = 1.0 if flag_counts[name] > 0 else 0.0
    for name in ["ack", "syn", "fin", "urg", "rst"]:
        f[f"{name}_count"] = float(flag_counts[name])

    for proto_name, ports in PORT_PROTOCOLS.items():
        f[proto_name] = 1.0 if seen_ports & ports else 0.0
    for name in ["TCP", "UDP", "ICMP", "ARP", "IPv"]:
        f[name] = 1.0 if layer_seen[name] > 0 else 0.0
    f["LLC"] = 0.0

    f["Tot sum"]  = float(sizes_arr.sum())
    f["Min"]      = float(sizes_arr.min())
    f["Max"]      = float(sizes_arr.max())
    f["AVG"]      = float(sizes_arr.mean())
    f["Std"]      = float(sizes_arr.std())
    f["Tot size"] = float(sizes_arr.mean())
    f["Number"]   = float(n)
    f["Variance"] = float(sizes_arr.var())

    iats = np.diff(sorted(times)) if n > 1 else np.array([0.0])
    f["IAT"] = float(np.mean(iats))

    avg_in  = float(np.mean(in_sizes)) if in_sizes else 0.0
    avg_out = float(np.mean(out_sizes)) if out_sizes else 0.0
    var_in  = float(np.var(in_sizes)) if in_sizes else 0.0
    var_out = float(np.var(out_sizes)) if out_sizes else 0.0
    f["Magnitue"] = math.sqrt(avg_in + avg_out)          # (sic — dataset typo)
    f["Radius"]   = math.sqrt(var_in + var_out)
    f["Weight"]   = float(len(in_sizes) * len(out_sizes))
    k = min(len(in_sizes), len(out_sizes))
    f["Covariance"] = float(np.cov(in_sizes[:k], out_sizes[:k])[0][1]) if k > 1 else 0.0

    return f


def run(interface=None, window=2.0, sensitivity=1.0):
    if sniff is None:
        print("❌ scapy is not installed. Run: pip install scapy")
        return
    local_ip = _local_ip()
    print("=" * 62)
    print("  🛡️  CyberWatch AI V3 — LIVE NETWORK MONITOR")
    print(f"  interface: {interface or 'default'} | window: {window}s "
          f"| sensitivity: {sensitivity}")
    print(f"  local IP: {local_ip} | Ctrl+C to stop")
    print("=" * 62)

    # warm up models before capture starts
    utils.load_network()
    utils.load_anomaly()
    columns = utils.load_network()["columns"]
    print("✅ models loaded — monitoring...\n")

    windows_seen, alerts = 0, 0
    try:
        while True:
            packets = sniff(timeout=window, iface=interface)
            feats = window_features(list(packets), window, local_ip)
            if feats is None:
                continue
            windows_seen += 1
            row = [feats[c] for c in columns]

            label, conf = utils.predict_network(row)
            is_threat, detail = utils.predict_anomaly(row, sensitivity)

            stamp = time.strftime("%H:%M:%S")
            if label != "Benign" or is_threat:
                alerts += 1
                why = []
                if label != "Benign":
                    why.append(f"classifier: {label} ({conf}%)")
                if is_threat:
                    why.append(f"anomaly: AE err "
                               f"{detail['autoencoder_error']:.4f} "
                               f"> {detail['threshold']:.4f}"
                               if detail["autoencoder"] == "ANOMALY"
                               else "anomaly: isolation forest")
                print(f"🚨 {stamp} | THREAT | {int(feats['Number'])} pkts | "
                      + " + ".join(why))
            else:
                print(f"✅ {stamp} | normal | {int(feats['Number'])} pkts | "
                      f"{label} ({conf}%)")
    except KeyboardInterrupt:
        print(f"\n--- monitor stopped: {windows_seen} windows, "
              f"{alerts} alerts ---")
    except PermissionError:
        print("\n❌ Permission denied — run the terminal as Administrator "
              "and make sure Npcap is installed (https://npcap.com).")


if __name__ == "__main__":
    run()
