"""
=============================================================================
 HYBRID NETWORK THREAT DETECTOR -- Tkinter Dashboard
=============================================================================

 A GUI front-end for the hybrid (IsolationForest + LLM) network threat
 detector. It shows:

   * Live status bar      : capture state, training progress, counters
   * Flow table           : every analyzed flow, color-coded by risk
   * Detail panel         : ALL parameter values for the selected flow,
                            each with a plain-English explanation of what
                            the value means and whether it looks concerning
   * LLM analysis box      : the model's written reasoning
   * Metric legend         : quick reference for every feature

 Two run modes (pick in the UI):
   * LIVE  -> sniff real packets via scapy/npcap (needs admin + npcap)
   * DEMO  -> generate synthetic flows so you can see the whole UI work
              with NO drivers / admin rights (great for screenshots/dev)

 Threading model (important for Tkinter, which is single-threaded):
   * Packet capture + flow processing run in BACKGROUND threads.
   * Completed flows are pushed onto a thread-safe Queue.
   * The GUI polls that queue on the MAIN thread via root.after(),
     so all widget updates happen safely on the Tk thread.

 Dependencies:
     pip install scikit-learn pandas matplotlib
     pip install scapy ollama        # only needed for LIVE mode + LLM
=============================================================================
"""

import math
import random
import statistics
import threading
import time
import csv
import json
import os
from datetime import datetime, timezone
from collections import Counter
from queue import Queue, Empty
from ollama import Client
import tkinter as tk
from tkinter import ttk, scrolledtext

import pandas as pd
from sklearn.ensemble import IsolationForest

# --- Optional deps: import lazily so DEMO mode works without them ---
try:
    import ollama
    _HAS_OLLAMA = True
except Exception:
    _HAS_OLLAMA = False

try:
    import joblib
    _HAS_JOBLIB = True
except Exception:
    _HAS_JOBLIB = False

# Interlink to the standalone scapy attack simulator (for REAL packet sending).
try:
    import simulator.py as atksim
    _HAS_SIM = True
except Exception:
    _HAS_SIM = False

# Path to the supervised model bundle produced by train_and_evaluate.py.
SUPERVISED_MODEL_PATH = "threat_model.joblib"
MODEL_METRICS_PATH = os.path.join("model_eval", "metrics.json")

# Persisted IsolationForest baseline (this network's learned "normal").
BASELINE_PATH = "network_baseline.joblib"

# IsolationForest sensitivity presets -> contamination (expected anomaly %).
# Lower contamination = calmer detector = fewer false anomalies.
SENSITIVITY_PRESETS = {"Low": 0.01, "Medium": 0.03, "High": 0.06}

# The 8 features shared between the live engine and the trained model.
# MUST match SHARED_FEATURES in train_and_evaluate.py.
SHARED_FEATURES = [
    "flow_duration", "packet_count", "byte_count", "packet_rate",
    "byte_rate", "avg_packet_size", "iat_mean", "iat_std",
]


def load_supervised_model():
    """
    Load the RandomForest bundle saved by train_and_evaluate.py.
    Returns (bundle_dict, error_str). bundle is None if unavailable.
    """
    if not _HAS_JOBLIB:
        return None, "joblib not installed"
    if not os.path.exists(SUPERVISED_MODEL_PATH):
        return None, "no trained model (run train_and_evaluate.py)"
    try:
        bundle = joblib.load(SUPERVISED_MODEL_PATH)
        return bundle, None
    except Exception as e:
        return None, f"load failed: {e}"


# =========================================================
# CONFIG
# =========================================================

INTERFACE = r"\Device\NPF_{2C1A57ED-E9B5-4654-977B-8EF3BD4B6AEB}"
FLOW_TIMEOUT = 10
TRAINING_THRESHOLD = 20

client = Client(host="http://127.0.0.1:11434")
def list_interfaces():
    """
    Return [(label, sniff_value), ...] of capture interfaces, "Auto" first.
    sniff_value is what scapy's sniff(iface=...) accepts (None = default).
    Hardcoded INTERFACE GUIDs rarely match a given machine, so we let the
    user pick the right NIC at runtime.
    """
    out = [("Auto (default)", None)]
    try:
        from scapy.all import get_working_ifaces
        for it in get_working_ifaces():
            label = (getattr(it, "description", None) or getattr(it, "name", None)
                     or str(it))
            out.append((label, getattr(it, "name", str(it))))
    except Exception:
        try:
            from scapy.all import get_if_list
            for n in get_if_list():
                out.append((n, n))
        except Exception:
            pass
    return out

# Append-only audit trail (SIEM-friendly, one JSON object per line).
AUDIT_LOG_PATH = "threat_audit_log.jsonl"


# =========================================================
# MITRE ATT&CK CATALOG
#   Each technique we can map to, with its official ID/name/tactic and a
#   base severity weight (0-10) used by the CVSS-style scorer below.
# =========================================================

MITRE_TECHNIQUES = {
    "T1046": {
        "name": "Network Service Discovery",
        "tactic": "Discovery",
        "weight": 6.0,
        "desc": "Probing hosts/ports to enumerate services (scanning).",
    },
    "T1595": {
        "name": "Active Scanning",
        "tactic": "Reconnaissance",
        "weight": 5.0,
        "desc": "High-volume, low-payload probing of targets.",
    },
    "T1498": {
        "name": "Network Denial of Service",
        "tactic": "Impact",
        "weight": 8.0,
        "desc": "Flooding a target to exhaust bandwidth/resources.",
    },
    "T1048": {
        "name": "Exfiltration Over Alternative Protocol",
        "tactic": "Exfiltration",
        "weight": 7.5,
        "desc": "Sustained high-volume outbound transfer of data.",
    },
    "T1071": {
        "name": "Application Layer Protocol (C2 beaconing)",
        "tactic": "Command and Control",
        "weight": 7.0,
        "desc": "Highly regular, low-jitter timing typical of C2 beacons.",
    },
}


def severity_label(score):
    """Map a 0-10 CVSS-style score to a qualitative band."""
    if score >= 9.0:
        return "CRITICAL"
    if score >= 7.0:
        return "HIGH"
    if score >= 4.0:
        return "MEDIUM"
    if score >= 0.1:
        return "LOW"
    return "NONE"


def compute_evidence(features, feature_stats):
    """
    Z-score based evidence: for each numeric feature, compute how many
    standard deviations it sits from the learned baseline mean. Features
    with |z| >= 2 are 'statistically unusual' and become explainable proof.

    Returns a list of dicts sorted by |z| (strongest evidence first).
    """
    evidence = []
    if not feature_stats:
        return evidence
    for name, (mean, std) in feature_stats.items():
        val = features.get(name, 0)
        if std and std > 1e-9:
            z = (val - mean) / std
            if abs(z) >= 2.0:
                direction = "above" if z > 0 else "below"
                evidence.append({
                    "feature": name,
                    "value": round(float(val), 3),
                    "baseline_mean": round(float(mean), 3),
                    "z_score": round(float(z), 2),
                    "note": f"{name}={round(float(val), 2)} is {abs(z):.1f}σ "
                            f"{direction} normal",
                })
    evidence.sort(key=lambda e: abs(e["z_score"]), reverse=True)
    return evidence


def map_mitre(features, ml_result):
    """
    Heuristic mapping from observed telemetry to MITRE ATT&CK techniques.
    Only triggers when the ML model already flagged the flow as anomalous,
    so MITRE tags reinforce (not replace) the statistical verdict.
    """
    techniques = []
    if ml_result["prediction"] != "ANOMALOUS":
        return techniques

    f = features
    # --- Scanning: many connection attempts vs acknowledgements ---
    if f["syn_ratio"] >= 2.0 or (f["syn_count"] >= 5 and f["ack_count"] == 0):
        techniques.append("T1046")
    # --- Active scanning: lots of tiny, low-entropy packets ---
    if f["packet_count"] >= 40 and f["avg_packet_size"] < 120 and f["packet_entropy"] < 1.5:
        techniques.append("T1595")
    # --- DoS / flooding: very high packet rate ---
    if f["packet_rate"] >= 200:
        techniques.append("T1498")
    # --- Exfiltration: sustained high byte throughput ---
    if f["byte_rate"] >= 1_000_000 and f["byte_count"] >= 5_000_000:
        techniques.append("T1048")
    # --- C2 beaconing: very regular timing (low jitter) ---
    if f["packet_count"] >= 10 and f["iat_std"] < 0.01 and f["timing_entropy"] < 0.5:
        techniques.append("T1071")

    # de-duplicate while preserving order
    seen, ordered = set(), []
    for t in techniques:
        if t not in seen:
            seen.add(t)
            ordered.append(t)
    return ordered


def heuristic_detect(features):
    """
    Signature / threshold rules for BLATANT attacks, independent of the ML
    models. This is standard IDS practice: ML catches subtle/unknown threats,
    explicit rules catch obvious floods/scans that any analyst would flag.

    Thresholds are intentionally high so normal traffic does NOT trip them
    (typical browsing flows have far lower packet_rate / syn_ratio).

    Returns dict: {techniques, risk, severity, reasons} ('' risk if nothing).
    """
    f = features
    techniques, reasons = [], []
    risk, severity = "LOW RISK", 0.0

    def raise_to(r, s, t, why):
        nonlocal risk, severity
        order = {"LOW RISK": 0, "MEDIUM RISK": 1, "HIGH RISK": 2}
        if order[r] > order[risk]:
            risk = r
        severity = max(severity, s)
        if t and t not in techniques:
            techniques.append(t)
        reasons.append(why)

    # --- SYN flood / scan: overwhelming SYNs with no acknowledgements ---
    if f.get("syn_count", 0) >= 20 and f.get("syn_ratio", 0) >= 3:
        raise_to("HIGH RISK", 8.5, "T1046",
                 f"SYN flood/scan signature (syn={f['syn_count']}, ratio={f['syn_ratio']:.0f})")

    # --- Packet-rate flood (DoS): many small packets, very high rate ---
    if f.get("packet_rate", 0) >= 300 and f.get("packet_count", 0) >= 200 \
            and f.get("avg_packet_size", 9999) < 300:
        raise_to("HIGH RISK", 8.0, "T1498",
                 f"Flood signature (rate={f['packet_rate']:.0f}/s, "
                 f"pkts={f['packet_count']}, avg_size={f['avg_packet_size']:.0f}B)")

    # --- Data exfiltration: sustained very high outbound byte throughput ---
    if f.get("byte_rate", 0) >= 2_000_000 and f.get("byte_count", 0) >= 5_000_000:
        raise_to("MEDIUM RISK", 6.5, "T1048",
                 f"High-volume transfer (byte_rate={f['byte_rate']/1e6:.1f} MB/s)")

    # --- C2 beacon: clock-like timing, small repeated packets ---
    if f.get("packet_count", 0) >= 10 and f.get("iat_std", 9) < 0.05 \
            and f.get("timing_entropy", 9) < 0.6 and f.get("avg_packet_size", 9999) < 200:
        raise_to("MEDIUM RISK", 6.0, "T1071",
                 "Regular low-jitter beaconing pattern")

    return {"techniques": techniques, "risk": risk,
            "severity": severity, "reasons": reasons}


def score_severity(ml_result, techniques, evidence):
    """
    CVSS-style 0-10 severity plus a 0-1 confidence value.

    severity  = blend of (a) how far below the decision boundary the
                anomaly score sits and (b) the worst matched MITRE weight.
    confidence = grows with anomaly-score magnitude AND amount of
                 corroborating statistical evidence.
    """
    score = ml_result["anomaly_score"]

    # (a) anomaly magnitude component: negative scores => anomalous side
    anomaly_component = max(0.0, -score) * 25.0          # ~0..7
    # (b) strongest MITRE technique weight
    mitre_component = max([MITRE_TECHNIQUES[t]["weight"] for t in techniques], default=0.0)

    severity = min(10.0, max(anomaly_component, mitre_component))
    if ml_result["prediction"] == "NORMAL":
        severity = min(severity, 1.0)

    # confidence: combine score magnitude + evidence count + mitre corroboration
    conf = 0.45 + min(abs(score) * 1.2, 0.30)
    conf += min(len(evidence) * 0.04, 0.15)
    conf += 0.10 if techniques else 0.0
    confidence = round(min(conf, 0.99), 2)

    return round(severity, 1), confidence


# =========================================================
# FEATURE EXPLANATIONS
#   For each feature: (friendly label, what it means, why it matters)
#   Used by the detail panel to teach the user what each value represents.
# =========================================================

FEATURE_INFO = {
    "protocol": (
        "Protocol",
        "Transport protocol of the flow (TCP/UDP).",
        "TCP is connection-based; UDP is connectionless and common in floods/scans.",
    ),
    "flow_duration": (
        "Duration (s)",
        "How long the flow lasted, first to last packet.",
        "Very short bursts with many packets can indicate scanning or flooding.",
    ),
    "packet_count": (
        "Packet Count",
        "Total packets exchanged in the flow.",
        "Unusually high counts in a short time can signal flooding.",
    ),
    "byte_count": (
        "Byte Count",
        "Total bytes transferred.",
        "Large transfers may be downloads/exfiltration; tiny ones may be probes.",
    ),
    "packet_rate": (
        "Packet Rate (pkt/s)",
        "Packets per second = packet_count / duration.",
        "High packet rate is a classic flooding / DoS indicator.",
    ),
    "byte_rate": (
        "Byte Rate (B/s)",
        "Bytes per second = byte_count / duration.",
        "High byte rate = heavy throughput; near-zero with many packets = probing.",
    ),
    "avg_packet_size": (
        "Avg Packet Size (B)",
        "Mean size of packets in the flow.",
        "Tiny uniform packets often mean control/scan traffic, not real data.",
    ),
    "min_packet_size": (
        "Min Packet Size (B)",
        "Smallest packet observed.",
        "Many minimum-size packets suggest SYN scans / keepalives.",
    ),
    "max_packet_size": (
        "Max Packet Size (B)",
        "Largest packet observed.",
        "Large max with small avg = mixed control + bulk data.",
    ),
    "syn_count": (
        "SYN Count",
        "Number of TCP SYN (connection-open) packets.",
        "Many SYNs without matching ACKs is the signature of a SYN flood/scan.",
    ),
    "ack_count": (
        "ACK Count",
        "Number of TCP ACK (acknowledgement) packets.",
        "Healthy connections balance SYN and ACK.",
    ),
    "syn_ratio": (
        "SYN Ratio",
        "syn_count / (ack_count + 1).",
        ">1 means more connection attempts than acknowledgements -> scanning.",
    ),
    "iat_mean": (
        "IAT Mean (s)",
        "Average inter-arrival time between packets.",
        "Very small + very regular timing can indicate automated/bot traffic.",
    ),
    "iat_std": (
        "IAT Std (s)",
        "Variation in inter-arrival times.",
        "Near-zero variation = machine-like regularity; bursty = human/app traffic.",
    ),
    "packet_entropy": (
        "Packet Entropy",
        "Randomness of packet sizes (bits).",
        "Low = repetitive sizes (scans); high = varied real payloads.",
    ),
    "timing_entropy": (
        "Timing Entropy",
        "Randomness of inter-arrival times (bits).",
        "Low = clock-like timing (automation); high = irregular human traffic.",
    ),
    "port_entropy": (
        "Port Entropy",
        "Randomness across the src/dst ports.",
        "Context feature; varied ports across flows can hint at port scanning.",
    ),
}

# Order in which features are shown in the detail panel.
DETAIL_ORDER = [
    "protocol", "flow_duration", "packet_count", "byte_count",
    "packet_rate", "byte_rate", "avg_packet_size",
    "min_packet_size", "max_packet_size",
    "syn_count", "ack_count", "syn_ratio",
    "iat_mean", "iat_std",
    "packet_entropy", "timing_entropy", "port_entropy",
]


# =========================================================
# FEATURE ENGINE
# =========================================================

def shannon_entropy(values):
    """Shannon entropy: measures randomness of a list of values."""
    if not values:
        return 0
    counter = Counter(values)
    total = len(values)
    entropy = 0
    for count in counter.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


def safe_divide(a, b):
    """Divide a/b, returning 0 on divide-by-zero."""
    return 0 if b == 0 else a / b


def extract_features(flow):
    """Turn a raw flow dict into a structured feature vector."""
    duration = flow.get("duration", 0)
    packet_count = flow.get("packet_count", 0)
    byte_count = flow.get("byte_count", 0)
    syn_count = flow.get("syn_count", 0)
    ack_count = flow.get("ack_count", 0)
    packet_sizes = flow.get("packet_sizes", [])
    inter_arrival = flow.get("inter_arrival_times", [])

    packet_rate = safe_divide(packet_count, duration)
    byte_rate = safe_divide(byte_count, duration)
    avg_packet_size = safe_divide(flow["total_packet_size"], packet_count)
    syn_ratio = safe_divide(syn_count, ack_count + 1)

    if len(inter_arrival) > 0:
        iat_mean = statistics.mean(inter_arrival)
        iat_std = statistics.stdev(inter_arrival) if len(inter_arrival) > 1 else 0
    else:
        iat_mean = 0
        iat_std = 0

    return {
        "flow_id": flow["flow_id"],
        "src_ip": flow["src_ip"],
        "dst_ip": flow["dst_ip"],
        "src_port": flow["src_port"],
        "dst_port": flow["dst_port"],
        "protocol": flow["protocol"],
        "flow_duration": duration,
        "packet_count": packet_count,
        "byte_count": byte_count,
        "packet_rate": packet_rate,
        "byte_rate": byte_rate,
        "avg_packet_size": avg_packet_size,
        "min_packet_size": flow["min_packet_size"],
        "max_packet_size": flow["max_packet_size"],
        "syn_count": syn_count,
        "ack_count": ack_count,
        "syn_ratio": syn_ratio,
        "iat_mean": iat_mean,
        "iat_std": iat_std,
        "packet_entropy": shannon_entropy(packet_sizes),
        "timing_entropy": shannon_entropy([round(t, 3) for t in inter_arrival]),
        "port_entropy": shannon_entropy([flow["src_port"], flow["dst_port"]]),
    }


# =========================================================
# ML + LLM ENGINE
# =========================================================

class HybridThreatDetector:
    """IsolationForest anomaly detection + optional LLM reasoning + fusion."""

    FEATURE_NAMES = [
        "flow_duration", "packet_count", "byte_count",
        "packet_rate", "byte_rate", "avg_packet_size",
        "syn_ratio", "iat_mean", "iat_std",
        "packet_entropy", "timing_entropy", "port_entropy",
    ]

    def __init__(self, use_llm=True, baseline_n=TRAINING_THRESHOLD,
                 supervised_bundle=None, contamination=0.03,
                 baseline_path=BASELINE_PATH):
        self.contamination = contamination
        self.model = IsolationForest(
            n_estimators=150, contamination=contamination, random_state=42
        )
        self.training_data = []
        self.is_trained = False
        self.use_llm = use_llm and _HAS_OLLAMA
        # configurable number of baseline flows before IForest trains
        self.baseline_n = max(5, int(baseline_n))
        # evaluation results filled in after training
        self.eval_summary = None
        # per-feature baseline mean/std for z-score evidence
        self.feature_stats = {}
        # where this network's learned normal is persisted
        self.baseline_path = baseline_path
        self.baseline_source = "fresh"     # "fresh" | "loaded"

    # ---- baseline persistence (reuse this network's learned normal) ----
    def save_baseline(self):
        """Persist the trained IForest + baseline stats so it survives restarts."""
        if not (_HAS_JOBLIB and self.is_trained):
            return
        try:
            joblib.dump({
                "model": self.model,
                "feature_stats": self.feature_stats,
                "eval_summary": self.eval_summary,
                "baseline_n": self.baseline_n,
                "contamination": self.contamination,
                "feature_names": self.FEATURE_NAMES,
            }, self.baseline_path)
        except Exception:
            pass

    def load_baseline(self):
        """Load a previously saved baseline. Returns True on success."""
        if not (_HAS_JOBLIB and os.path.exists(self.baseline_path)):
            return False
        try:
            data = joblib.load(self.baseline_path)
            if data.get("feature_names") != self.FEATURE_NAMES:
                return False
            self.model = data["model"]
            self.feature_stats = data.get("feature_stats", {})
            self.eval_summary = data.get("eval_summary")
            self.contamination = data.get("contamination", self.contamination)
            self.is_trained = True
            self.baseline_source = "loaded"
            return True
        except Exception:
            return False

        # ---- supervised model (RandomForest trained on UNSW-NB15) ----
        self.supervised_bundle = supervised_bundle
        self.supervised_model = supervised_bundle["model"] if supervised_bundle else None
        self.supervised_features = (supervised_bundle["features"]
                                    if supervised_bundle else SHARED_FEATURES)
        # tuned low-false-positive threshold (proba >= threshold => ATTACK)
        self.supervised_threshold = (supervised_bundle.get("threshold", 0.5)
                                     if supervised_bundle else 0.5)

    def supervised_predict(self, features):
        """
        Classify a flow with the trained RandomForest using the shared
        feature subset. Uses the tuned low-FPR threshold instead of 0.5 so
        real benign traffic is not constantly flagged.
        Returns (label_str, attack_probability) or None.
        """
        # defensive getattr so an out-of-sync copy can't crash the analysis thread
        model = getattr(self, "supervised_model", None)
        if model is None:
            return None
        feats = getattr(self, "supervised_features", SHARED_FEATURES)
        threshold = getattr(self, "supervised_threshold", 0.5)
        try:
            df = pd.DataFrame([[features[name] for name in feats]], columns=feats)
            proba = float(model.predict_proba(df)[0][1])
            label = "ATTACK" if proba >= threshold else "BENIGN"
            return label, round(proba, 4)
        except Exception:
            return None

    def prepare_features(self, f):
        return [f[name] for name in self.FEATURE_NAMES]

    # ---- training ----
    def train(self):
        if len(self.training_data) < self.baseline_n:
            return False
        df = pd.DataFrame(self.training_data, columns=self.FEATURE_NAMES)
        self.model.fit(df)
        self.is_trained = True
        self.baseline_source = "fresh"
        self._evaluate(df)
        self.save_baseline()        # persist so it is reused next run
        return True

    def _evaluate(self, df):
        """Compute unsupervised evaluation summary right after training."""
        preds = self.model.predict(df)
        scores = self.model.decision_function(df)
        self.eval_summary = {
            "samples": len(preds),
            "normal": int((preds == 1).sum()),
            "anomalous": int((preds == -1).sum()),
            "anomaly_rate": safe_divide(int((preds == -1).sum()), len(preds)),
            "score_min": float(scores.min()),
            "score_max": float(scores.max()),
            "score_mean": float(scores.mean()),
            "score_std": float(scores.std()),
        }
        # learn baseline mean/std per feature (used for evidence z-scores)
        self.feature_stats = {
            name: (float(df[name].mean()), float(df[name].std(ddof=0)))
            for name in self.FEATURE_NAMES
        }

    # ---- detection ----
    def _score_flow(self, features):
        """Run the trained IsolationForest on one flow (assumes is_trained)."""
        df = pd.DataFrame([self.prepare_features(features)], columns=self.FEATURE_NAMES)
        prediction = self.model.predict(df)[0]
        score = self.model.decision_function(df)[0]
        return {
            "prediction": "ANOMALOUS" if prediction == -1 else "NORMAL",
            "anomaly_score": round(float(score), 4),
        }

    def detect_anomaly(self, features):
        """Back-compat: collect baseline while training, else score the flow."""
        if not self.is_trained:
            self.training_data.append(self.prepare_features(features))
            self.train()
            return None
        return self._score_flow(features)

    # ---- LLM ----
    def build_context(self, f, ml):
        return f"""
You are analyzing behavioral network telemetry.

Protocol: {f['protocol']}
Flow Duration: {round(f['flow_duration'], 2)} sec
Packet Count: {f['packet_count']}
Packet Rate: {round(f['packet_rate'], 2)}
Byte Rate: {round(f['byte_rate'], 2)}
Average Packet Size: {round(f['avg_packet_size'], 2)}
SYN Ratio: {round(f['syn_ratio'], 2)}
Packet Entropy: {round(f['packet_entropy'], 2)}
Timing Entropy: {round(f['timing_entropy'], 2)}
ML Prediction: {ml['prediction']}
Anomaly Score: {ml['anomaly_score']}

Classify this behavior as: NORMAL, SLIGHTLY SUSPICIOUS, or HIGHLY SUSPICIOUS.
Explain ONLY using telemetry evidence. Do NOT invent attacks. Be concise.
"""

    from ollama import Client

    client = Client(host="http://127.0.0.1:11434")

    def llm_reasoning(self, context):
        if not self.use_llm:
            return (
                "LLM disabled / unavailable. Verdict is ML-only: "
                "the flow's feature pattern deviates from the learned normal baseline."
            )

        try:
            resp = client.chat(
                model="phi3",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert cybersecurity analyst. Analyze network telemetry only."
                    },
                    {
                        "role": "user",
                        "content": context
                    },
                ],
            )

            return resp.message.content

        except Exception as e:
            return f"LLM unavailable ({type(e).__name__}: {e}). Using ML-only analysis."

    def fusion_decision(self, ml, llm_output):
        low = llm_output.lower()
        if ml["prediction"] == "ANOMALOUS" and (
            "highly suspicious" in low or "scanning" in low or "flooding" in low
        ):
            return "HIGH RISK"
        if ml["prediction"] == "ANOMALOUS" and "slightly suspicious" in low:
            return "MEDIUM RISK"
        return "LOW RISK"

    def analyze(self, features):
        # --- signature/threshold rules: work WITH OR WITHOUT a baseline ---
        rules = heuristic_detect(features)
        rule_flag = bool(rules["techniques"])

        # --- IsolationForest: learn baseline from NORMAL flows only ---
        if not self.is_trained:
            if not rule_flag:                      # never train on obvious attacks
                self.training_data.append(self.prepare_features(features))
                self.train()
            ml = None
        else:
            ml = self._score_flow(features)

        iforest_flag = (ml is not None) and (ml["prediction"] == "ANOMALOUS")
        anomaly_score = ml["anomaly_score"] if ml else 0.0

        # --- supervised verdict (independent of the baseline) ---
        sup = self.supervised_predict(features)            # (label, proba) or None
        sup_label = sup[0] if sup else None
        sup_proba = sup[1] if sup else None
        supervised_flag = sup_label == "ATTACK"

        # still building the baseline AND nothing flags this flow -> no verdict yet
        if (ml is None) and (not rule_flag) and (not supervised_flag):
            return None

        # --- evidence + MITRE ---
        evidence = compute_evidence(features, self.feature_stats)
        techniques = list(rules["techniques"])
        if iforest_flag or supervised_flag:        # add model-derived MITRE tags
            for t in map_mitre(features, {"prediction": "ANOMALOUS"}):
                if t not in techniques:
                    techniques.append(t)
        mitre_detail = [
            {"id": t, **{k: MITRE_TECHNIQUES[t][k] for k in ("name", "tactic", "desc")}}
            for t in techniques
        ]

        # =====================================================
        # FUSION  (calm by default -- LOW unless there's real evidence)
        #   rule_flag           -> explicit attack signature -> its own risk
        #   iforest AND superv. -> both ML models agree       -> HIGH
        #   supervised (>=0.90) -> trained model very sure     -> MEDIUM
        #   anything else (incl. IForest alone)                -> LOW
        #   (IForest alone is NOT enough to alert -- it is the noisiest signal
        #    and was the cause of "MEDIUM by default".)
        # =====================================================
        if rule_flag:
            risk = rules["risk"]
            base_sev = rules["severity"]
        elif iforest_flag and supervised_flag:
            risk = "HIGH RISK"
            base_sev = 8.0
        elif supervised_flag and (sup_proba or 0) >= 0.90:
            risk = "MEDIUM RISK"
            base_sev = 5.0
        else:
            risk = "LOW RISK"
            base_sev = 1.0

        mitre_w = max([MITRE_TECHNIQUES[t]["weight"] for t in techniques], default=0.0)
        proba_boost = (sup_proba if supervised_flag else 0.0) * 2.0
        severity = round(min(10.0, base_sev + proba_boost * 0.5 +
                             (mitre_w - base_sev if mitre_w > base_sev else 0) * 0.3), 1)

        if rule_flag:
            confidence = 0.95
        elif iforest_flag and supervised_flag:
            confidence = round(min(0.99, 0.6 + (sup_proba or 0) * 0.3), 2)
        elif supervised_flag:
            confidence = round(0.5 + (sup_proba or 0) * 0.3, 2)
        else:
            confidence = 0.9   # confident it's benign

        # --- LOW risk: short-circuit (skip slow LLM) ---
        if risk == "LOW RISK":
            note = ("Signal too weak to alert (calm baseline)."
                    if (iforest_flag or supervised_flag)
                    else "Behavior is consistent with normal traffic.")
            return {
                "ml_prediction": "NORMAL", "anomaly_score": anomaly_score,
                "supervised": sup_label, "supervised_proba": sup_proba,
                "risk_level": "LOW RISK", "severity": severity,
                "severity_label": severity_label(severity), "confidence": confidence,
                "mitre": mitre_detail, "evidence": evidence, "llm_analysis": note,
            }

        # rule hits -> instant signature explanation; ML-only -> LLM reasoning
        if rule_flag:
            llm_output = "RULE-BASED DETECTION:\n - " + "\n - ".join(rules["reasons"])
        elif ml is not None:
            llm_output = self.llm_reasoning(self.build_context(features, ml))
        else:
            llm_output = f"Supervised model flagged this flow (attack probability {sup_proba:.0%})."

        return {
            "ml_prediction": "ANOMALOUS",
            "anomaly_score": anomaly_score,
            "supervised": sup_label,
            "supervised_proba": sup_proba,
            "risk_level": risk,
            "severity": severity,
            "severity_label": severity_label(severity),
            "confidence": confidence,
            "mitre": mitre_detail,
            "evidence": evidence,
            "llm_analysis": llm_output,
        }


# =========================================================
# CAPTURE BACKEND
#   Encapsulates packet capture / flow processing in background threads.
#   Pushes (features, result) tuples onto an output queue for the GUI.
# =========================================================

class CaptureEngine:
    def __init__(self, detector, output_queue, status_cb, mode="demo", iface=None):
        self.detector = detector
        self.output_queue = output_queue   # (features, result_or_None)
        self.status_cb = status_cb          # callback(str) for status text
        self.mode = mode                    # "demo" or "live"
        self.iface = iface                  # scapy interface to sniff (None=default)
        self.running = False
        self.captured = 0                   # raw packets seen (capture health)

        self.packet_queue = Queue()
        self.completed_flows = Queue()
        self.flows = {}
        self._threads = []

    # ---- lifecycle ----
    def start(self):
        if self.running:
            return
        self.running = True
        if self.mode == "live":
            self._start_live()
        else:
            self._start_demo()
        # common processing + analysis threads
        self._spawn(self._process_flows)
        self._spawn(self._analyze_completed)

    def stop(self):
        self.running = False

    def _spawn(self, target):
        t = threading.Thread(target=target, daemon=True)
        t.start()
        self._threads.append(t)

    # ---- LIVE capture (scapy) ----
    def _start_live(self):
        try:
            from scapy.all import sniff, IP, TCP, UDP
        except Exception as e:
            self.status_cb(f"scapy unavailable ({e}); switch to DEMO mode.")
            self.running = False
            return
        self._scapy = (IP, TCP, UDP)

        def _sniff():
            kwargs = dict(filter="ip", prn=self._on_packet, store=False,
                          stop_filter=lambda p: not self.running)
            if self.iface:                 # None => scapy default interface
                kwargs["iface"] = self.iface
            try:
                self.status_cb(f"Sniffing on: {self.iface or 'default'} ...")
                sniff(**kwargs)
            except Exception as e:
                self.status_cb(f"Capture failed on '{self.iface}': {e}")
                self.running = False
        self._spawn(_sniff)

    def _on_packet(self, packet):
        self.captured += 1
        IP, TCP, UDP = self._scapy
        if not packet.haslayer(IP):
            return
        src_ip, dst_ip = packet[IP].src, packet[IP].dst
        if dst_ip.startswith(("224.", "239.")) or dst_ip.endswith(".255"):
            return
        parsed = {
            "timestamp": float(packet.time), "src_ip": src_ip, "dst_ip": dst_ip,
            "protocol": "OTHER", "src_port": None, "dst_port": None,
            "packet_size": len(packet), "flags": None,
        }
        if packet.haslayer(TCP):
            parsed.update(protocol="TCP", src_port=packet[TCP].sport,
                          dst_port=packet[TCP].dport, flags=str(packet[TCP].flags))
        elif packet.haslayer(UDP):
            parsed.update(protocol="UDP", src_port=packet[UDP].sport,
                          dst_port=packet[UDP].dport)
        if parsed["protocol"] == "OTHER" or parsed["src_port"] is None:
            return
        self.packet_queue.put(parsed)

    # ---- DEMO capture (synthetic packets) ----
    def _start_demo(self):
        self._spawn(self._demo_traffic)

    def _demo_traffic(self):
        """Generate synthetic packets: mostly normal, occasionally an attack."""
        rng = random.Random(7)
        while self.running:
            attack = rng.random() < 0.18  # ~18% suspicious flows
            src_ip = f"192.168.1.{rng.randint(2, 50)}"
            dst_ip = f"10.0.0.{rng.randint(2, 50)}"
            src_port = rng.randint(1024, 65535)
            dst_port = rng.choice([80, 443, 22, 53, rng.randint(1, 1024)])
            proto = rng.choice(["TCP", "TCP", "UDP"])

            if attack:
                # flood/scan: many tiny packets, fast, lots of SYN
                n = rng.randint(60, 200)
                base = 60
                gap = 0.001
                flags_seq = "S"
            else:
                # normal: moderate packets, varied sizes, balanced flags
                n = rng.randint(5, 40)
                base = rng.randint(200, 1200)
                gap = rng.uniform(0.02, 0.3)
                flags_seq = "SA"

            now = time.time()
            for i in range(n):
                size = base + (rng.randint(-30, 400) if not attack else rng.randint(-5, 5))
                size = max(40, size)
                flag = "S" if (attack and i == 0) else rng.choice(flags_seq)
                self.packet_queue.put({
                    "timestamp": now + i * gap,
                    "src_ip": src_ip, "dst_ip": dst_ip,
                    "protocol": proto, "src_port": src_port, "dst_port": dst_port,
                    "packet_size": size, "flags": flag if proto == "TCP" else None,
                })
            # force this synthetic flow to expire quickly
            time.sleep(rng.uniform(0.4, 1.2))
            self._force_expire_all()

    def _force_expire_all(self):
        """Demo helper: close every active flow immediately."""
        for key in list(self.flows.keys()):
            flow = self.flows.pop(key, None)
            if flow:
                flow["duration"] = max(flow["last_seen"] - flow["start_time"], 0.001)
                self.completed_flows.put(flow)

    # ---- flow grouping ----
    @staticmethod
    def _flow_key(p):
        e1 = (p["src_ip"], p["src_port"])
        e2 = (p["dst_ip"], p["dst_port"])
        a, b = sorted([e1, e2])
        return (a, b, p["protocol"])

    def _new_flow(self, p):
        now = p["timestamp"]
        flow = {
            "flow_id": str(hash(str(p) + str(now)))[:12],
            "src_ip": p["src_ip"], "dst_ip": p["dst_ip"],
            "src_port": p["src_port"], "dst_port": p["dst_port"],
            "protocol": p["protocol"], "start_time": now, "last_seen": now,
            "packet_count": 1, "byte_count": p["packet_size"],
            "syn_count": 0, "ack_count": 0,
            "min_packet_size": p["packet_size"], "max_packet_size": p["packet_size"],
            "total_packet_size": p["packet_size"],
            "packet_sizes": [p["packet_size"]], "inter_arrival_times": [],
        }
        self._count_flags(flow, p)
        return flow

    def _update_flow(self, flow, p):
        now = p["timestamp"]
        flow["inter_arrival_times"].append(now - flow["last_seen"])
        flow["last_seen"] = now
        flow["packet_count"] += 1
        size = p["packet_size"]
        flow["byte_count"] += size
        flow["total_packet_size"] += size
        flow["packet_sizes"].append(size)
        flow["min_packet_size"] = min(flow["min_packet_size"], size)
        flow["max_packet_size"] = max(flow["max_packet_size"], size)
        self._count_flags(flow, p)

    @staticmethod
    def _count_flags(flow, p):
        if p["flags"]:
            if "S" in p["flags"]:
                flow["syn_count"] += 1
            if "A" in p["flags"]:
                flow["ack_count"] += 1

    def _process_flows(self):
        while self.running:
            while not self.packet_queue.empty():
                p = self.packet_queue.get()
                key = self._flow_key(p)
                if key not in self.flows:
                    self.flows[key] = self._new_flow(p)
                else:
                    self._update_flow(self.flows[key], p)
            if self.mode == "live":
                self._close_expired()
            extra = f" | captured pkts: {self.captured}" if self.mode == "live" else ""
            self.status_cb(f"Active flows: {len(self.flows)}{extra}")
            time.sleep(0.1)

    def _close_expired(self):
        now = time.time()
        for key in [k for k, f in self.flows.items()
                    if now - f["last_seen"] > FLOW_TIMEOUT]:
            flow = self.flows.pop(key)
            flow["duration"] = flow["last_seen"] - flow["start_time"]
            self.completed_flows.put(flow)

    # ---- analysis ----
    def _analyze_completed(self):
        while self.running:
            try:
                flow = self.completed_flows.get(timeout=0.5)
            except Empty:
                continue
            features = extract_features(flow)
            result = self.detector.analyze(features)
            self.output_queue.put((features, result))


# =========================================================
# TKINTER GUI
# =========================================================

RISK_COLORS = {
    "HIGH RISK": "#ff5252",
    "MEDIUM RISK": "#ffb74d",
    "LOW RISK": "#81c784",
    "TRAINING": "#90caf9",
}


class ThreatDashboard(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Hybrid Network Threat Detector")
        self.geometry("1180x720")
        self.configure(bg="#1e1e2e")

        self.gui_queue = Queue()         # (features, result) from backend
        self.detector = None
        self.engine = None
        self.rows = {}                   # tree row id -> (features, result)
        self.total_analyzed = 0

        # ---- aggregate ML statistics (for the ML Analysis tab) ----
        self.outcome_counts = {           # running outcome tally
            "NORMAL": 0, "ANOMALOUS": 0, "TRAINING": 0,
            "LOW RISK": 0, "MEDIUM RISK": 0, "HIGH RISK": 0,
        }
        self.recent_scores = []           # last N anomaly scores (for chart)
        self.MAX_CHART_POINTS = 80

        # ---- time-stamped event history (for charts + chatbot queries) ----
        self.flow_history = []            # list of dicts with 'ts' + verdict fields
        self.proto_counts = Counter()     # protocol -> count
        self.chat_queue = Queue()         # (role, text) answers from LLM thread

        # ---- load the trained supervised model + its evaluation metrics ----
        self.model_bundle, self.model_err = load_supervised_model()
        self.model_metrics = self._load_model_metrics()

        self._build_styles()
        self._build_header()
        self._build_body()
        self._build_footer()

        # start polling the backend queue on the Tk main thread
        self.after(200, self._poll_queue)

    # ---------- styling ----------
    def _build_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Treeview",
                        background="#2a2a3c", foreground="#e0e0e0",
                        fieldbackground="#2a2a3c", rowheight=24)
        style.configure("Treeview.Heading",
                        background="#3a3a4c", foreground="#ffffff")
        style.map("Treeview", background=[("selected", "#4a4a6a")])

    # ---------- header / controls (two rows) ----------
    def _build_header(self):
        # ===== ROW 1: title + configuration =====
        bar = tk.Frame(self, bg="#181825")
        bar.pack(fill="x", padx=8, pady=(6, 0))

        tk.Label(bar, text="HYBRID NETWORK THREAT DETECTOR",
                 fg="#cba6f7", bg="#181825",
                 font=("Segoe UI", 14, "bold")).pack(side="left", padx=8)

        self.mode_var = tk.StringVar(value="demo")
        tk.Label(bar, text="Mode:", fg="#cdd6f4", bg="#181825").pack(side="left", padx=(16, 2))
        ttk.Combobox(bar, textvariable=self.mode_var, width=6,
                     values=["demo", "live"], state="readonly").pack(side="left")

        # capture interface selector (LIVE mode) - replaces the hardcoded GUID
        self._iface_map = dict(list_interfaces())
        tk.Label(bar, text="Iface:", fg="#cdd6f4", bg="#181825").pack(side="left", padx=(10, 2))
        self.iface_var = tk.StringVar(value=next(iter(self._iface_map)))
        ttk.Combobox(bar, textvariable=self.iface_var, width=20, state="readonly",
                     values=list(self._iface_map)).pack(side="left")

        self.llm_var = tk.BooleanVar(value=_HAS_OLLAMA)
        tk.Checkbutton(bar, text="Use LLM", variable=self.llm_var,
                       fg="#cdd6f4", bg="#181825", selectcolor="#313244",
                       activebackground="#181825", activeforeground="#cdd6f4"
                       ).pack(side="left", padx=10)

        # trained-model status indicator (right side of row 1)
        model_txt = ("Model: RandomForest loaded" if self.model_bundle
                     else f"Model: none ({self.model_err})")
        model_color = "#a6e3a1" if self.model_bundle else "#f38ba8"
        tk.Label(bar, text=model_txt, fg=model_color, bg="#181825",
                 font=("Segoe UI", 9)).pack(side="right", padx=10)

        # ===== ROW 2: actions + tuning =====
        bar2 = tk.Frame(self, bg="#181825")
        bar2.pack(fill="x", padx=8, pady=(2, 6))

        self.start_btn = tk.Button(bar2, text="Start", width=8, command=self.start,
                                   bg="#a6e3a1", fg="#1e1e2e", relief="flat")
        self.start_btn.pack(side="left", padx=4)
        self.stop_btn = tk.Button(bar2, text="Stop", width=8, command=self.stop,
                                  bg="#f38ba8", fg="#1e1e2e", relief="flat",
                                  state="disabled")
        self.stop_btn.pack(side="left", padx=4)

        # configurable IsolationForest baseline size
        tk.Label(bar2, text="Baseline:", fg="#cdd6f4", bg="#181825").pack(side="left", padx=(16, 2))
        self.baseline_var = tk.IntVar(value=TRAINING_THRESHOLD)
        tk.Spinbox(bar2, from_=5, to=1000, increment=5, width=5,
                   textvariable=self.baseline_var).pack(side="left")

        # IsolationForest sensitivity (contamination)
        tk.Label(bar2, text="Sens:", fg="#cdd6f4", bg="#181825").pack(side="left", padx=(10, 2))
        self.sens_var = tk.StringVar(value="Medium")
        ttk.Combobox(bar2, textvariable=self.sens_var, width=7, state="readonly",
                     values=list(SENSITIVITY_PRESETS)).pack(side="left")

        # reuse the saved baseline (this network's learned normal)
        self.reuse_var = tk.BooleanVar(value=os.path.exists(BASELINE_PATH))
        tk.Checkbutton(bar2, text="Reuse baseline", variable=self.reuse_var,
                       fg="#cdd6f4", bg="#181825", selectcolor="#313244",
                       activebackground="#181825", activeforeground="#cdd6f4"
                       ).pack(side="left", padx=8)

        self.baseline_btn = tk.Button(bar2, text="Run Baseline", width=11,
                                      command=self.run_baseline,
                                      bg="#f9e2af", fg="#1e1e2e", relief="flat")
        self.baseline_btn.pack(side="left", padx=4)

        # ---- quick inject + full attack simulator popup ----
        tk.Label(bar2, text="Inject:", fg="#cdd6f4", bg="#181825").pack(side="left", padx=(12, 2))
        self.inject_var = tk.StringVar(value="UDP Flood")
        ttk.Combobox(bar2, textvariable=self.inject_var, width=12, state="readonly",
                     values=["UDP Flood", "SYN Flood", "Port Scan",
                             "Data Exfil", "C2 Beacon", "Normal"]).pack(side="left")
        self.inject_btn = tk.Button(bar2, text="Inject", width=7,
                                    command=self.inject_attack,
                                    bg="#eba0ac", fg="#1e1e2e", relief="flat")
        self.inject_btn.pack(side="left", padx=4)
        self.sim_btn = tk.Button(bar2, text="Attack Sim...", width=11,
                                 command=self.open_attack_sim,
                                 bg="#cba6f7", fg="#1e1e2e", relief="flat")
        self.sim_btn.pack(side="left", padx=4)

        self.export_btn = tk.Button(bar2, text="Export CSV", width=10,
                                    command=self.export_csv,
                                    bg="#89b4fa", fg="#1e1e2e", relief="flat")
        self.export_btn.pack(side="right", padx=8)

    # ---------- main body (tabbed) ----------
    def _build_body(self):
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=8, pady=4)

        self.monitor_tab = tk.Frame(self.notebook, bg="#1e1e2e")
        self.ml_tab = tk.Frame(self.notebook, bg="#1e1e2e")
        self.assistant_tab = tk.Frame(self.notebook, bg="#1e1e2e")
        self.notebook.add(self.monitor_tab, text="  Live Monitor  ")
        self.notebook.add(self.ml_tab, text="  ML Analysis  ")
        self.notebook.add(self.assistant_tab, text="  Assistant  ")

        self._build_monitor_tab(self.monitor_tab)
        self._build_ml_tab(self.ml_tab)
        self._build_assistant_tab(self.assistant_tab)

    # ---------- TAB 1: live monitor ----------
    def _build_monitor_tab(self, parent):
        body = tk.Frame(parent, bg="#1e1e2e")
        body.pack(fill="both", expand=True, padx=4, pady=4)

        # left: flow table
        left = tk.Frame(body, bg="#1e1e2e")
        left.pack(side="left", fill="both", expand=True)

        tk.Label(left, text="Analyzed Flows (click a row for details)",
                 fg="#cdd6f4", bg="#1e1e2e", font=("Segoe UI", 10, "bold")
                 ).pack(anchor="w")

        cols = ("time", "src", "dst", "proto", "pred", "score", "sev", "mitre", "risk")
        self.tree = ttk.Treeview(left, columns=cols, show="headings", height=22)
        headings = {
            "time": ("Time", 65), "src": ("Source", 135), "dst": ("Dest", 135),
            "proto": ("Proto", 50), "pred": ("ML", 85),
            "score": ("Score", 55), "sev": ("Severity", 90),
            "mitre": ("MITRE", 110), "risk": ("Risk", 95),
        }
        for c, (txt, w) in headings.items():
            self.tree.heading(c, text=txt)
            self.tree.column(c, width=w, anchor="center")
        self.tree.pack(fill="both", expand=True)
        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        # risk color tags
        for risk, color in RISK_COLORS.items():
            self.tree.tag_configure(risk, foreground=color)

        # right: detail + LLM
        right = tk.Frame(body, bg="#1e1e2e", width=460)
        right.pack(side="right", fill="both")
        right.pack_propagate(False)

        tk.Label(right, text="Flow Parameters & What They Mean",
                 fg="#cdd6f4", bg="#1e1e2e", font=("Segoe UI", 10, "bold")
                 ).pack(anchor="w")

        dcols = ("param", "value", "meaning")
        self.detail = ttk.Treeview(right, columns=dcols, show="headings", height=18)
        self.detail.heading("param", text="Parameter")
        self.detail.heading("value", text="Value")
        self.detail.heading("meaning", text="Meaning / Why it matters")
        self.detail.column("param", width=130, anchor="w")
        self.detail.column("value", width=80, anchor="center")
        self.detail.column("meaning", width=240, anchor="w")
        self.detail.pack(fill="x")

        tk.Label(right, text="LLM / AI Analysis",
                 fg="#cdd6f4", bg="#1e1e2e", font=("Segoe UI", 10, "bold")
                 ).pack(anchor="w", pady=(8, 0))
        self.llm_box = scrolledtext.ScrolledText(
            right, height=9, bg="#2a2a3c", fg="#e0e0e0",
            wrap="word", font=("Consolas", 9))
        self.llm_box.pack(fill="both", expand=True)

    # ---------- TAB 2: ML analysis & outcomes ----------
    def _build_ml_tab(self, parent):
        wrap = tk.Frame(parent, bg="#1e1e2e")
        wrap.pack(fill="both", expand=True, padx=8, pady=8)

        # --- row of outcome "stat cards" ---
        cards = tk.Frame(wrap, bg="#1e1e2e")
        cards.pack(fill="x")

        self.stat_labels = {}
        card_specs = [
            ("NORMAL", "Normal flows", "#81c784"),
            ("ANOMALOUS", "Anomalous flows", "#ff8a80"),
            ("LOW RISK", "Low risk", "#81c784"),
            ("MEDIUM RISK", "Medium risk", "#ffb74d"),
            ("HIGH RISK", "High risk", "#ff5252"),
        ]
        for key, title, color in card_specs:
            card = tk.Frame(cards, bg="#2a2a3c", bd=0, relief="flat")
            card.pack(side="left", expand=True, fill="x", padx=4)
            tk.Label(card, text=title, fg="#cdd6f4", bg="#2a2a3c",
                     font=("Segoe UI", 9)).pack(pady=(8, 0))
            lbl = tk.Label(card, text="0", fg=color, bg="#2a2a3c",
                           font=("Segoe UI", 20, "bold"))
            lbl.pack(pady=(0, 8))
            self.stat_labels[key] = lbl

        # --- two columns: model report (left) + live score chart (right) ---
        mid = tk.Frame(wrap, bg="#1e1e2e")
        mid.pack(fill="both", expand=True, pady=(10, 0))

        # left: model evaluation report
        left = tk.Frame(mid, bg="#1e1e2e")
        left.pack(side="left", fill="both", expand=True)

        tk.Label(left, text="Model Status & Evaluation",
                 fg="#cdd6f4", bg="#1e1e2e",
                 font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.ml_report = scrolledtext.ScrolledText(
            left, bg="#2a2a3c", fg="#e0e0e0", wrap="word",
            font=("Consolas", 9), height=16)
        self.ml_report.pack(fill="both", expand=True, pady=(2, 0))

        # right: live anomaly-score chart
        right = tk.Frame(mid, bg="#1e1e2e", width=480)
        right.pack(side="right", fill="both")
        right.pack_propagate(False)

        tk.Label(right, text="Live Anomaly Scores  (red line = decision boundary)",
                 fg="#cdd6f4", bg="#1e1e2e",
                 font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.chart = tk.Canvas(right, bg="#11111b", highlightthickness=0, height=230)
        self.chart.pack(fill="both", expand=True, pady=(2, 0))

        # --- bottom row: three live mini charts ---
        bottom = tk.Frame(wrap, bg="#1e1e2e", height=210)
        bottom.pack(fill="x", pady=(10, 0))
        bottom.pack_propagate(False)

        def _mini(parent, title):
            f = tk.Frame(parent, bg="#1e1e2e")
            f.pack(side="left", fill="both", expand=True, padx=4)
            tk.Label(f, text=title, fg="#cdd6f4", bg="#1e1e2e",
                     font=("Segoe UI", 9, "bold")).pack(anchor="w")
            cv = tk.Canvas(f, bg="#11111b", highlightthickness=0, height=180)
            cv.pack(fill="both", expand=True, pady=(2, 0))
            return cv

        self.chart_risk = _mini(bottom, "Risk Distribution")
        self.chart_proto = _mini(bottom, "Protocol Mix")
        self.chart_timeline = _mini(bottom, "Flows / minute (last 15)")

        self._render_ml_report()

    # ---------- TAB 3: assistant / chatbot ----------
    def _build_assistant_tab(self, parent):
        wrap = tk.Frame(parent, bg="#1e1e2e")
        wrap.pack(fill="both", expand=True, padx=10, pady=8)

        tk.Label(wrap, text="Network Assistant  -  ask about your traffic",
                 fg="#cba6f7", bg="#1e1e2e",
                 font=("Segoe UI", 12, "bold")).pack(anchor="w")
        engine = "LLM (Ollama)" if _HAS_OLLAMA else "built-in stats engine"
        tk.Label(wrap, text=f"Answers from: {engine}.  Data is live from this session.",
                 fg="#a6adc8", bg="#1e1e2e", font=("Segoe UI", 8)).pack(anchor="w")

        # conversation view
        self.chat_view = scrolledtext.ScrolledText(
            wrap, bg="#11111b", fg="#e0e0e0", wrap="word",
            font=("Segoe UI", 10), height=20, state="disabled")
        self.chat_view.pack(fill="both", expand=True, pady=(6, 4))
        self.chat_view.tag_config("user", foreground="#89b4fa", font=("Segoe UI", 10, "bold"))
        self.chat_view.tag_config("bot", foreground="#a6e3a1")
        self.chat_view.tag_config("sys", foreground="#6c7086", font=("Segoe UI", 9, "italic"))

        # quick-question buttons
        quick = tk.Frame(wrap, bg="#1e1e2e")
        quick.pack(fill="x", pady=(2, 4))
        for label in ["Last hour stats", "Current conditions", "Top talkers",
                      "Any threats?", "Last 5 minutes"]:
            tk.Button(quick, text=label, relief="flat", bg="#313244", fg="#cdd6f4",
                      activebackground="#45475a",
                      command=lambda q=label: self._ask(q)).pack(side="left", padx=3)

        # input row
        row = tk.Frame(wrap, bg="#1e1e2e")
        row.pack(fill="x")
        self.chat_entry = tk.Entry(row, bg="#2a2a3c", fg="#e0e0e0",
                                   insertbackground="#e0e0e0", font=("Segoe UI", 10))
        self.chat_entry.pack(side="left", fill="x", expand=True, ipady=4)
        self.chat_entry.bind("<Return>", lambda e: self._ask(self.chat_entry.get()))
        tk.Button(row, text="Send", width=8, bg="#89b4fa", fg="#1e1e2e", relief="flat",
                  command=lambda: self._ask(self.chat_entry.get())).pack(side="left", padx=(6, 0))

        self._chat_say("sys", "Assistant ready. Try a quick button or ask e.g. "
                              "\"what happened in the last hour?\"")

    # ---------- chat plumbing ----------
    def _chat_say(self, role, text):
        self.chat_view.config(state="normal")
        prefix = {"user": "You: ", "bot": "Assistant: ", "sys": ""}.get(role, "")
        self.chat_view.insert("end", prefix, role)
        self.chat_view.insert("end", text + "\n\n")
        self.chat_view.config(state="disabled")
        self.chat_view.see("end")

    def _ask(self, question):
        question = (question or "").strip()
        if not question:
            return
        self.chat_entry.delete(0, "end")
        self._chat_say("user", question)
        snapshot = self._network_snapshot()
        if _HAS_OLLAMA:
            self._chat_say("sys", "thinking...")
            threading.Thread(target=self._llm_answer, args=(question, snapshot),
                             daemon=True).start()
        else:
            self._chat_say("bot", self._fallback_answer(question, snapshot))

    def _llm_answer(self, question, snapshot):
        """Run the LLM in a background thread; post result via chat_queue."""
        try:
            resp = ollama.chat(
                model="phi3",
                messages=[
                    {"role": "system", "content":
                        "You are a network monitoring assistant. Answer the user's "
                        "question ONLY using the telemetry snapshot provided. Be concise "
                        "and concrete with numbers. If the snapshot lacks the data, say so."},
                    {"role": "user", "content":
                        f"TELEMETRY SNAPSHOT:\n{snapshot}\n\nQUESTION: {question}"},
                ])
            self.chat_queue.put(("bot", resp["message"]["content"].strip()))
        except Exception as e:
            self.chat_queue.put(("bot", f"(LLM error: {e})\n\n" +
                                 self._fallback_answer(question, snapshot)))

    # ---------- stats engine (powers charts + chatbot) ----------
    def _window_events(self, seconds):
        """Return history events within the last `seconds` (None = all)."""
        if seconds is None:
            return list(self.flow_history)
        cutoff = time.time() - seconds
        return [e for e in self.flow_history if e["ts"] >= cutoff]

    def _summarize(self, events):
        """Compute a dict of stats for a list of events."""
        risk = Counter(e["risk"] for e in events)
        proto = Counter(e["protocol"] for e in events)
        srcs = Counter(e["src"] for e in events)
        dsts = Counter(e["dst"] for e in events)
        mitre = Counter(t for e in events for t in e["mitre"])
        attacks = [e for e in events if e["risk"] in ("MEDIUM RISK", "HIGH RISK")]
        return {
            "total": len(events),
            "risk": dict(risk),
            "proto": dict(proto),
            "top_src": srcs.most_common(5),
            "top_dst": dsts.most_common(5),
            "mitre": mitre.most_common(5),
            "alerts": attacks,
        }

    def _network_snapshot(self):
        """Human-readable telemetry snapshot across several time windows."""
        if not self.flow_history:
            return "No flows analyzed yet this session."

        def block(name, secs):
            s = self._summarize(self._window_events(secs))
            lines = [f"[{name}] flows={s['total']}",
                     f"  risk: {s['risk']}",
                     f"  protocols: {s['proto']}"]
            if s["top_src"]:
                lines.append("  top sources: " +
                             ", ".join(f"{ip}({c})" for ip, c in s["top_src"]))
            if s["mitre"]:
                lines.append("  MITRE seen: " +
                             ", ".join(f"{t}({c})" for t, c in s["mitre"]))
            return "\n".join(lines)

        parts = [
            f"Session totals: analyzed={self.total_analyzed}, "
            f"NORMAL={self.outcome_counts['NORMAL']}, ANOMALOUS={self.outcome_counts['ANOMALOUS']}, "
            f"risk LOW/MED/HIGH={self.outcome_counts['LOW RISK']}/"
            f"{self.outcome_counts['MEDIUM RISK']}/{self.outcome_counts['HIGH RISK']}.",
            block("last 5 min", 300),
            block("last 1 hour", 3600),
            block("all session", None),
        ]
        # recent notable alerts
        alerts = [e for e in self.flow_history if e["risk"] == "HIGH RISK"][-5:]
        if alerts:
            parts.append("Recent HIGH-risk alerts:")
            for e in alerts:
                t = time.strftime("%H:%M:%S", time.localtime(e["ts"]))
                parts.append(f"  {t} {e['src_full']} -> {e['dst_full']} "
                             f"[{e['protocol']}] sev={e['severity']} mitre={e['mitre']}")
        return "\n".join(parts)

    def _fallback_answer(self, question, snapshot):
        """Rule-based answer when no LLM is available."""
        q = question.lower()
        if "5 min" in q or "five min" in q:
            s = self._summarize(self._window_events(300)); win = "last 5 minutes"
        elif "hour" in q:
            s = self._summarize(self._window_events(3600)); win = "last hour"
        elif "top" in q or "talker" in q:
            s = self._summarize(self._window_events(None))
            top = "\n".join(f"  {ip}: {c} flows" for ip, c in s["top_src"]) or "  none"
            return f"Top source IPs this session:\n{top}"
        elif "threat" in q or "attack" in q or "anomal" in q:
            s = self._summarize(self._window_events(None))
            n = len(s["alerts"])
            if n == 0:
                return "No MEDIUM/HIGH risk flows detected this session. All clear."
            mit = ", ".join(f"{t}({c})" for t, c in s["mitre"]) or "none"
            return (f"{n} suspicious flow(s) detected. "
                    f"Risk breakdown: {s['risk']}. MITRE techniques: {mit}.")
        else:
            s = self._summarize(self._window_events(None)); win = "this session"
        return (f"In the {win}: {s['total']} flows. "
                f"Risk: {s['risk']}. Protocols: {s['proto']}.")

    # ---------- footer / status ----------
    def _build_footer(self):
        footer = tk.Frame(self, bg="#181825")
        footer.pack(fill="x", padx=8, pady=6)

        self.status_var = tk.StringVar(value="Idle. Choose a mode and press Start.")
        tk.Label(footer, textvariable=self.status_var, fg="#a6adc8",
                 bg="#181825", anchor="w").pack(side="left", padx=8)

        self.train_var = tk.StringVar(value="Model: untrained")
        tk.Label(footer, textvariable=self.train_var, fg="#f9e2af",
                 bg="#181825").pack(side="right", padx=8)

        self.counts_var = tk.StringVar(value="Analyzed: 0")
        tk.Label(footer, textvariable=self.counts_var, fg="#89b4fa",
                 bg="#181825").pack(side="right", padx=12)

    def _load_model_metrics(self):
        """Load the saved test-set metrics produced by train_and_evaluate.py."""
        if not os.path.exists(MODEL_METRICS_PATH):
            return None
        try:
            with open(MODEL_METRICS_PATH, encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return None

    # ---------- lifecycle ----------
    def start(self):
        contamination = SENSITIVITY_PRESETS.get(self.sens_var.get(), 0.03)
        self.detector = HybridThreatDetector(
            use_llm=self.llm_var.get(),
            baseline_n=self.baseline_var.get(),
            supervised_bundle=self.model_bundle,
            contamination=contamination,
        )

        # reuse this network's previously-learned normal, if requested
        baseline_msg = "learning fresh baseline"
        if self.reuse_var.get() and self.detector.load_baseline():
            baseline_msg = "reusing saved baseline"

        self.engine = CaptureEngine(
            self.detector, self.gui_queue,
            status_cb=lambda s: self.status_var.set(s),
            mode=self.mode_var.get(),
            iface=self._iface_map.get(self.iface_var.get()),
        )
        self.engine.start()
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.status_var.set(
            f"Running in {self.mode_var.get().upper()} mode "
            f"(sens={self.sens_var.get()}, {baseline_msg})...")

    def stop(self):
        if self.engine:
            self.engine.stop()
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status_var.set("Stopped.")

    def run_baseline(self):
        """
        (Re)run the baseline test: reset the anomaly detector and re-learn the
        normal baseline using the CURRENT 'Baseline' value. Useful to retune
        the IsolationForest sensitivity without restarting capture.
        """
        if not self.detector:
            self.status_var.set("Start capture first, then Run Baseline.")
            return
        n = self.baseline_var.get()
        contamination = SENSITIVITY_PRESETS.get(self.sens_var.get(), 0.03)
        # reset the unsupervised baseline in-place (keeps supervised model)
        self.detector.baseline_n = max(5, int(n))
        self.detector.contamination = contamination
        self.detector.model = IsolationForest(
            n_estimators=150, contamination=contamination, random_state=42)
        self.detector.training_data = []
        self.detector.is_trained = False
        self.detector.eval_summary = None
        self.detector.feature_stats = {}
        self.detector.baseline_source = "fresh"
        self.recent_scores.clear()
        self.status_var.set(
            f"Baseline reset -> collecting {n} flows (sens={self.sens_var.get()}).")
        self._refresh_training_status()

    # ---------- built-in attack injector (foolproof demo) ----------
    def _make_synthetic_flow(self, kind):
        """Craft a flow with the feature signature of the chosen attack/normal."""
        now = time.time()
        base = dict(
            flow_id=f"inj-{int(now*1000)%100000}",
            src_ip="192.168.1.66", dst_ip="10.0.0.9",
            src_port=40000, dst_port=80, protocol="TCP",
            start_time=now, last_seen=now,
        )
        if kind == "UDP Flood":
            base.update(protocol="UDP", dst_port=53, duration=5.0, packet_count=5000,
                        byte_count=300000, syn_count=0, ack_count=0,
                        total_packet_size=300000, min_packet_size=58, max_packet_size=64,
                        packet_sizes=[60] * 60, inter_arrival_times=[0.001] * 60)
        elif kind == "SYN Flood":
            base.update(protocol="TCP", dst_port=80, duration=5.0, packet_count=5000,
                        byte_count=300000, syn_count=5000, ack_count=0,
                        total_packet_size=300000, min_packet_size=58, max_packet_size=60,
                        packet_sizes=[60] * 60, inter_arrival_times=[0.001] * 60)
        elif kind == "Port Scan":
            base.update(protocol="TCP", dst_port=445, duration=3.0, packet_count=600,
                        byte_count=36000, syn_count=600, ack_count=0,
                        total_packet_size=36000, min_packet_size=58, max_packet_size=60,
                        packet_sizes=[60] * 40, inter_arrival_times=[0.005] * 40)
        elif kind == "Data Exfil":
            base.update(protocol="TCP", dst_port=443, duration=2.0, packet_count=4000,
                        byte_count=5_600_000, syn_count=1, ack_count=4000,
                        total_packet_size=5_600_000, min_packet_size=1400, max_packet_size=1500,
                        packet_sizes=[1400] * 60, inter_arrival_times=[0.0005] * 60)
        elif kind == "C2 Beacon":
            base.update(protocol="TCP", dst_port=8080, duration=30.0, packet_count=30,
                        byte_count=1800, syn_count=1, ack_count=29,
                        total_packet_size=1800, min_packet_size=58, max_packet_size=64,
                        packet_sizes=[60] * 30, inter_arrival_times=[1.0] * 29)
        else:  # Normal
            base.update(protocol="TCP", dst_port=443, duration=8.0, packet_count=28,
                        byte_count=18000, syn_count=1, ack_count=20,
                        total_packet_size=18000, min_packet_size=60, max_packet_size=1460,
                        packet_sizes=[120, 600, 1460, 300, 800] * 5,
                        inter_arrival_times=[0.05, 0.2, 0.1, 0.3, 0.15] * 5)
        return base

    def inject_attack(self):
        """Push a synthetic flow straight into the analysis pipeline (no capture)."""
        if not self.detector or not self.engine:
            self.status_var.set("Press Start first, then Inject.")
            return
        kind = self.inject_var.get()
        flow = self._make_synthetic_flow(kind)
        # feed it through the same path real flows take
        self.engine.completed_flows.put(flow)
        self.status_var.set(f"Injected synthetic '{kind}' flow -> analyzing...")

    # ---------- embedded attack simulator (popup, custom values) ----------
    def _synthetic_from_params(self, p):
        """Build a synthetic flow from CUSTOM simulator values (no packets)."""
        if p["attack"] == "Normal":
            return self._make_synthetic_flow("Normal")   # always benign -> green
        kind, count = p["attack"], max(int(p["count"]) or 200, 1)
        rate = max(float(p["rate"]) or 1, 0.001)
        proto = "UDP" if kind == "UDP Flood" else "TCP"
        hdr = 42 if proto == "UDP" else 54
        psize = max(int(p["payload"]) + hdr, 60)
        duration = max(count / rate, 0.001)
        now = time.time()
        syn = count if kind in ("SYN Flood", "Port Scan") else (1 if proto == "TCP" else 0)
        ack = 0 if kind in ("SYN Flood", "Port Scan") else (count if proto == "TCP" else 0)
        if kind == "C2 Beacon":
            iats = [max(p.get("beacon_interval", 1.0), 0.05)] * min(count, 60)
        else:
            iats = [duration / count] * min(count, 60)
        return {
            "flow_id": f"sim-{int(now*1000)%100000}",
            "src_ip": "192.168.1.66", "dst_ip": p["target"],
            "src_port": int(p["src_port"]), "dst_port": int(p["port_start"]),
            "protocol": proto, "start_time": now, "last_seen": now,
            "duration": duration, "packet_count": count,
            "byte_count": count * psize, "syn_count": syn, "ack_count": ack,
            "total_packet_size": count * psize,
            "min_packet_size": psize, "max_packet_size": psize,
            "packet_sizes": [psize] * min(count, 60), "inter_arrival_times": iats,
        }

    def open_attack_sim(self):
        """Open the embedded Attack Simulator popup with editable fields."""
        if getattr(self, "_sim_win", None) and tk.Toplevel.winfo_exists(self._sim_win):
            self._sim_win.lift()
            return

        win = tk.Toplevel(self)
        self._sim_win = win
        win.title("Attack Simulator (embedded)")
        win.configure(bg="#1e1e2e")
        win.geometry("560x520")

        self._sim_log_q = Queue()
        self._sim_engine = None

        tk.Label(win, text="ATTACK SIMULATOR", bg="#1e1e2e", fg="#cba6f7",
                 font=("Segoe UI", 13, "bold")).pack(anchor="w", padx=10, pady=(8, 0))
        mode = "scapy available" if _HAS_SIM and atksim._HAS_SCAPY else "synthetic only (no scapy)"
        tk.Label(win, text=f"Defaults are filled; edit any value. Real-send: {mode}.",
                 bg="#1e1e2e", fg="#a6adc8", font=("Segoe UI", 8)).pack(anchor="w", padx=10)

        form = tk.Frame(win, bg="#1e1e2e")
        form.pack(fill="x", padx=10, pady=8)

        gw = atksim.default_target() if _HAS_SIM else "192.168.1.1"
        v = {
            "attack": tk.StringVar(value="UDP Flood"),
            "target": tk.StringVar(value=gw),
            "rate": tk.IntVar(value=1000),
            "count": tk.IntVar(value=5000),
            "port_start": tk.IntVar(value=53),
            "payload": tk.IntVar(value=32),
            "src_port": tk.IntVar(value=40000),
            "fixed_sport": tk.BooleanVar(value=True),
            "spoof": tk.BooleanVar(value=False),
        }
        self._sim_vars = v

        tk.Label(form, text="Attack type", bg="#1e1e2e", fg="#cdd6f4").grid(row=0, column=0, sticky="w", pady=3)
        atk_cb = ttk.Combobox(form, textvariable=v["attack"], state="readonly", width=18,
                              values=["UDP Flood", "SYN Flood", "Port Scan",
                                      "Data Exfil", "C2 Beacon", "Normal"])
        atk_cb.grid(row=0, column=1, sticky="w", padx=8)

        def add(row, label, key):
            tk.Label(form, text=label, bg="#1e1e2e", fg="#cdd6f4").grid(row=row, column=0, sticky="w", pady=3)
            tk.Entry(form, textvariable=v[key], width=20, bg="#2a2a3c", fg="#e0e0e0",
                     insertbackground="#e0e0e0").grid(row=row, column=1, sticky="w", padx=8)

        add(1, "Target IP", "target")
        add(2, "Rate (pkts/sec)", "rate")
        add(3, "Count", "count")
        add(4, "Port", "port_start")
        add(5, "Payload bytes", "payload")
        add(6, "Source port", "src_port")
        tk.Checkbutton(form, text="Fixed source port (one flow)", variable=v["fixed_sport"],
                       bg="#1e1e2e", fg="#cdd6f4", selectcolor="#313244",
                       activebackground="#1e1e2e").grid(row=7, column=1, sticky="w")
        tk.Checkbutton(form, text="Spoof source IP", variable=v["spoof"],
                       bg="#1e1e2e", fg="#cdd6f4", selectcolor="#313244",
                       activebackground="#1e1e2e").grid(row=8, column=1, sticky="w")

        def apply_preset(_e=None):
            p = atksim.PRESETS.get(v["attack"].get()) if _HAS_SIM else None
            if p:
                v["rate"].set(p["rate"]); v["count"].set(p["count"] or 2000)
                v["port_start"].set(p["port_start"]); v["payload"].set(p["payload"])
        atk_cb.bind("<<ComboboxSelected>>", apply_preset)

        btns = tk.Frame(win, bg="#1e1e2e")
        btns.pack(fill="x", padx=10)
        tk.Button(btns, text="Inject (synthetic)", bg="#a6e3a1", fg="#1e1e2e", relief="flat",
                  command=self._sim_inject_synthetic).pack(side="left", padx=4)
        # tk.Button(btns, text="Send real packets", bg="#eba0ac", fg="#1e1e2e", relief="flat",
        #           command=self._sim_send_real).pack(side="left", padx=4)
        tk.Button(btns, text="Stop", bg="#f9e2af", fg="#1e1e2e", relief="flat",
                  command=self._sim_stop).pack(side="left", padx=4)

        self._sim_log = scrolledtext.ScrolledText(win, bg="#11111b", fg="#e0e0e0",
                                                  font=("Consolas", 9), height=10)
        self._sim_log.pack(fill="both", expand=True, padx=10, pady=8)
        self._sim_say("Ready. 'Inject (synthetic)' always works (needs Start). "
                      "'Send real packets' needs scapy + admin.")
        self._sim_poll()

    def _sim_params(self):
        v = self._sim_vars
        return {
            "attack": v["attack"].get(), "target": v["target"].get().strip(),
            "rate": v["rate"].get(), "count": v["count"].get(),
            "port_start": v["port_start"].get(), "port_end": v["port_start"].get(),
            "payload": v["payload"].get(), "src_port": v["src_port"].get(),
            "fixed_sport": v["fixed_sport"].get(), "spoof": v["spoof"].get(),
            "beacon_interval": 1.0,
        }

    def _sim_say(self, text):
        self._sim_log.insert("end", text + "\n"); self._sim_log.see("end")

    def _sim_inject_synthetic(self):
        if not self.detector or not self.engine:
            self._sim_say("Press Start in the main window first."); return
        p = self._sim_params()
        flow = self._synthetic_from_params(p)
        self.engine.completed_flows.put(flow)
        self._sim_say(f"Injected synthetic {p['attack']} "
                      f"(count={p['count']}, rate={p['rate']}, port={p['port_start']}) -> analyzing.")

    def _sim_send_real(self):
        if not (_HAS_SIM and atksim._HAS_SCAPY):
            self._sim_say("scapy not available -> use 'Inject (synthetic)' instead."); return
        if self._sim_engine and self._sim_engine.running:
            self._sim_say("Already running. Press Stop first."); return
        self._sim_engine = atksim.AttackEngine(
            log_cb=lambda s: self._sim_log_q.put(s),
            stat_cb=lambda n, k: None,
            done_cb=lambda: self._sim_log_q.put("(finished)"))
        self._sim_engine.launch(self._sim_params())
        self._sim_say("Launching real packets... (detector must be LIVE on the right iface)")

    def _sim_stop(self):
        if self._sim_engine:
            self._sim_engine.stop()
            self._sim_say("Stop requested.")

    def _sim_poll(self):
        if not (getattr(self, "_sim_win", None) and tk.Toplevel.winfo_exists(self._sim_win)):
            return
        try:
            while True:
                self._sim_say(self._sim_log_q.get_nowait())
        except Empty:
            pass
        self.after(200, self._sim_poll)

    # ---------- queue polling (main thread) ----------
    def _poll_queue(self):
        try:
            while True:
                features, result = self.gui_queue.get_nowait()
                self._add_flow(features, result)
        except Empty:
            pass
        # deliver any chatbot answers produced by the background LLM thread
        try:
            while True:
                role, text = self.chat_queue.get_nowait()
                self._chat_say(role, text)
        except Empty:
            pass
        self._refresh_training_status()
        self.after(200, self._poll_queue)

    def _refresh_training_status(self):
        if not self.detector:
            return
        if self.detector.is_trained:
            s = self.detector.eval_summary or {}
            self.train_var.set(
                f"Model TRAINED | baseline={s.get('samples', '?')} "
                f"anomaly_rate={s.get('anomaly_rate', 0):.2f}"
            )
        else:
            n = len(self.detector.training_data)
            self.train_var.set(f"Training baseline: {n}/{self.detector.baseline_n}")
        self._render_ml_report()

    def _add_flow(self, features, result):
        self.total_analyzed += 1
        self.counts_var.set(f"Analyzed: {self.total_analyzed}")

        ts = time.strftime("%H:%M:%S")
        src = f"{features['src_ip']}:{features['src_port']}"
        dst = f"{features['dst_ip']}:{features['dst_port']}"

        if result is None:
            pred, score, risk, tag = "(training)", "-", "TRAINING", "TRAINING"
            sev_txt, mitre_txt = "-", "-"
            self.outcome_counts["TRAINING"] += 1
        else:
            pred = result["ml_prediction"]
            score = result["anomaly_score"]
            risk = result["risk_level"]
            tag = risk
            sev_txt = f"{result['severity']} {result['severity_label']}"
            mitre_txt = ", ".join(m["id"] for m in result["mitre"]) or "-"
            # update aggregate ML stats
            self.outcome_counts[pred] = self.outcome_counts.get(pred, 0) + 1
            self.outcome_counts[risk] = self.outcome_counts.get(risk, 0) + 1
            self.recent_scores.append(score)
            if len(self.recent_scores) > self.MAX_CHART_POINTS:
                self.recent_scores.pop(0)
            # time-stamped history for charts + chatbot
            self.proto_counts[features["protocol"]] += 1
            self.flow_history.append({
                "ts": time.time(),
                "src": features["src_ip"], "dst": features["dst_ip"],
                "src_full": src, "dst_full": dst,
                "protocol": features["protocol"],
                "ml_prediction": pred,
                "supervised": result.get("supervised"),
                "risk": risk, "severity": result["severity"],
                "mitre": [m["id"] for m in result["mitre"]],
            })
            if len(self.flow_history) > 5000:
                self.flow_history.pop(0)
            self._update_ml_tab()
            self._write_audit(features, result)

        row = self.tree.insert(
            "", 0, values=(ts, src, dst, features["protocol"], pred, score,
                           sev_txt, mitre_txt, risk),
            tags=(tag,))
        self.rows[row] = (features, result)

        # keep table bounded
        children = self.tree.get_children()
        if len(children) > 300:
            old = children[-1]
            self.tree.delete(old)
            self.rows.pop(old, None)

    # ---------- selection -> detail panel ----------
    def _on_select(self, _event):
        sel = self.tree.selection()
        if not sel:
            return
        features, result = self.rows.get(sel[0], (None, None))
        if not features:
            return

        # fill detail tree
        self.detail.delete(*self.detail.get_children())
        for key in DETAIL_ORDER:
            label, meaning, _why = FEATURE_INFO[key]
            val = features.get(key, "")
            if isinstance(val, float):
                val = round(val, 3)
            self.detail.insert("", "end", values=(label, val, meaning))

        # fill LLM box
        self.llm_box.delete("1.0", "end")
        header = (f"Flow {features['src_ip']}:{features['src_port']} -> "
                  f"{features['dst_ip']}:{features['dst_port']}  "
                  f"[{features['protocol']}]\n")
        self.llm_box.insert("end", header + ("-" * 50) + "\n")
        if result is None:
            self.llm_box.insert(
                "end", "This flow was captured during the TRAINING phase, so it "
                       "was used to learn the baseline and has no verdict yet.\n")
        else:
            sup = result.get("supervised")
            sup_line = (f"Supervised RF : {sup} ({result.get('supervised_proba', 0):.0%} attack)\n"
                        if sup else "Supervised RF : (model not loaded)\n")
            self.llm_box.insert(
                "end",
                f"IForest (anom): {result['ml_prediction']}\n"
                f"Anomaly Score : {result['anomaly_score']}  "
                f"(higher = more normal; negative = anomalous side)\n"
                + sup_line +
                f"Severity      : {result['severity']} / 10  "
                f"({result['severity_label']})\n"
                f"Confidence    : {result['confidence']:.0%}\n"
                f"Risk Level    : {result['risk_level']}\n")

            # ---- MITRE ATT&CK mapping ----
            self.llm_box.insert("end", "\nMITRE ATT&CK:\n")
            if result["mitre"]:
                for m in result["mitre"]:
                    self.llm_box.insert(
                        "end",
                        f"  - {m['id']} {m['name']} [{m['tactic']}]\n"
                        f"      {m['desc']}\n")
            else:
                self.llm_box.insert("end", "  (no technique pattern matched)\n")

            # ---- statistical evidence (proof) ----
            self.llm_box.insert("end", "\nEvidence (deviation from baseline):\n")
            if result["evidence"]:
                for e in result["evidence"][:6]:
                    self.llm_box.insert("end", f"  - {e['note']}\n")
            else:
                self.llm_box.insert(
                    "end", "  (no single feature exceeded 2σ; verdict from combined pattern)\n")

            self.llm_box.insert("end", f"\nAnalysis:\n{result['llm_analysis']}\n")

    # ---------- ML Analysis tab updates ----------
    def _update_ml_tab(self):
        """Refresh stat cards, the text report, and all live charts."""
        for key, lbl in self.stat_labels.items():
            lbl.config(text=str(self.outcome_counts.get(key, 0)))
        self._render_ml_report()
        self._draw_chart()
        self._draw_mini_charts()

    def _draw_bars(self, canvas, labels, values, colors):
        """Generic vertical bar chart on a tk.Canvas."""
        canvas.delete("all")
        w = canvas.winfo_width() or 220
        h = canvas.winfo_height() or 180
        pad = 24
        if not values or max(values) == 0:
            canvas.create_text(w / 2, h / 2, text="no data", fill="#6c7086",
                               font=("Segoe UI", 9))
            return
        n = len(values)
        slot = (w - 2 * pad) / n
        bw = slot * 0.6
        top = max(values)
        canvas.create_line(pad, h - pad, w - pad, h - pad, fill="#45475a")
        for i, (lab, val) in enumerate(zip(labels, values)):
            x = pad + i * slot + (slot - bw) / 2
            bh = (h - 2 * pad) * (val / top)
            y0 = h - pad - bh
            canvas.create_rectangle(x, y0, x + bw, h - pad,
                                    fill=colors[i % len(colors)], outline="")
            canvas.create_text(x + bw / 2, y0 - 7, text=str(val),
                               fill="#cdd6f4", font=("Segoe UI", 8))
            canvas.create_text(x + bw / 2, h - pad + 9, text=lab,
                               fill="#a6adc8", font=("Segoe UI", 7))

    def _draw_mini_charts(self):
        """Risk distribution, protocol mix, and per-minute timeline charts."""
        if not hasattr(self, "chart_risk"):
            return
        # 1) risk distribution
        self._draw_bars(
            self.chart_risk, ["LOW", "MED", "HIGH"],
            [self.outcome_counts["LOW RISK"], self.outcome_counts["MEDIUM RISK"],
             self.outcome_counts["HIGH RISK"]],
            ["#81c784", "#ffb74d", "#ff5252"])
        # 2) protocol mix
        protos = self.proto_counts.most_common(5)
        self._draw_bars(self.chart_proto, [p for p, _ in protos],
                        [c for _, c in protos], ["#89b4fa", "#cba6f7", "#f9e2af"])
        # 3) flows per minute over the last 15 minutes
        now = time.time()
        buckets = [0] * 15
        for e in self.flow_history:
            age_min = int((now - e["ts"]) // 60)
            if 0 <= age_min < 15:
                buckets[14 - age_min] += 1
        self._draw_bars(self.chart_timeline,
                        [("now" if i == 14 else "") for i in range(15)],
                        buckets, ["#94e2d5"])

    def _render_ml_report(self):
        """Write the model status + evaluation summary as text."""
        if not hasattr(self, "ml_report"):
            return
        self.ml_report.delete("1.0", "end")

        if self.detector is None:
            self.ml_report.insert("end", "Model not started.\nPress Start to begin.\n")
            return

        lines = []

        # ---- supervised model (trained offline on UNSW-NB15) ----
        lines.append("SUPERVISED MODEL (trained on UNSW-NB15)")
        if self.model_bundle:
            b = self.model_bundle
            lines.append(f"  type     : {b.get('model_type', '?')}")
            lines.append(f"  features : {len(b.get('features', []))} shared features")
            lines.append(f"  trained  : {b.get('n_train', '?')} rows / "
                         f"tested {b.get('n_test', '?')} rows")
            m = (self.model_metrics or {}).get("RandomForest", b.get("metrics", {}))
            if m:
                lines.append("  TEST-SET METRICS (held-out, ground truth known):")
                lines.append(f"    accuracy  : {m.get('accuracy', 0):.3f}")
                lines.append(f"    precision : {m.get('precision', 0):.3f}")
                lines.append(f"    recall    : {m.get('recall', 0):.3f}")
                lines.append(f"    f1        : {m.get('f1', 0):.3f}")
                lines.append(f"    roc_auc   : {m.get('roc_auc', 0):.3f}")
                cm = m.get("confusion_matrix")
                if cm:
                    lines.append(f"    confusion : TN={cm[0][0]} FP={cm[0][1]} "
                                 f"FN={cm[1][0]} TP={cm[1][1]}")
            lines.append("  -> curves saved in ./model_eval/ (ROC, PR, learning curve...)")
        else:
            lines.append(f"  not loaded ({self.model_err})")
            lines.append("  run:  python train_and_evaluate.py")
        lines.append("")

        lines.append("ANOMALY LAYER (this network's baseline)")
        lines.append("  IsolationForest (unsupervised, flags novel/unknown traffic)")
        lines.append(f"  sensitivity (contamination) = {self.detector.contamination}")
        lines.append(f"  baseline source = {self.detector.baseline_source.upper()}"
                     + (" (persisted)" if os.path.exists(BASELINE_PATH) else ""))
        lines.append(f"  LLM reasoning: {'ENABLED' if self.detector.use_llm else 'disabled'}")
        lines.append("")

        if not self.detector.is_trained:
            n = len(self.detector.training_data)
            lines.append("PHASE: TRAINING (learning normal baseline)")
            lines.append(f"  collected {n}/{self.detector.baseline_n} baseline flows")
            lines.append("  -> no verdicts issued until baseline is complete")
        else:
            s = self.detector.eval_summary or {}
            total = self.total_analyzed
            anom = self.outcome_counts["ANOMALOUS"]
            norm = self.outcome_counts["NORMAL"]
            lines.append("PHASE: DETECTING (model trained)")
            lines.append("")
            lines.append("POST-TRAINING EVALUATION (on baseline data):")
            lines.append(f"  baseline samples   : {s.get('samples', '?')}")
            lines.append(f"  predicted normal    : {s.get('normal', '?')}")
            lines.append(f"  predicted anomalous : {s.get('anomalous', '?')}")
            lines.append(f"  baseline anomaly %  : {s.get('anomaly_rate', 0):.2%}")
            lines.append(f"  score min / max     : {s.get('score_min', 0):.3f} / {s.get('score_max', 0):.3f}")
            lines.append(f"  score mean / std    : {s.get('score_mean', 0):.3f} / {s.get('score_std', 0):.3f}")
            lines.append("")
            lines.append("LIVE OUTCOMES (since training finished):")
            lines.append(f"  flows scored        : {norm + anom}")
            lines.append(f"  normal / anomalous  : {norm} / {anom}")
            rate = safe_divide(anom, norm + anom)
            lines.append(f"  live anomaly rate   : {rate:.2%}")
            lines.append(f"  risk  LOW/MED/HIGH  : "
                         f"{self.outcome_counts['LOW RISK']} / "
                         f"{self.outcome_counts['MEDIUM RISK']} / "
                         f"{self.outcome_counts['HIGH RISK']}")
            lines.append("")
            lines.append("INTERPRETATION:")
            if rate == 0:
                lines.append("  All live traffic matches the learned baseline.")
            elif rate < 0.10:
                lines.append("  A small fraction of flows deviate - typical, worth a glance.")
            else:
                lines.append("  High anomaly rate - investigate the HIGH/MEDIUM rows.")

        self.ml_report.insert("end", "\n".join(lines) + "\n")

    def _draw_chart(self):
        """Draw the recent anomaly scores as a line chart on the canvas."""
        c = self.chart
        c.delete("all")
        w = c.winfo_width() or 460
        h = c.winfo_height() or 300
        pad = 30

        scores = self.recent_scores
        if not scores:
            c.create_text(w / 2, h / 2, text="No scored flows yet",
                          fill="#6c7086", font=("Segoe UI", 10))
            return

        lo = min(min(scores), -0.2)
        hi = max(max(scores), 0.2)
        span = (hi - lo) or 1.0

        def x_at(i):
            if len(scores) == 1:
                return pad
            return pad + (w - 2 * pad) * i / (len(scores) - 1)

        def y_at(v):
            return h - pad - (h - 2 * pad) * (v - lo) / span

        # axes
        c.create_line(pad, h - pad, w - pad, h - pad, fill="#45475a")
        c.create_line(pad, pad, pad, h - pad, fill="#45475a")

        # decision boundary at score = 0
        y0 = y_at(0)
        c.create_line(pad, y0, w - pad, y0, fill="#f38ba8", dash=(4, 3))
        c.create_text(w - pad - 4, y0 - 8, text="0", fill="#f38ba8",
                      anchor="e", font=("Segoe UI", 8))

        # line + points (points below 0 drawn red)
        prev = None
        for i, v in enumerate(scores):
            x, y = x_at(i), y_at(v)
            if prev is not None:
                c.create_line(prev[0], prev[1], x, y, fill="#89b4fa", width=2)
            prev = (x, y)
        for i, v in enumerate(scores):
            x, y = x_at(i), y_at(v)
            color = "#ff5252" if v < 0 else "#a6e3a1"
            c.create_oval(x - 3, y - 3, x + 3, y + 3, fill=color, outline="")

    # ---------- proof artifacts: audit log + CSV export ----------
    def _write_audit(self, features, result):
        """Append one structured JSON record to the audit trail (SIEM-ready)."""
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "src": f"{features['src_ip']}:{features['src_port']}",
            "dst": f"{features['dst_ip']}:{features['dst_port']}",
            "protocol": features["protocol"],
            "ml_prediction": result["ml_prediction"],
            "anomaly_score": result["anomaly_score"],
            "supervised": result.get("supervised"),
            "supervised_proba": result.get("supervised_proba"),
            "severity": result["severity"],
            "severity_label": result["severity_label"],
            "confidence": result["confidence"],
            "risk_level": result["risk_level"],
            "mitre": [m["id"] for m in result["mitre"]],
            "evidence": [e["note"] for e in result["evidence"][:6]],
        }
        try:
            with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
        except Exception as e:
            self.status_var.set(f"Audit write failed: {e}")

    def export_csv(self):
        """Export every analyzed flow currently in the table to a CSV file."""
        if not self.rows:
            self.status_var.set("Nothing to export yet.")
            return

        fname = f"threat_report_{datetime.now():%Y%m%d_%H%M%S}.csv"
        fields = [
            "src", "dst", "protocol", "ml_prediction", "anomaly_score",
            "supervised", "supervised_proba",
            "severity", "severity_label", "confidence", "risk_level",
            "mitre", "evidence",
            "packet_count", "byte_count", "packet_rate", "byte_rate",
            "syn_ratio", "packet_entropy", "timing_entropy",
        ]
        try:
            with open(fname, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=fields)
                writer.writeheader()
                # oldest-first for readability
                for row_id in reversed(self.tree.get_children()):
                    features, result = self.rows.get(row_id, (None, None))
                    if not features or result is None:
                        continue
                    writer.writerow({
                        "src": f"{features['src_ip']}:{features['src_port']}",
                        "dst": f"{features['dst_ip']}:{features['dst_port']}",
                        "protocol": features["protocol"],
                        "ml_prediction": result["ml_prediction"],
                        "anomaly_score": result["anomaly_score"],
                        "supervised": result.get("supervised"),
                        "supervised_proba": result.get("supervised_proba"),
                        "severity": result["severity"],
                        "severity_label": result["severity_label"],
                        "confidence": result["confidence"],
                        "risk_level": result["risk_level"],
                        "mitre": "|".join(m["id"] for m in result["mitre"]),
                        "evidence": "; ".join(e["note"] for e in result["evidence"][:6]),
                        "packet_count": features["packet_count"],
                        "byte_count": features["byte_count"],
                        "packet_rate": round(features["packet_rate"], 3),
                        "byte_rate": round(features["byte_rate"], 3),
                        "syn_ratio": round(features["syn_ratio"], 3),
                        "packet_entropy": round(features["packet_entropy"], 3),
                        "timing_entropy": round(features["timing_entropy"], 3),
                    })
            self.status_var.set(f"Exported CSV -> {os.path.abspath(fname)}")
        except Exception as e:
            self.status_var.set(f"CSV export failed: {e}")


if __name__ == "__main__":
    app = ThreatDashboard()
    app.mainloop()



