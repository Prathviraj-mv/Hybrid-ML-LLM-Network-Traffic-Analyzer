"""
=============================================================================
 TRAIN & EVALUATE  --  UNSW-NB15 -> live-deployable threat model
=============================================================================

 WHAT THIS DOES
   1. Loads the UNSW-NB15 training CSV (real labeled intrusion data).
   2. Maps its native columns onto the SAME 8 features the live GUI can
      compute in real time (see SHARED_FEATURES). This is the key step:
      a model is only useful live if it is trained on features we can also
      measure live. Train-time and inference-time feature spaces must match.
   3. Splits into train / test (the test set is hidden during training so we
      can honestly measure correctness on unseen data).
   4. Trains and COMPARES three models:
         - RandomForest        (supervised, deployed model)
         - LogisticRegression  (supervised linear baseline)
         - IsolationForest     (unsupervised anomaly detector, like the GUI)
   5. Computes EVERY standard evaluation metric:
         accuracy, precision, recall, F1, ROC-AUC, confusion matrix,
         full classification report.
   6. Saves all evaluation CURVES as PNGs in ./model_eval/:
         ROC, Precision-Recall, confusion matrices, learning curve,
         feature importances, model-comparison bar chart.
   7. Saves the deployed model bundle -> threat_model.joblib, which the GUI
      (network_threat_gui.py) loads automatically.

 WHY THIS PROVES THE MODEL IS "RIGHT"
   Because UNSW-NB15 rows are LABELED (benign vs attack), we train on part
   of the data and test on a hidden part where we already know the answer.
   The metrics/curves measure exactly how often the model agrees with the
   ground truth on data it never saw -> objective evidence of correctness.

 USAGE
   python train_and_evaluate.py
   python train_and_evaluate.py --csv "UNSW_NB15_training-set(in).csv" --sample 80000
=============================================================================
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")          # file output, no display needed
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, learning_curve, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve,
)


# =========================================================
# SHARED FEATURE SCHEMA
#   These 8 features are produced BOTH by UNSW-NB15 (after the mapping
#   below) AND by the live capture engine in network_threat_gui.py.
#   Keeping this list identical on both sides is what makes the trained
#   model usable on live traffic.
# =========================================================

SHARED_FEATURES = [
    "flow_duration",
    "packet_count",
    "byte_count",
    "packet_rate",
    "byte_rate",
    "avg_packet_size",
    "iat_mean",
    "iat_std",
]

DEFAULT_CSV = "UNSW_NB15_training-set(in).csv"
OUTPUT_DIR = "model_eval"
MODEL_PATH = "threat_model.joblib"

EPS = 1e-6  # guard against divide-by-zero


# =========================================================
# 1. LOAD + MAP UNSW-NB15 -> SHARED FEATURES
# =========================================================

def unsw_to_shared(df):
    """
    Derive the 8 shared features from UNSW-NB15 columns.

    Mapping (and why):
      flow_duration   <- dur
      packet_count    <- spkts + dpkts            (total packets both ways)
      byte_count      <- sbytes + dbytes          (total bytes both ways)
      packet_rate     <- packet_count / dur       (computed like the live GUI)
      byte_rate       <- byte_count / dur          (computed like the live GUI)
      avg_packet_size <- byte_count / packet_count
      iat_mean        <- mean(sinpkt, dinpkt) / 1000   (ms -> seconds, like live)
      iat_std         <- mean(sjit,  djit)   / 1000    (jitter as IAT spread)
    """
    out = pd.DataFrame()
    dur = df["dur"].clip(lower=EPS)

    out["flow_duration"] = df["dur"]
    out["packet_count"] = df["spkts"] + df["dpkts"]
    out["byte_count"] = df["sbytes"] + df["dbytes"]
    out["packet_rate"] = out["packet_count"] / dur
    out["byte_rate"] = out["byte_count"] / dur
    out["avg_packet_size"] = out["byte_count"] / out["packet_count"].clip(lower=1)
    # UNSW inter-packet times / jitter are in milliseconds -> convert to seconds
    out["iat_mean"] = (df["sinpkt"].fillna(0) + df["dinpkt"].fillna(0)) / 2.0 / 1000.0
    out["iat_std"] = (df["sjit"].fillna(0) + df["djit"].fillna(0)) / 2.0 / 1000.0

    # clean up infinities / NaNs introduced by division
    out = out.replace([np.inf, -np.inf], 0).fillna(0)
    return out[SHARED_FEATURES]


def load_dataset(csv_path, sample=None, seed=42):
    print(f"[*] Loading {csv_path} ...")
    df = pd.read_csv(csv_path)
    if "label" not in df.columns:
        raise ValueError("Expected a 'label' column (0=benign, 1=attack).")

    if sample and sample < len(df):
        # stratified subsample keeps the benign/attack ratio intact
        frac = sample / len(df)
        parts = [g.sample(max(1, int(round(len(g) * frac))), random_state=seed)
                 for _, g in df.groupby("label")]
        df = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
        print(f"[*] Subsampled to {len(df)} rows (stratified).")

    X = unsw_to_shared(df)
    y = df["label"].astype(int).values
    print(f"[*] Features: {SHARED_FEATURES}")
    print(f"[*] Class balance -> benign={int((y == 0).sum())}, attack={int((y == 1).sum())}")
    return X, y


# =========================================================
# 2. METRICS HELPERS
# =========================================================

def binary_metrics(y_true, y_pred, y_score=None):
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_score is not None:
        try:
            m["roc_auc"] = roc_auc_score(y_true, y_score)
        except ValueError:
            m["roc_auc"] = float("nan")
    cm = confusion_matrix(y_true, y_pred)
    m["confusion_matrix"] = cm.tolist()
    return m


def print_report(name, y_true, y_pred, metrics):
    print("\n" + "=" * 64)
    print(f" {name} ".center(64, "="))
    print("=" * 64)
    for k in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        if k in metrics:
            print(f"  {k:<10}: {metrics[k]:.4f}")
    print("  confusion matrix [rows=true, cols=pred]  (0=benign,1=attack):")
    cm = np.array(metrics["confusion_matrix"])
    print(f"     TN={cm[0,0]:>7}  FP={cm[0,1]:>7}")
    print(f"     FN={cm[1,0]:>7}  TP={cm[1,1]:>7}")
    print("\n" + classification_report(y_true, y_pred,
                                       target_names=["benign", "attack"],
                                       zero_division=0))


# =========================================================
# 3. PLOTS
# =========================================================

def plot_confusion(cm, title, path):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["benign", "attack"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["benign", "attack"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_roc(curves, path):
    plt.figure(figsize=(6, 5))
    for name, (y_true, y_score, auc) in curves.items():
        fpr, tpr, _ = roc_curve(y_true, y_score)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_pr(curves, path):
    plt.figure(figsize=(6, 5))
    for name, (y_true, y_score, _auc) in curves.items():
        prec, rec, _ = precision_recall_curve(y_true, y_score)
        plt.plot(rec, prec, label=name)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_feature_importance(model, path):
    imp = model.feature_importances_
    order = np.argsort(imp)[::-1]
    plt.figure(figsize=(7, 4.5))
    plt.bar(range(len(imp)), imp[order])
    plt.xticks(range(len(imp)), [SHARED_FEATURES[i] for i in order],
               rotation=45, ha="right")
    plt.title("RandomForest Feature Importance")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_learning_curve(estimator, X, y, path):
    print("[*] Computing learning curve (this can take a moment) ...")
    sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=3, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 6), scoring="f1")
    plt.figure(figsize=(6, 5))
    plt.plot(sizes, train_scores.mean(axis=1), "o-", label="train F1")
    plt.plot(sizes, val_scores.mean(axis=1), "o-", label="cross-val F1")
    plt.fill_between(sizes, val_scores.mean(1) - val_scores.std(1),
                     val_scores.mean(1) + val_scores.std(1), alpha=0.15)
    plt.xlabel("Training examples")
    plt.ylabel("F1 score")
    plt.title("Learning Curve (RandomForest)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_model_comparison(all_metrics, path):
    names = list(all_metrics.keys())
    keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    x = np.arange(len(keys))
    width = 0.8 / len(names)
    plt.figure(figsize=(8, 5))
    for i, name in enumerate(names):
        vals = [all_metrics[name].get(k, 0) for k in keys]
        plt.bar(x + i * width, vals, width, label=name)
    plt.xticks(x + width * (len(names) - 1) / 2, keys)
    plt.ylim(0, 1.05)
    plt.title("Model Comparison")
    plt.ylabel("Score")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


# =========================================================
# 4. MAIN PIPELINE
# =========================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV, help="UNSW-NB15 training CSV path")
    ap.add_argument("--sample", type=int, default=80000,
                    help="stratified subsample size for speed (0 = use all rows)")
    ap.add_argument("--test-size", type=float, default=0.30)
    ap.add_argument("--threshold-objective", default="f1",
                    choices=["f1", "accuracy", "youden", "fpr"],
                    help="how to pick the decision threshold "
                         "(f1=balanced, accuracy=max acc, youden=balance sens/spec, "
                         "fpr=lowest false alarms)")
    args = ap.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    X, y = load_dataset(args.csv, sample=(args.sample or None))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y)
    print(f"[*] Train rows: {len(X_train)}   Test rows (hidden): {len(X_test)}")

    all_metrics = {}
    roc_curves = {}

    # ---------- RandomForest (deployed model) ----------
    # Stronger forest + class_weight="balanced" (counters UNSW's attack-heavy mix).
    print("\n[*] Training RandomForest (balanced, tuned) ...")
    rf = RandomForestClassifier(
        n_estimators=400, max_depth=None, min_samples_leaf=2,
        max_features="sqrt", n_jobs=-1, random_state=42, class_weight="balanced")
    rf.fit(X_train, y_train)
    rf_score = rf.predict_proba(X_test)[:, 1]

    # ---- choose the decision threshold by the requested objective ----
    # Tuned on CROSS-VALIDATED training predictions (NOT the test set), so the
    # threshold choice never peeks at the data we report metrics on.
    print(f"[*] Tuning threshold (objective={args.threshold_objective}) via CV ...")
    oof = cross_val_predict(
        RandomForestClassifier(
            n_estimators=200, min_samples_leaf=2, n_jobs=-1,
            random_state=42, class_weight="balanced"),
        X_train, y_train, cv=3, method="predict_proba", n_jobs=-1)[:, 1]

    cand = np.linspace(0.05, 0.95, 181)
    def score_thr(t):
        pred = (oof >= t).astype(int)
        tn = int(((y_train == 0) & (pred == 0)).sum())
        fp = int(((y_train == 0) & (pred == 1)).sum())
        fpr_t = fp / max(tn + fp, 1)
        if args.threshold_objective == "fpr":
            # best recall subject to FPR <= 2%
            return (-1e9 if fpr_t > 0.02 else recall_score(y_train, pred, zero_division=0))
        if args.threshold_objective == "accuracy":
            return accuracy_score(y_train, pred)
        if args.threshold_objective == "youden":
            return recall_score(y_train, pred, zero_division=0) - fpr_t
        return f1_score(y_train, pred, zero_division=0)        # default: f1
    rf_threshold = float(max(cand, key=score_thr))
    print(f"[*] Selected reporting threshold = {rf_threshold:.3f}")

    # also compute a conservative LIVE threshold (FPR<=2% on CV data) so the
    # GUI can run calm on real traffic regardless of the reporting objective.
    def fpr_at(t):
        pred = (oof >= t).astype(int)
        tn = int(((y_train == 0) & (pred == 0)).sum())
        fp = int(((y_train == 0) & (pred == 1)).sum())
        return fp / max(tn + fp, 1)
    live_ok = [t for t in cand if fpr_at(t) <= 0.02]
    rf_threshold_live = float(min(live_ok)) if live_ok else 0.9
    print(f"[*] Conservative live threshold (FPR<=2%) = {rf_threshold_live:.3f}")

    rf_pred = (rf_score >= rf_threshold).astype(int)
    all_metrics["RandomForest"] = binary_metrics(y_test, rf_pred, rf_score)
    cm = np.array(all_metrics["RandomForest"]["confusion_matrix"])
    realized_fpr = cm[0, 1] / max(cm[0].sum(), 1)
    print(f"[*] Realized benign false-positive rate = {realized_fpr:.3f}")
    roc_curves["RandomForest"] = (y_test, rf_score, all_metrics["RandomForest"]["roc_auc"])
    print_report("RandomForest", y_test, rf_pred, all_metrics["RandomForest"])

    # ---------- LogisticRegression (scaled baseline) ----------
    print("\n[*] Training LogisticRegression ...")
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_score = lr.predict_proba(X_test)[:, 1]
    all_metrics["LogisticRegression"] = binary_metrics(y_test, lr_pred, lr_score)
    roc_curves["LogisticRegression"] = (y_test, lr_score, all_metrics["LogisticRegression"]["roc_auc"])
    print_report("LogisticRegression", y_test, lr_pred, all_metrics["LogisticRegression"])

    # ---------- IsolationForest (unsupervised, like the live GUI) ----------
    print("\n[*] Training IsolationForest (unsupervised, on benign only) ...")
    iforest = IsolationForest(n_estimators=200, contamination=0.1, random_state=42)
    iforest.fit(X_train[y_train == 0])               # learn 'normal' from benign rows
    if_raw = iforest.predict(X_test)                 # -1 anomaly, +1 normal
    if_pred = (if_raw == -1).astype(int)             # map anomaly -> attack(1)
    if_score = -iforest.decision_function(X_test)    # higher = more anomalous
    all_metrics["IsolationForest"] = binary_metrics(y_test, if_pred, if_score)
    roc_curves["IsolationForest"] = (y_test, if_score, all_metrics["IsolationForest"]["roc_auc"])
    print_report("IsolationForest", y_test, if_pred, all_metrics["IsolationForest"])

    # ---------- plots ----------
    print("\n[*] Saving evaluation plots ...")
    plot_roc(roc_curves, os.path.join(OUTPUT_DIR, "roc_curves.png"))
    plot_pr(roc_curves, os.path.join(OUTPUT_DIR, "precision_recall.png"))
    plot_confusion(all_metrics["RandomForest"]["confusion_matrix"],
                   "RandomForest Confusion", os.path.join(OUTPUT_DIR, "confusion_rf.png"))
    plot_confusion(all_metrics["LogisticRegression"]["confusion_matrix"],
                   "LogReg Confusion", os.path.join(OUTPUT_DIR, "confusion_lr.png"))
    plot_confusion(all_metrics["IsolationForest"]["confusion_matrix"],
                   "IsolationForest Confusion", os.path.join(OUTPUT_DIR, "confusion_iforest.png"))
    plot_feature_importance(rf, os.path.join(OUTPUT_DIR, "feature_importance.png"))
    plot_model_comparison(all_metrics, os.path.join(OUTPUT_DIR, "model_comparison.png"))
    plot_learning_curve(
        RandomForestClassifier(n_estimators=120, n_jobs=-1, random_state=42),
        X_train, y_train, os.path.join(OUTPUT_DIR, "learning_curve.png"))

    # ---------- persist metrics ----------
    metrics_out = {k: {mk: mv for mk, mv in v.items() if mk != "confusion_matrix"}
                   for k, v in all_metrics.items()}
    for k in all_metrics:
        metrics_out[k]["confusion_matrix"] = all_metrics[k]["confusion_matrix"]
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as fh:
        json.dump(metrics_out, fh, indent=2)
    pd.DataFrame(
        {k: {mk: v.get(mk) for mk in ("accuracy", "precision", "recall", "f1", "roc_auc")}
         for k, v in all_metrics.items()}
    ).T.to_csv(os.path.join(OUTPUT_DIR, "metrics_summary.csv"))

    # ---------- save deployed model bundle (RandomForest) ----------
    # GUI uses the conservative live threshold (calm on real traffic); the
    # reporting threshold (high accuracy/F1) is kept for transparency.
    bundle = {
        "model": rf,
        "features": SHARED_FEATURES,
        "model_type": "RandomForestClassifier (balanced)",
        "classes": {"0": "benign", "1": "attack"},
        "threshold": rf_threshold_live,        # deployed (low false-positive) point
        "threshold_report": rf_threshold,      # high-accuracy reporting point
        "metrics": metrics_out["RandomForest"],
        "trained_on": os.path.basename(args.csv),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    joblib.dump(bundle, MODEL_PATH)

    print("\n" + "=" * 64)
    print(f"[+] Saved deployed model -> {os.path.abspath(MODEL_PATH)}")
    print(f"[+] Plots + metrics in   -> {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 64)
    print("\nSummary (test set):")
    for name, m in all_metrics.items():
        print(f"  {name:<20} acc={m['accuracy']:.3f}  f1={m['f1']:.3f}  "
              f"auc={m.get('roc_auc', float('nan')):.3f}")


if __name__ == "__main__":
    main()
