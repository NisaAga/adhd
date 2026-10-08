"""Figures: EDA + evaluation plots (confusion matrix, ROC, paper-vs-ours comparison)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch
from sklearn.metrics import confusion_matrix, roc_curve, auc
from . import config as C

# Paper results (Table 4 multi-headed, Table 5 baseline): (Acc, Prec, Rec, F1, AUC)
PAPER_MULTI = {
 "BI-LSTM,BI-LSTM": (76.58, 87.76, 58.10, 69.92, 0.75), "BI-LSTM,LSTM": (87.97, 82.35, 94.59, 88.05, 0.88),
 "BI-LSTM,GRU": (74.68, 72.97, 72.97, 72.97, 0.75),     "LSTM,BI-LSTM": (89.87, 83.72, 97.30, 90.00, 0.90),
 "LSTM,LSTM": (80.38, 75.90, 85.14, 80.25, 0.81),       "LSTM,GRU": (81.65, 83.58, 75.68, 79.43, 0.81),
 "GRU,BI-LSTM": (80.38, 83.08, 72.97, 77.70, 0.80),     "GRU,LSTM": (79.11, 73.56, 86.49, 79.50, 0.80),
 "GRU,GRU": (70.25, 64.21, 82.43, 72.19, 0.71)}
PAPER_BASE = {"BI-LSTM": (70.89, 75.00, 56.76, 64.62, 0.70), "LSTM": (82.28, 76.14, 90.54, 82.72, 0.83),
              "GRU": (74.05, 70.89, 75.68, 73.20, 0.74)}


def _save(fig, name):
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    p = C.FIG_DIR / name
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
    return p


def class_balance(data):
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, t in enumerate(("train", "val", "test")):
        y = data[f"y_{t}"]
        ax.bar(i - .2, (y == 0).sum(), .4, color="tab:blue", label="Healthy" if i == 0 else "")
        ax.bar(i + .2, (y == 1).sum(), .4, color="tab:red", label="ADHD" if i == 0 else "")
    ax.set_xticks(range(3)); ax.set_xticklabels(["train (cell 4)", "val (cell 10)", "test (cell 11)"])
    ax.set_ylabel("subjects"); ax.set_title("Class balance"); ax.legend()
    return _save(fig, "class_balance.png")


def sample_signals(data, split="train"):
    X, y = data[f"X_{split}"], data[f"y_{split}"]
    ch = data[f"channels_{split}"]; fs = float(data["fs"])
    fig, axes = plt.subplots(2, 2, figsize=(11, 5), sharex=True)
    n = min(X.shape[1], int(10 * fs))
    for r, (lab, nm) in enumerate(((0, "Healthy"), (1, "ADHD"))):
        i = np.where(y == lab)[0][0]; t = np.arange(X.shape[1]) / fs
        for c in range(2):
            axes[r, c].plot(t[:n], X[i, :n, c], lw=.6)
            axes[r, c].set_title(f"{nm} - {ch[c]}")
    for a in axes[-1]: a.set_xlabel("time (s)")
    return _save(fig, f"sample_signals_{split}.png")


def mean_psd(data, split="train"):
    X, y = data[f"X_{split}"], data[f"y_{split}"]; fs = float(data["fs"]); ch = data[f"channels_{split}"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for c in range(2):
        for lab, nm, col in ((0, "Healthy", "tab:blue"), (1, "ADHD", "tab:red")):
            f, p = welch(X[y == lab][:, :, c], fs=fs, nperseg=min(512, X.shape[1]), axis=1)
            axes[c].semilogy(f, p.mean(0), label=nm, color=col)
        axes[c].set_xlim(0, 40); axes[c].set_title(f"Mean PSD - {ch[c]}"); axes[c].set_xlabel("Hz"); axes[c].legend()
    return _save(fig, f"mean_psd_{split}.png")


def confusion_matrix_plot(y_true, y_pred, title, fname):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 4)); ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm): ax.text(j, i, v, ha="center", va="center", fontsize=14)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Healthy", "ADHD"]); ax.set_yticks([0, 1]); ax.set_yticklabels(["Healthy", "ADHD"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    return _save(fig, fname)


def roc_plot(y_true, y_score, title, fname):
    fpr, tpr, _ = roc_curve(y_true, y_score); a = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(4.5, 4.5)); ax.plot(fpr, tpr, label=f"AUC={a:.2f}"); ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title(title); ax.legend()
    return _save(fig, fname)


def paper_comparison(ours: dict, fname="ours_vs_paper_accuracy.png"):
    """ours: {'LSTM,BI-LSTM': 88.6, ...} accuracy % -> bar chart against the paper."""
    keys = list(PAPER_MULTI)
    fig, ax = plt.subplots(figsize=(11, 4)); x = np.arange(len(keys))
    ax.bar(x - .2, [PAPER_MULTI[k][0] for k in keys], .4, label="Paper")
    ax.bar(x + .2, [ours.get(k, np.nan) for k in keys], .4, label="Ours")
    ax.set_xticks(x); ax.set_xticklabels(keys, rotation=30, ha="right"); ax.set_ylabel("Accuracy (%)"); ax.legend()
    ax.set_title("Multi-headed accuracy: paper vs reproduction")
    return _save(fig, fname)
