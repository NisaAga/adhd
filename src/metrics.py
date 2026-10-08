"""Metrics exactly as defined in the paper (Eq. 1-5) + helper returning a table row."""
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score


def compute_metrics(y_true, y_prob, threshold=0.5):
    """y_prob = P(ADHD). Returns dict with ACC, Prec, Rec, F1 (all in %), AUC, TP/TN/FP/FN."""
    y_true = np.asarray(y_true); y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = (tp + tn) / (tp + tn + fp + fn)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"ACC": 100 * acc, "Prec": 100 * prec, "Rec": 100 * rec, "F1": 100 * f1,
            "AUC": float(roc_auc_score(y_true, y_prob)), "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)}
