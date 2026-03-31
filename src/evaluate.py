import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    f1_score,
    accuracy_score,
    confusion_matrix,
)

from src.utils import string_similarity


def evaluate_name_baseline(labels_df: pd.DataFrame) -> pd.DataFrame:
    preds = []

    for _, row in labels_df.iterrows():
        sim = string_similarity(row["column_a"], row["column_b"])
        pred = 1 if sim > 0.5 else 0

        preds.append({
            "column_a": row["column_a"],
            "column_b": row["column_b"],
            "label": row["label"],
            "similarity": sim,
            "prediction": pred
        })

    return pd.DataFrame(preds)


def get_predictions(model, dataset) -> tuple[np.ndarray, np.ndarray]:
    """Run inference and return (probs, true_labels) as numpy arrays."""
    model.eval()
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    all_probs, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            logits = model(batch["text_a"], batch["text_b"])
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(batch["label"].numpy().tolist())

    return np.array(all_probs), np.array(all_labels)


def find_best_threshold(probs: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Scan thresholds in [0.1, 0.9] and return the one with the highest F1."""
    best_thresh, best_f1 = 0.5, 0.0
    for thresh in np.arange(0.1, 0.91, 0.01):
        preds = (probs >= thresh).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(thresh)
    return best_thresh, best_f1


def compute_metrics(probs: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    preds = (probs >= threshold).astype(int)
    pr_auc, _, _ = compute_pr_auc(probs, labels)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, zero_division=0),
        "pr_auc": pr_auc,
        "threshold": threshold,
    }


def compute_pr_auc(probs: np.ndarray, labels: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    precision, recall, _ = precision_recall_curve(labels, probs)
    return auc(recall, precision), precision, recall


def plot_pr_curve(probs: np.ndarray, labels: np.ndarray, save_path: str = "pr_curve.png") -> float:
    pr_auc, precision, recall = compute_pr_auc(probs, labels)

    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, label=f"PR-AUC = {pr_auc:.3f}", linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (Validation Set)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    return pr_auc


def print_confusion_details(dataset, probs: np.ndarray, labels: np.ndarray, threshold: float) -> None:
    """Print which specific column pairs are FP or FN, plus the confusion matrix."""
    preds = (probs >= threshold).astype(int)

    print("\nFalse Positives (predicted match, actually no match):")
    fp_rows = [dataset.data.iloc[i] for i, (p, l) in enumerate(zip(preds, labels)) if p == 1 and l == 0]
    if fp_rows:
        for row in fp_rows:
            print(f"  {row['column_a']:25s} <-> {row['column_b']}")
    else:
        print("  none")

    print("\nFalse Negatives (predicted no match, actually match):")
    fn_rows = [dataset.data.iloc[i] for i, (p, l) in enumerate(zip(preds, labels)) if p == 0 and l == 1]
    if fn_rows:
        for row in fn_rows:
            print(f"  {row['column_a']:25s} <-> {row['column_b']}")
    else:
        print("  none")

    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    print(f"\nConfusion Matrix — TN: {tn}  FP: {fp}  FN: {fn}  TP: {tp}")
