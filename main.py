import argparse
import yaml
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from src.dataset import build_full_dataset, ColumnMatchingDataset
from src.model import ColumnMatcher, get_device
from src.train import train_model
from src.evaluate import (
    get_predictions,
    find_best_threshold,
    compute_metrics,
    plot_pr_curve,
    print_confusion_details,
)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Schema Matching Training & Evaluation")
    parser.add_argument("--config", default="config.yaml", help="Pfad zur config.yaml")
    parser.add_argument("--load-model", action="store_true", help="Gespeichertes Modell laden, Training überspringen")
    args = parser.parse_args()

    cfg = load_config(args.config)

    labels_df = pd.read_csv(cfg["data"]["labels"])
    full_df = build_full_dataset(cfg["data"]["dir"], labels_df)

    n_pos = int(full_df["label"].sum())
    n_neg = int((1 - full_df["label"]).sum())
    pos_weight = n_neg / n_pos
    print(f"Dataset: {len(full_df)} Paare ({n_pos} positiv, {n_neg} negativ, pos_weight={pos_weight:.1f})")
    print(f"Device:  {get_device()}\n")

    train_df, val_df = train_test_split(
        full_df,
        test_size=cfg["data"]["val_split"],
        stratify=full_df["label"],
        random_state=cfg["data"]["random_seed"],
    )

    train_dataset = ColumnMatchingDataset(train_df.reset_index(drop=True))
    val_dataset = ColumnMatchingDataset(val_df.reset_index(drop=True))

    if args.load_model:
        model_path = cfg["output"]["model_path"]
        model = ColumnMatcher(
            encoder=cfg["model"]["encoder"],
            dropout=cfg["model"]["dropout"],
        )
        model.load_state_dict(torch.load(model_path, map_location=get_device()))
        print(f"Modell geladen: {model_path}")
    else:
        model = train_model(
            train_dataset,
            val_dataset,
            epochs=cfg["training"]["epochs"],
            lr=cfg["training"]["lr"],
            batch_size=cfg["training"]["batch_size"],
            patience=cfg["training"]["patience"],
            pos_weight=pos_weight,
            encoder=cfg["model"]["encoder"],
            dropout=cfg["model"]["dropout"],
        )
        model_path = cfg["output"]["model_path"]
        torch.save(model.state_dict(), model_path)
        print(f"\nModell gespeichert: {model_path}")

    # --- Threshold-Optimierung auf Validation Set ---
    val_probs, val_labels = get_predictions(model, val_dataset)
    best_thresh, best_val_f1 = find_best_threshold(val_probs, val_labels)
    print(f"\nBester Threshold (Val-F1-Optimierung): {best_thresh:.2f}  →  F1: {best_val_f1:.3f}")

    # --- Metriken mit optimiertem Threshold ---
    train_probs, train_labels = get_predictions(model, train_dataset)
    train_metrics = compute_metrics(train_probs, train_labels, threshold=best_thresh)
    val_metrics = compute_metrics(val_probs, val_labels, threshold=best_thresh)

    print(f"\nTrain — Accuracy: {train_metrics['accuracy']:.3f}  F1: {train_metrics['f1']:.3f}  PR-AUC: {train_metrics['pr_auc']:.3f}")
    print(f"Val   — Accuracy: {val_metrics['accuracy']:.3f}  F1: {val_metrics['f1']:.3f}  PR-AUC: {val_metrics['pr_auc']:.3f}")

    # --- Precision-Recall Kurve ---
    pr_curve_path = cfg["output"]["pr_curve"]
    pr_auc = plot_pr_curve(val_probs, val_labels, save_path=pr_curve_path)
    print(f"\nPR-Kurve gespeichert: {pr_curve_path}  (AUC={pr_auc:.3f})")

    # --- Confusion Details ---
    print_confusion_details(val_dataset, val_probs, val_labels, threshold=best_thresh)
