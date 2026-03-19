import pandas as pd
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