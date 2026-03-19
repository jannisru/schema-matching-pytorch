import pandas as pd
from src.preprocessing import column_to_text


def build_pair_dataset(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    labels_df: pd.DataFrame,
    table_a_name: str,
    table_b_name: str,
) -> pd.DataFrame:
    rows = []

    filtered = labels_df[
        (labels_df["table_a"] == table_a_name) &
        (labels_df["table_b"] == table_b_name)
    ]

    for _, row in filtered.iterrows():
        col_a = row["column_a"]
        col_b = row["column_b"]
        label = row["label"]

        text_a = column_to_text(df_a, col_a)
        text_b = column_to_text(df_b, col_b)

        rows.append({
            "text_a": text_a,
            "text_b": text_b,
            "label": label
        })

    return pd.DataFrame(rows)