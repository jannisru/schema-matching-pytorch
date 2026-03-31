import os
import pandas as pd
from src.preprocessing import column_to_text

import torch
from torch.utils.data import Dataset


TABLE_PAIRS = [
    ("customers_a", "customers_raw_a.csv", "customers_b", "customers_raw_b.csv"),
    ("orders_a", "orders_raw_a.csv", "orders_b", "orders_raw_b.csv"),
    ("products_a", "products_raw_a.csv", "products_b", "products_raw_b.csv"),
    ("employees_a", "employees_raw_a.csv", "employees_b", "employees_raw_b.csv"),
    ("payments_a", "payments_raw_a.csv", "payments_b", "payments_raw_b.csv"),
    ("shipments_a", "shipments_raw_a.csv", "shipments_b", "shipments_raw_b.csv"),
    ("suppliers_a", "suppliers_raw_a.csv", "suppliers_b", "suppliers_raw_b.csv"),
    ("reviews_a", "reviews_raw_a.csv", "reviews_b", "reviews_raw_b.csv"),
    ("invoices_a", "invoices_raw_a.csv", "invoices_b", "invoices_raw_b.csv"),
    ("contracts_a", "contracts_raw_a.csv", "contracts_b", "contracts_raw_b.csv"),
]


def build_full_dataset(data_dir: str, labels_df: pd.DataFrame) -> pd.DataFrame:
    all_rows = []
    for table_a_name, file_a, table_b_name, file_b in TABLE_PAIRS:
        df_a = pd.read_csv(os.path.join(data_dir, file_a))
        df_b = pd.read_csv(os.path.join(data_dir, file_b))
        pair_df = build_pair_dataset(df_a, df_b, labels_df, table_a_name, table_b_name)
        all_rows.append(pair_df)
    return pd.concat(all_rows, ignore_index=True)


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
            "column_a": col_a,
            "column_b": col_b,
            "text_a": text_a,
            "text_b": text_b,
            "label": label
        })

    return pd.DataFrame(rows)


class ColumnMatchingDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        return {
            "text_a": row["text_a"],
            "text_b": row["text_b"],
            "label": torch.tensor(row["label"], dtype=torch.float32)
        }