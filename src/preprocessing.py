import pandas as pd


def column_to_text(df: pd.DataFrame, column_name: str, n_samples: int = 5) -> str:
    col = df[column_name]

    values = col.dropna().astype(str).head(n_samples).tolist()
    values_text = ", ".join(values)

    dtype = str(col.dtype)
    null_rate = col.isna().mean()
    unique_ratio = col.nunique() / max(len(col), 1)

    parts = [
        f"column: {column_name}",
        f"dtype: {dtype}",
        f"null_rate: {null_rate:.2f}",
        f"unique_ratio: {unique_ratio:.2f}",
    ]

    if pd.api.types.is_numeric_dtype(col) and col.notna().any():
        parts.append(f"min: {col.min():.2f}")
        parts.append(f"max: {col.max():.2f}")
        parts.append(f"mean: {col.mean():.2f}")

    parts.append(f"values: {values_text}")

    return " | ".join(parts)
