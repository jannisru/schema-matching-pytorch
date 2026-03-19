import pandas as pd


def column_to_text(df: pd.DataFrame, column_name: str, n_samples: int = 5) -> str:
    values = df[column_name].dropna().astype(str).head(n_samples).tolist()
    values_text = ", ".join(values)
    return f"column: {column_name} | values: {values_text}"