"""CSV loading for respiration chamber data."""

import glob
import os
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


def load_csv(file_path) -> pd.DataFrame:
    """Load a single chamber CSV file.

    Expects semicolon-delimited format with header on row 0,
    two metadata rows (1-2), and data from row 3 onward.

    Accepts a filesystem path (``str`` or ``pathlib.Path``) **or** any
    file-like object with a ``read()`` method (e.g. ``io.StringIO``,
    ``io.BytesIO``, or a Streamlit ``UploadedFile``).

    Returns a DataFrame with parsed Timestamp, numeric columns,
    and derived ``time_sec`` / ``time_min`` columns relative to the
    first valid timestamp. Returns an empty DataFrame on parse failure.
    """
    if hasattr(file_path, "read"):
        raw = file_path.read()
        raw_lines = (raw.decode("utf-8") if isinstance(raw, bytes) else raw).splitlines()
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            raw_lines = f.read().splitlines()

    header_vals = raw_lines[0].split(";")
    if header_vals and header_vals[-1].strip() == "":
        header_vals = header_vals[:-1]
    n_cols = len(header_vals)

    data_lines = [row for row in raw_lines[3:] if str(row).strip() != ""]
    if not data_lines:
        return pd.DataFrame()

    parsed_rows = [row.split(";")[:n_cols] for row in data_lines]
    df = pd.DataFrame(parsed_rows, columns=header_vals)

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"], format="%d.%m.%Y %H:%M:%S", errors="coerce"
    )
    for col in df.columns:
        if col != "Timestamp":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.dropna(subset=["Timestamp"])
        .sort_values("Timestamp")
        .reset_index(drop=True)
    )
    if df.empty:
        return df

    df["time_sec"] = (df["Timestamp"] - df["Timestamp"].iloc[0]).dt.total_seconds()
    df["time_min"] = df["time_sec"] / 60.0
    return df


def load_directory(
    data_dir: str | Path,
    recursive: bool = True,
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Load all CSV files from a directory.

    Parameters
    ----------
    data_dir  : path to search
    recursive : include subdirectories
    verbose   : print loading progress

    Returns
    -------
    dict mapping filename → DataFrame (empty files are skipped)
    """
    pattern = (
        os.path.join(str(data_dir), "**", "*.csv")
        if recursive
        else os.path.join(str(data_dir), "*.csv")
    )
    csv_files = sorted(glob.glob(pattern, recursive=recursive))

    result: Dict[str, pd.DataFrame] = {}
    for fp in csv_files:
        fname = os.path.basename(fp)
        try:
            df = load_csv(fp)
            if df.empty:
                if verbose:
                    print(f"  [skipped – empty] {fname}")
                continue
            result[fname] = df
            if verbose:
                print(
                    f"  {fname}  ({len(df)} rows, "
                    f"{df['time_min'].max():.1f} min)"
                )
        except Exception as exc:
            if verbose:
                print(f"  [error] {fname}: {exc}")

    if verbose:
        print(f"\n{len(result)} file(s) loaded.")
    return result
