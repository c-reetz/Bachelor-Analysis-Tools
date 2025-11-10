import re
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame

HEADER_MARKER = "##Temp./°C;Time/min;Mass/%;Segment"


def _norm(s: str) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    s = str(s).strip().lower()
    s = s.replace("°", "").replace("celsius", "c").replace("°c", "c")
    s = re.sub(r"[^a-z0-9]+", "", s)  # strip punctuation/spaces
    # common compacting, e.g. "temp./c" -> "temp"
    s = s.replace("tempc", "temp").replace("temp/c", "temp")
    return s


def _read_excel_tg(path: str) -> pd.DataFrame:
    xl = pd.ExcelFile(path)
    for sh in xl.sheet_names:
        raw = xl.parse(sheet_name=sh, header=None, dtype=str)
        header_row = None
        col_idx = None
        for i in range(min(len(raw), 500)):  # scan first 500 rows for a header-like row
            row = [_norm(v) for v in raw.iloc[i].tolist()]
            has = {"temp": -1, "time": -1, "mass": -1, "segment": -1}
            for j, cell in enumerate(row):
                if has["temp"] == -1 and ("temp" in cell or cell in ("tc", "t")):
                    has["temp"] = j
                if has["time"] == -1 and "time" in cell:
                    has["time"] = j
                if has["mass"] == -1 and "mass" in cell:
                    has["mass"] = j
                if has["segment"] == -1 and "segment" in cell:
                    has["segment"] = j
            if all(v >= 0 for v in has.values()):
                header_row, col_idx = i, has
                break
        if header_row is None:
            continue  # try next sheet

        data = raw.iloc[header_row + 1:, :].copy()
        df = pd.DataFrame({
            "temp_C": data.iloc[:, col_idx["temp"]],
            "time_min": data.iloc[:, col_idx["time"]],
            "mass_pct": data.iloc[:, col_idx["mass"]],
            "segment": data.iloc[:, col_idx["segment"]],
        })

        # numeric coercion (supports comma decimals)
        for c in ("temp_C", "time_min", "mass_pct"):
            s = df[c].astype(str).str.replace(",", ".", regex=False)
            df[c] = pd.to_numeric(s, errors="coerce")

        # drop rows without any numeric info
        df = df.dropna(subset=["temp_C", "time_min", "mass_pct"], how="all")
        # forward-fill segment if blank
        df["segment"] = df["segment"].replace({np.nan: None}).ffill()

        return df.reset_index(drop=True)

    raise ValueError("Could not locate the header row in any Excel sheet.")


def _load_region_from_csv_text(text: str) -> pd.DataFrame:
    lines = text.splitlines()
    start_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith(HEADER_MARKER):
            start_idx = i
            break
    if start_idx is None:
        raise ValueError(f"{HEADER_MARKER} not found.")

    # Collect the block until the next line that starts with '##' (or EOF)
    block = []
    for j in range(start_idx, len(lines)):
        line = lines[j]
        if j > start_idx and line.strip().startswith("##"):
            break
        block.append(line)

    if not block:
        raise ValueError("Detected header but no data lines found.")

    # Clean the header line (remove leading '##')
    first = block[0].strip()
    if first.startswith("##"):
        first = first[2:]
    block[0] = first

    buf = StringIO("\n".join(block))

    # Try decimal comma then dot
    for decimal in (",", "."):
        try:
            df: DataFrame = pd.read_csv(buf, sep=";", decimal=decimal, engine="python")
            if not {"Temp./°C", "Time/min", "Mass/%", "Segment"}.issubset(df.columns):
                raise ValueError("Parsed columns do not match expected header.")
            df["Temp./°C"] = pd.to_numeric(df["Temp./°C"], errors="coerce")
            df["Time/min"] = pd.to_numeric(df["Time/min"], errors="coerce")
            df["Mass/%"] = pd.to_numeric(df["Mass/%"], errors="coerce")
            break
        except Exception:
            buf.seek(0)
            continue
    else:
        raise ValueError("Failed to parse region with decimal ',' or '.'.")

    return df.rename(columns={
        "Temp./°C": "temp_C",
        "Time/min": "time_min",
        "Mass/%": "mass_pct",
        "Segment": "segment",
    }).dropna(how="all")


def load_thermogravimetric_data(path: str | Path) -> pd.DataFrame:
    """
    Load only the '##Temp./°C;Time/min;Mass/%;Segment' region from a CSV or XLSX.
    Returns a DataFrame with columns: temp_C, time_min, mass_pct, segment.
    """
    path = Path(path)
    ext = path.suffix.lower()

    if ext.lower().endswith((".xlsx", ".xls")):
        return _read_excel_tg(path)

    if ext in {".csv", ".txt", ".dat"}:
        # Try common encodings
        for enc in ("utf-8-sig", "utf-16", "latin-1"):
            try:
                text = path.read_text(encoding=enc)
                break
            except Exception:
                text = None
        if text is None:
            text = path.read_bytes().decode("utf-8", errors="ignore")
        return _load_region_from_csv_text(text)

    raise ValueError(f"Unsupported file type: {ext}")
