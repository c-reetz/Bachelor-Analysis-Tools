import re
from io import StringIO
from pathlib import Path
from typing import Dict, Any, Optional, Callable
import numpy as np
import pandas as pd
from pandas import DataFrame
import logging

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

# Declarative spec: sample -> regime -> O2 label -> relative filepath (to base_dir)
SPEC: Dict[str, Dict[str, Dict[str, str]]] = {
    "BRF": {
        "isothermal_225": {
            "5%": "TG TEST 9 - ISOTHERMAL 225C 5% O2/ExpDat_BRF500.xlsx",
            "10%": "TG TEST 10 - ISOTHERMAL 225C 10% O2/ExpDat_BRF500.xlsx",
            "20%": "TG TEST 4 - ISOTHERMAL 225C 20% O2/ExpDat_BRF.xlsx",
        },
        "isothermal_250": {
            "5%": "TG TEST 12 - ISOTHERMAL 250C 5% O2/ExpDat_BRF500.xlsx",
            "10%": "TG TEST 11 - ISOTHERMAL 250C 10% o2/ExpDat_BRF500.xlsx",
            "20%": "TG TEST 3 - ISOTHERMAL 250C 20% O2/ExpDat_BRF 500.xlsx",
        },
        "linear": {
            "5%": "TG TEST 7 . OXIDATION 600C 5% O2/ExpDat_BRF500.xlsx",
            "10%": "TG TEST 6 - OXIDATION 600C 10% O2/ExpDat_BRF500.xlsx",
            "20%": "TG TEST 8 - OXIDATION 600C 20% O2/ExpDat_BRF500.xlsx",
        },
    },
    "WS": {
        "isothermal_225": {
            "5%": "TG TEST 9 - ISOTHERMAL 225C 5% O2/ExpDat_WS500.xlsx",
            "10%": "TG TEST 10 - ISOTHERMAL 225C 10% O2/ExpDat_WS500.xlsx",
            "20%": "TG TEST 4 - ISOTHERMAL 225C 20% O2/ExpDat_WS.xlsx",
        },
        "isothermal_250": {
            "5%": "TG TEST 12 - ISOTHERMAL 250C 5% O2/ExpDat_WS500.xlsx",
            "10%": "TG TEST 11 - ISOTHERMAL 250C 10% o2/ExpDat_WS500.xlsx",
            "20%": "TG TEST 3 - ISOTHERMAL 250C 20% O2/ExpDat_WS500.xlsx",
        },
        "linear": {
            "5%": "TG TEST 7 . OXIDATION 600C 5% O2/ExpDat_WS500.xlsx",
            "10%": "TG TEST 6 - OXIDATION 600C 10% O2/ExpDat_WS500.xlsx",
            "20%": "TG TEST 8 - OXIDATION 600C 20% O2/ExpDat_ws500.xlsx",
        },
    },
    "PW": {
        "isothermal_225": {
            "5%": "TG TEST 9 - ISOTHERMAL 225C 5% O2/ExpDat_PW500.xlsx",
            "10%": "TG TEST 10 - ISOTHERMAL 225C 10% O2/ExpDat_PW500.xlsx",
            "20%": "TG TEST 4 - ISOTHERMAL 225C 20% O2/ExpDat_PW.xlsx",
        },
        "isothermal_250": {
            "5%": "TG TEST 12 - ISOTHERMAL 250C 5% O2/ExpDat_PW500.xlsx",
            "10%": "TG TEST 11 - ISOTHERMAL 250C 10% o2/ExpDat_pw500.xlsx",
            "20%": "TG TEST 3 - ISOTHERMAL 250C 20% O2/ExpDat_PW500.xlsx",
        },
        "linear": {
            "5%": "TG TEST 7 . OXIDATION 600C 5% O2/ExpDat_PW500.xlsx",
            "10%": "TG TEST 6 - OXIDATION 600C 10% O2/ExpDat_PW500.xlsx",
            "20%": "TG TEST 8 - OXIDATION 600C 20% O2/ExpDat_PW500.xlsx",
        },
    },
}


def load_all_thermogravimetric_data(
    base_dir: Path | str,
    spec: Dict[str, Dict[str, Dict[str, str]]] = SPEC,
    loader: Optional[Callable[[str | Path], Any]] = None,
    raise_on_missing: bool = False,
) -> Dict[str, Dict[str, Dict[str, Optional[Any]]]]:
    """
    Load datasets defined in `spec` from `base_dir`.
    Returns nested dict: data[sample][regime][o2_label] -> DataFrame | None
    """
    if loader is None:
        loader = load_thermogravimetric_data  # use function from this module

    base = Path(base_dir)
    results: Dict[str, Dict[str, Dict[str, Optional[Any]]]] = {}

    for sample, regimes in spec.items():
        results[sample] = {}
        for regime, o2_map in regimes.items():
            results[sample][regime] = {}
            for o2_label, rel_path in o2_map.items():
                if not rel_path:
                    results[sample][regime][o2_label] = None
                    continue
                fp = base / rel_path
                if not fp.exists():
                    msg = f"missing file: {fp}"
                    if raise_on_missing:
                        raise FileNotFoundError(msg)
                    logging.warning(msg)
                    results[sample][regime][o2_label] = None
                    continue
                try:
                    results[sample][regime][o2_label] = loader(fp)
                except Exception as exc:
                    logging.exception("failed to load %s: %s", fp, exc)
                    results[sample][regime][o2_label] = None
    return results
