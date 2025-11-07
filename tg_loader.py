from pathlib import Path
from io import StringIO
import pandas as pd
from typing import Iterable, Union

HEADER_MARKER = "##Temp./°C;Time/min;Mass/%;Segment"


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
            df = pd.read_csv(buf, sep=";", decimal=decimal, engine="python")
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

    if ext in {".xlsx", ".xlsm", ".xls"}:
        xl = pd.ExcelFile(path)
        expected = ["Temp./°C", "Time/min", "Mass/%", "Segment"]
        for sheet in xl.sheet_names:
            df = xl.parse(sheet_name=sheet, header=None, dtype=str)
            # search for a row that exactly matches the expected header sequence
            for i in range(len(df)):
                row = [str(v).strip() if v is not None else "" for v in df.iloc[i].tolist()]
                for start in range(0, max(0, len(row) - len(expected) + 1)):
                    if row[start:start + 4] == expected:
                        data = df.iloc[i + 1:, start:start + 4].copy()
                        data.columns = expected
                        # Trim trailing empty rows
                        mask_empty = (data.replace("", pd.NA).isna()).all(axis=1)
                        if (~mask_empty).any():
                            last = mask_empty[~mask_empty].index[-1]
                            data = data.loc[:last]
                        # Coerce numerics (accept comma decimals)
                        for col in ["Temp./°C", "Time/min", "Mass/%"]:
                            data[col] = (
                                data[col].astype(str)
                                .str.replace(",", ".", regex=False)
                                .pipe(pd.to_numeric, errors="coerce")
                            )
                        return data.rename(columns={
                            "Temp./°C": "temp_C",
                            "Time/min": "time_min",
                            "Mass/%": "mass_pct",
                            "Segment": "segment",
                        }).dropna(how="all")
        raise ValueError("Could not locate the header row in any Excel sheet.")

    raise ValueError(f"Unsupported file type: {ext}")


def load_many_thermogravimetric(
        paths: Union[Iterable[Union[str, Path]], str],
        *,
        use_glob: bool = False,
        as_dict: bool = False,
        ignore_errors: bool = True
):
    """
    Load multiple CSV/XLSX files into DataFrames.

    paths:
        - Iterable of paths (['a.csv','b.xlsx', ...]) OR
        - A glob string (e.g. 'data/**/*.csv') with use_glob=True
    use_glob:
        If True, interpret `paths` as a single glob pattern.
    as_dict:
        If True, return a dict keyed by file stem; else a list in the same order.
    ignore_errors:
        If True, skip files that fail to parse; else raise.

    Returns:
        list[pd.DataFrame] or dict[str, pd.DataFrame]
    """
    if use_glob:
        file_list = sorted(Path().glob(paths))
    else:
        file_list = [Path(p) for p in paths]

    results_list = []
    results_dict = {}

    for p in file_list:
        try:
            df = load_thermogravimetric_data(p)
            if as_dict:
                key = p.stem
                # If duplicate stems, make them unique
                i, base = 1, key
                while key in results_dict:
                    i += 1
                    key = f"{base}__{i}"
                results_dict[key] = df
            else:
                results_list.append(df)
        except Exception as e:
            if ignore_errors:
                print(f"[skip] {p}: {e}")
                continue
            raise

    return results_dict if as_dict else results_list
