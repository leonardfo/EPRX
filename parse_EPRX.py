import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta, date
from datetime import time as time_timedelta
from zoneinfo import ZoneInfo

# ============================================================
# EPRX CONFIGURATION
# ============================================================

EPRX_BASE = "https://www.eprx.or.jp"
EPRX_DIR = Path(r"C:\Develop\data\eprx")
EPRX_REQ_DIR = EPRX_DIR / "requirements"

EPRX_PRODUCTS = {
    "1-0": "primary",
    "1-1": "primary_offline",
    "2-1": "secondary-1",
    "2-2": "secondary-2",
    "3-1": "tertiary-1",
    "3-2": "tertiary-2",
    "4-0": "compound",
}

EPRX_START_YEAR = 2021
EPRX_END_YEAR = datetime.now(ZoneInfo("Asia/Tokyo")).year

EPRX_DIR.mkdir(parents=True, exist_ok=True)
EPRX_REQ_DIR.mkdir(parents=True, exist_ok=True)

print(f"EPRX dirs ready. RAW → {EPRX_DIR}")
print(f"Years {EPRX_START_YEAR}-{EPRX_END_YEAR}, products: {list(EPRX_PRODUCTS.keys())}")

# ============================================================
# EPRX PARSING FUNCTIONS
# ============================================================

def parse_eprx_csv(raw_bytes: bytes) -> pd.DataFrame:
    """
    Parse one EPRX result / prompt / rt_prompt CSV file.

    Two P-row formats detected by length of the date/period field (p_parts[2]):

    NEW (2024+) — monthly CSV, len(p_parts[2]) == 6  → "202404"
      P,確報値,{yyyymm},{product},{blocks_per_day}[,,,trailing commas...]
      data rows: {yyyymmdd}B{nn},{values...}

    OLD (2021-2023) — daily CSV, len(p_parts[2]) == 8  → "20210401"
      P,確報値,{yyyymmdd},{yyyymmdd},{product},{blocks_per_day}
      data rows: B{nn},{values...}

    Both formats may contain RT sub-header rows and an E end marker,
    and CSV lines may carry trailing empty fields (trailing commas).

    Returns DataFrame with: datetime, date, block, blocks_per_day,
    value_type, product, <all columns from TT/RT rows>
    """
    for enc in ("shift-jis", "utf-8", "cp932"):
        try:
            text = raw_bytes.decode(enc)
            break
        except UnicodeDecodeError:
            continue

    lines = text.splitlines()

    # P row — detect format by length of field[2], not by total field count
    p_parts = next(l for l in lines if l.startswith("P,")).split(",")
    value_type = p_parts[1]

    if len(p_parts[2]) == 8:
        # OLD daily format: P,type,yyyymmdd,yyyymmdd,product,blocks_per_day
        file_date = pd.Timestamp(p_parts[2])
        product = p_parts[4]
        blocks_per_day = int(p_parts[5].strip())

        def _parse_key(key: str):
            return file_date, int(key[1:])        # "B01" → (date, 1)

        def _is_data(key: str) -> bool:
            return key.startswith("B") and key[1:].isdigit()

    else:
        # NEW monthly format: P,type,yyyymm,product,blocks_per_day[,trailing...]
        file_date = None
        product = p_parts[3]
        blocks_per_day = int(p_parts[4].strip())

        def _parse_key(key: str):
            date_str, block_str = key.split("B", 1)
            return pd.Timestamp(date_str), int(block_str)  # "20260401B01" → (date, 1)

        def _is_data(key: str) -> bool:
            return key[:1].isdigit()

    minutes_per_block = (24 * 60) // blocks_per_day

    # Collect all section headers (TT and RT) in order with their column names
    section_headers: list[tuple[int, list[str]]] = []
    for i, line in enumerate(lines):
        prefix = line.split(",")[0]
        if prefix in ("TT", "RT"):
            section_headers.append((i, line.split(",")[1:]))

    # Parse each section's data rows
    records = []
    for sec_idx, (start_i, col_names) in enumerate(section_headers):
        end_i = section_headers[sec_idx + 1][0] if sec_idx + 1 < len(section_headers) else len(lines)
        for line in lines[start_i + 1 : end_i]:
            if not line.strip():
                continue
            parts = line.split(",")
            if not _is_data(parts[0]):
                continue
            date, block = _parse_key(parts[0])
            row = {
                "datetime": date + pd.Timedelta(minutes=(block - 1) * minutes_per_block),
                "date": date,
                "block": block,
                "blocks_per_day": blocks_per_day,
                "value_type": value_type,
                "product": product,
                **dict(zip(col_names, parts[1:])),
            }
            records.append(row)

    df = pd.DataFrame(records)
    str_cols = {"datetime", "date", "block", "blocks_per_day", "value_type", "product",
                "調達区分", "取引情報", "応札・落札結果"}
    for col in df.columns:
        if col not in str_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_eprx_product_type(product_code: str, file_type: str) -> pd.DataFrame:
    """
    Concatenate all parsed CSVs for a product code and file_type from local ZIPs.
    Looks in EPRX_DIR (flat layout).
    file_type: 'result' | 'prompt' | 'rt_prompt'
    """
    dfs = []
    for zip_path in sorted(EPRX_DIR.glob(f"*_{product_code}_{file_type}.zip")):
        with zipfile.ZipFile(zip_path) as zf:
            for csv_name in (n for n in zf.namelist() if n.endswith(".csv")):
                raw = zf.read(csv_name)
                df = parse_eprx_csv(raw)
                dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def build_eprx_parquets():
    """Parse all local ZIPs and save per-category parquets to EPRX_DIR."""
    for code, category in EPRX_PRODUCTS.items():
        for file_type in ("result", "prompt", "rt_prompt"):
            df = load_eprx_product_type(code, file_type)
            if df.empty:
                log.info(f"[EPRX] {code}/{file_type}: no data (skip)")
                continue
            out = EPRX_DIR / f"{category}_{file_type}.parquet"
            df.to_parquet(out, index=False)
            log.info(
                f"[EPRX] {out.name}: {len(df):,} rows  "
                f"{df['date'].min().date()} → {df['date'].max().date()}"
            )
    print("EPRX parquets built.")


print("EPRX parsing functions ready.")