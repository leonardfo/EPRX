import logging
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from dateutil.relativedelta import relativedelta

# ============================================================
# EPRX CONFIGURATION — balancing market (需給調整力)
# ============================================================

EPRX_BASE = "https://www.eprx.or.jp"
EPRX_DIR = Path(__file__).resolve().parent
EPRX_REQ_DIR = EPRX_DIR / "requirements"

# code → (category, Japanese name)
EPRX_PRODUCTS: dict[str, tuple[str, str]] = {
    "1-0": ("primary", "一次調整力"),
    "1-1": ("primary_offline", "一次オフライン"),
    "2-1": ("secondary-1", "二次調整力1"),
    "2-2": ("secondary-2", "二次調整力2"),
    "3-1": ("tertiary-1", "三次調整力1"),
    "3-2": ("tertiary-2", "三次調整力2"),
    "4-0": ("compound", "複合商品"),
}

EPRX_JP_TO_CODE = {jp: code for code, (_, jp) in EPRX_PRODUCTS.items()}
EPRX_VALUE_RESULT = "確報値"  # confirmed; never use 速報値 (prompt)

EPRX_START_YEAR = 2021
EPRX_END_YEAR = datetime.now(ZoneInfo("Asia/Tokyo")).year
EPRX_RECENT_MONTHS = 2

log = logging.getLogger(__name__)

EPRX_DIR.mkdir(parents=True, exist_ok=True)
EPRX_REQ_DIR.mkdir(parents=True, exist_ok=True)


def recent_yyyymm_threshold() -> str:
    """First yyyymm included in the rolling recent window (current + previous month)."""
    now = datetime.now(ZoneInfo("Asia/Tokyo"))
    first_of_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    return (first_of_month - relativedelta(months=EPRX_RECENT_MONTHS - 1)).strftime("%Y%m")


def _decode_csv(raw_bytes: bytes) -> str:
    for enc in ("shift-jis", "cp932", "utf-8"):
        try:
            return raw_bytes.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw_bytes.decode("shift-jis", errors="replace")


def _resolve_product_code(product_field: str, product_code: str | None) -> str:
    if product_code and product_code in EPRX_PRODUCTS:
        return product_code
    if product_field in EPRX_JP_TO_CODE:
        return EPRX_JP_TO_CODE[product_field]
    if product_field in EPRX_PRODUCTS:
        return product_field
    raise ValueError(f"Unknown balancing product: {product_field!r}")


# ============================================================
# EPRX PARSING
# ============================================================

def parse_eprx_csv(raw_bytes: bytes, product_code: str | None = None) -> pd.DataFrame:
    """
    Parse one EPRX balancing-market result CSV.

    P-row formats:

    NEW monthly (2024+):
      P,確報値,{yyyymm},{product_jp},{blocks_per_day}
      data: {yyyymmdd}B{block},{metric},{values...}

    OLD daily (2021-2023):
      P,確報値,{yyyymmdd},{yyyymmdd},{product},{blocks_per_day}
      data: B{block},{metric},{values...}

    blocks_per_day is 48 (30 min) or 8 (3 h).
    Date keys use the form {yyyymmdd}B{block}; block is 1-based.
  """
    lines = _decode_csv(raw_bytes).splitlines()
    p_parts = next(line for line in lines if line.startswith("P,")).split(",")

    value_type = p_parts[1].strip()
    if value_type != EPRX_VALUE_RESULT:
        raise ValueError(f"Expected {EPRX_VALUE_RESULT!r}, got {value_type!r}")

    if len(p_parts[2].strip()) == 8:
        file_date = pd.Timestamp(p_parts[2].strip())
        product_field = p_parts[4].strip()
        blocks_per_day = int(p_parts[5].strip())

        def _parse_key(key: str) -> tuple[pd.Timestamp, int]:
            return file_date, int(key[1:])

        def _is_data(key: str) -> bool:
            return key.startswith("B") and key[1:].isdigit()

    else:
        if p_parts[3].strip().isdigit():
            raise ValueError("Not a balancing-market CSV (tieline/interconnector format)")
        product_field = p_parts[3].strip()
        blocks_per_day = int(p_parts[4].strip())

        def _parse_key(key: str) -> tuple[pd.Timestamp, int]:
            date_str, block_str = key.split("B", 1)
            return pd.Timestamp(date_str), int(block_str)

        def _is_data(key: str) -> bool:
            return "B" in key and key.split("B", 1)[0].isdigit()

    code = _resolve_product_code(product_field, product_code)
    category, _ = EPRX_PRODUCTS[code]
    minutes_per_block = (24 * 60) // blocks_per_day

    section_headers: list[tuple[int, list[str]]] = []
    for i, line in enumerate(lines):
        prefix = line.split(",", 1)[0]
        if prefix in ("TT", "RT", "T"):
            section_headers.append((i, line.split(",")[1:]))

    records: list[dict] = []
    for sec_idx, (start_i, col_names) in enumerate(section_headers):
        end_i = section_headers[sec_idx + 1][0] if sec_idx + 1 < len(section_headers) else len(lines)
        for line in lines[start_i + 1 : end_i]:
            if not line.strip():
                continue
            parts = line.split(",")
            if not _is_data(parts[0]):
                continue
            row_date, block = _parse_key(parts[0])
            records.append(
                {
                    "datetime": row_date + pd.Timedelta(minutes=(block - 1) * minutes_per_block),
                    "date": row_date,
                    "block": block,
                    "blocks_per_day": blocks_per_day,
                    "value_type": value_type,
                    "product_code": code,
                    "product": category,
                    **dict(zip(col_names, parts[1:])),
                }
            )

    df = pd.DataFrame(records)
    str_cols = {
        "datetime",
        "date",
        "block",
        "blocks_per_day",
        "value_type",
        "product_code",
        "product",
        "調達区分",
        "取引情報",
        "応札・落札結果",
        "順方向原エリア",
        "逆方向原エリア",
    }
    for col in df.columns:
        if col not in str_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _iter_result_sources(product_code: str):
    """Yield (yyyymm, raw_bytes) for result archives only — no prompt fallback."""
    seen: set[str] = set()
    patterns = (
        f"*_{product_code}_result.csv",
        f"*_{product_code}_result.zip",
    )
    for pattern in patterns:
        for path in sorted(EPRX_DIR.glob(pattern)):
            yyyymm = path.name[:6]
            if not yyyymm.isdigit() or yyyymm in seen:
                continue
            seen.add(yyyymm)
            if path.suffix == ".csv":
                yield yyyymm, path.read_bytes()
            else:
                with zipfile.ZipFile(path) as zf:
                    for csv_name in sorted(n for n in zf.namelist() if n.endswith(".csv")):
                        yield yyyymm, zf.read(csv_name)


def load_eprx_product(product_code: str) -> pd.DataFrame:
    """
    Load confirmed (確報値) balancing-market data for one product code.

    For the last EPRX_RECENT_MONTHS calendar months, only result files are
    considered. Older months also use result only. Prompt / rt_prompt are
    never used as fallback.
    """
    dfs: list[pd.DataFrame] = []
    for yyyymm, raw in _iter_result_sources(product_code):
        df = parse_eprx_csv(raw, product_code=product_code)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def build_eprx_parquets() -> None:
    """Parse all local result CSV/ZIPs and save per-category parquets."""
    for code, (category, _) in EPRX_PRODUCTS.items():
        df = load_eprx_product(code)
        if df.empty:
            log.info("[EPRX] %s: no result data", code)
            continue
        out = EPRX_DIR / f"{category}_result.parquet"
        df.to_parquet(out, index=False)
        log.info(
            "[EPRX] %s → %s: %s rows, %s → %s",
            code,
            out.name,
            f"{len(df):,}",
            df["date"].min().date(),
            df["date"].max().date(),
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print(f"EPRX dir: {EPRX_DIR}")
    print(f"Products: {list(EPRX_PRODUCTS)}")
    print(f"Recent window from yyyymm={recent_yyyymm_threshold()} (result only)")
    build_eprx_parquets()
