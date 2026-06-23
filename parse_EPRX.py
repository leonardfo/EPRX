import logging
import re
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


_RESULT_ARCHIVE_RE = re.compile(
    r"^(?P<period>\d{4,6})_(?P<code>[\d]+-[\d]+)_result\.(?P<ext>zip|csv)$",
    re.IGNORECASE,
)


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
    """
    Yield (period, raw_bytes) from flat archives in EPRX_DIR.

    Expected layout (no subfolders):
      {year}_{product_code}_result.zip   e.g. 2025_1-1_result.zip
      {yyyymm}_{product_code}_result.zip e.g. 202506_1-0_result.zip
    Each zip contains one or more CSV files with 確報値 data.
    """
    for path in sorted(EPRX_DIR.iterdir()):
        if not path.is_file():
            continue
        match = _RESULT_ARCHIVE_RE.match(path.name)
        if not match or match.group("code") != product_code:
            continue

        period = match.group("period")
        ext = match.group("ext").lower()

        if ext == "csv":
            yield period, path.read_bytes()
            continue

        with zipfile.ZipFile(path) as zf:
            for csv_name in sorted(n for n in zf.namelist() if n.lower().endswith(".csv")):
                yield period, zf.read(csv_name)


def load_eprx_product(product_code: str) -> pd.DataFrame:
    """
    Load confirmed (確報値) balancing-market data for one product code.

    For the last EPRX_RECENT_MONTHS calendar months, only result files are
    considered. Older months also use result only. Prompt / rt_prompt are
    never used as fallback.
    """
    dfs: list[pd.DataFrame] = []
    for period, raw in _iter_result_sources(product_code):
        df = parse_eprx_csv(raw, product_code=product_code)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


_REQ_ARCHIVE_RE = re.compile(r"^(?P<fiscal_year>\d{4})_Requirement_archive\.zip$", re.IGNORECASE)
_REQ_COMPOUND_RE = re.compile(
    r"^(?P<fiscal_year>\d{4})_(?P<table>Requirement|AdditionalRequirement)_(?P<update>\d{8})(?:\((?P<effective>\d{8})\))?\.csv$",
    re.IGNORECASE,
)
_REQ_T32_RE = re.compile(
    r"^(?P<fiscal_year>\d{4})_3-2_(?:\((?P<sigma_formula>[^)]+)\))?(?P<table>.+?)_(?P<update>\d{8})(?:\((?P<effective>\d{8})\))?\.csv$",
    re.IGNORECASE,
)
_DASH = r"[～〜]"
_MONTH_LABEL_RE = re.compile(rf"^(?P<year>\d{{4}})/(?P<month>\d{{1,2}})_?$")
_AREA_MONTH_RE = re.compile(rf"^(?P<area>.+?)\s*(?P<year>\d{{4}})/(?P<month>\d{{1,2}})$")
_TIME_SLOT_48_RE = re.compile(rf"^(?P<h>\d{{2}}):(?P<m>\d{{2}}){_DASH}")
_BLOCK_8_RE = re.compile(rf"^(?:ブロック\d+_)?(?P<start>\d+){_DASH}(?P<end>\d+)時$")
_QUANTILE_RE = re.compile(rf"^(?P<lo>\d+){_DASH}(?P<hi>\d+)%$")


def _parse_quantile(label: str) -> str | None:
    match = _QUANTILE_RE.match(label.strip())
    if not match:
        return None
    return f"q{match.group('lo')}_{match.group('hi')}"


def _parse_month_label(label: str) -> tuple[int, int]:
    match = _MONTH_LABEL_RE.match(label.strip())
    if not match:
        raise ValueError(f"Invalid month label: {label!r}")
    return int(match.group("year")), int(match.group("month"))


def _parse_area_month(label: str) -> tuple[str, int, int]:
    match = _AREA_MONTH_RE.match(label.strip())
    if not match:
        raise ValueError(f"Invalid area/month label: {label!r}")
    return match.group("area").strip(), int(match.group("year")), int(match.group("month"))


def _block_from_time_slot(slot: str) -> int:
    match = _TIME_SLOT_48_RE.match(slot.strip())
    if not match:
        raise ValueError(f"Invalid 30-min time slot: {slot!r}")
    return int(match.group("h")) * 2 + (1 if match.group("m") == "00" else 2)


def _block_from_hour_label(label: str) -> int:
    match = _BLOCK_8_RE.match(label.strip())
    if not match:
        raise ValueError(f"Invalid 3-hour block label: {label!r}")
    return int(match.group("start")) // 3 + 1


def _split_csv_groups(parts: list[str], group_size: int) -> list[list[str]]:
    groups: list[list[str]] = []
    for i in range(0, len(parts), group_size):
        chunk = parts[i : i + group_size]
        if len(chunk) == group_size:
            groups.append(chunk)
    return groups


def _parse_requirement_metadata(filename: str) -> dict:
    name = Path(filename).name
    meta = {
        "source_file": name,
        "fiscal_year": None,
        "product_code": None,
        "table_type": None,
        "sigma_formula": None,
        "update_date": None,
        "effective_date": None,
    }
    for pattern, product_code in ((_REQ_COMPOUND_RE, "4-0"),):
        match = pattern.match(name)
        if match:
            meta.update(
                fiscal_year=int(match.group("fiscal_year")),
                product_code=product_code,
                table_type=match.group("table"),
                update_date=match.group("update"),
                effective_date=match.group("effective"),
            )
            return meta
    match = _REQ_T32_RE.match(name)
    if match:
        meta.update(
            fiscal_year=int(match.group("fiscal_year")),
            product_code="3-2",
            table_type=match.group("table"),
            sigma_formula=match.group("sigma_formula"),
            update_date=match.group("update"),
            effective_date=match.group("effective"),
        )
        return meta
    return meta


def _detect_requirement_layout(header_row: str, first_data_row: str) -> str:
    if _QUANTILE_RE.match(first_data_row.split(",", 1)[0].strip()):
        return "quantile_8"
    if _TIME_SLOT_48_RE.match(first_data_row.split(",", 1)[0].strip()):
        return "wide_48"
    if "ブロック1" in header_row or "ブロック1" in header_row.replace("〜", "～"):
        return "area_row_8"
    if "0～3時" in header_row or "0〜3時" in header_row:
        return "quantile_8"
    raise ValueError("Unknown requirement-table layout")


def _parse_requirement_value(raw: str) -> float | None:
    value = raw.strip()
    if not value or value == "-":
        return None
    return float(value)


def _split_requirement_sections(lines: list[str]) -> list[tuple[str, list[str]]]:
    sections: list[tuple[str, list[str]]] = []
    title: str | None = None
    body: list[str] = []
    for line in lines:
        first = line.split(",", 1)[0].strip()
        is_title = (
            first.endswith("Requirement_[MW]")
            or first.endswith("ReductionFactor")
            or "RequirementTable" in first
            or first.endswith("[MW]")
        )
        if is_title:
            if title and body:
                sections.append((title, body))
            title = first
            body = []
            continue
        if title is not None and line.replace(",", "").strip():
            body.append(line)
    if title and body:
        sections.append((title, body))
    return sections


def _section_metadata(section_title: str, file_meta: dict) -> dict:
    meta = file_meta.copy()
    section = section_title.removesuffix("[MW]").rstrip("_")
    if section.startswith("Compound_Requirement"):
        meta.update(product_code="4-0", table_type="Requirement")
    elif section.startswith("1_Requirement"):
        meta.update(product_code="1-0", table_type="Requirement")
    elif match := re.match(r"^(\d-\d)_Requirement$", section):
        meta.update(product_code=match.group(1), table_type="Requirement")
    elif section.startswith("3-2_ReductionFactor"):
        meta.update(product_code="3-2", table_type="ReductionFactor")
    elif "RequirementTable" in section:
        meta.setdefault("product_code", "3-2")
        meta["table_type"] = section.split("_", 1)[-1] if "_" in section else section
    elif meta.get("table_type") is None:
        meta["table_type"] = section
    return meta


def _parse_requirement_section(section_title: str, section_lines: list[str], file_meta: dict) -> list[dict]:
    if len(section_lines) < 2:
        return []
    meta = _section_metadata(section_title, file_meta)
    table_type = meta["table_type"]
    header_row = section_lines[0]
    first_data = section_lines[1]
    layout = _detect_requirement_layout(header_row, first_data)
    header_parts = header_row.split(",")
    records: list[dict] = []

    if layout == "wide_48":
        month_groups = _split_csv_groups(header_parts, 10)
        for line in section_lines[1:]:
            for (month_label, *areas), (time_slot, *values) in zip(
                month_groups, _split_csv_groups(line.split(","), 10), strict=False
            ):
                if not time_slot.strip() or not month_label.strip():
                    continue
                if not _TIME_SLOT_48_RE.match(time_slot.strip()):
                    continue
                year, month = _parse_month_label(month_label)
                block = _block_from_time_slot(time_slot)
                for area, value in zip(areas, values, strict=False):
                    val = _parse_requirement_value(value)
                    if val is None:
                        continue
                    records.append(
                        {
                            "area": area.strip(),
                            "year": year,
                            "month": month,
                            "block": block,
                            "blocks_per_day": 48,
                            "quantile": None,
                            "value": val,
                            "table_type": table_type,
                            **meta,
                        }
                    )

    elif layout == "area_row_8":
        for line in section_lines[1:]:
            for group in _split_csv_groups(line.split(","), 9):
                if not group[0].strip():
                    continue
                area, year, month = _parse_area_month(group[0])
                for block_idx, value in enumerate(group[1:], start=1):
                    val = _parse_requirement_value(value)
                    if val is None:
                        continue
                    records.append(
                        {
                            "area": area,
                            "year": year,
                            "month": month,
                            "block": block_idx,
                            "blocks_per_day": 8,
                            "quantile": None,
                            "value": val,
                            "table_type": table_type,
                            **meta,
                        }
                    )

    else:  # quantile_8
        header_groups = _split_csv_groups(header_parts, 9)
        for line in section_lines[1:]:
            for header_group, data_group in zip(
                header_groups, _split_csv_groups(line.split(","), 9), strict=False
            ):
                if not header_group[0].strip() or not _QUANTILE_RE.match(data_group[0].strip()):
                    continue
                area, year, month = _parse_area_month(header_group[0])
                quantile = _parse_quantile(data_group[0])
                for block_label, value in zip(header_group[1:], data_group[1:], strict=False):
                    val = _parse_requirement_value(value)
                    if val is None:
                        continue
                    records.append(
                        {
                            "area": area,
                            "year": year,
                            "month": month,
                            "block": _block_from_hour_label(block_label),
                            "blocks_per_day": 8,
                            "quantile": quantile,
                            "value": val,
                            "table_type": table_type,
                            **meta,
                        }
                    )
    return records


def parse_requirement_csv(raw_bytes: bytes, source_name: str) -> pd.DataFrame:
    """Parse one 調整力必要量 CSV into long format."""
    lines = [line for line in _decode_csv(raw_bytes).splitlines() if line.strip()]
    if len(lines) < 2:
        return pd.DataFrame()

    file_meta = _parse_requirement_metadata(source_name)
    sections = _split_requirement_sections(lines)
    if not sections:
        sections = [(lines[0].split(",", 1)[0], lines[1:])]

    records: list[dict] = []
    for title, section_lines in sections:
        records.extend(_parse_requirement_section(title, section_lines, file_meta))
    return pd.DataFrame(records)


def _is_requirement_filename(name: str) -> bool:
    lower = name.lower()
    if _REQ_ARCHIVE_RE.match(name):
        return True
    if not lower.endswith(".csv"):
        return False
    return bool(
        _REQ_COMPOUND_RE.match(name)
        or _REQ_T32_RE.match(name)
        or re.match(r"^\d{4}_3-2_.+\.csv$", name, re.IGNORECASE)
    )


def _iter_requirement_sources():
    """Yield (display_name, raw_bytes) from EPRX_REQ_DIR and flat archives in EPRX_DIR."""
    seen: set[str] = set()
    for directory in (EPRX_REQ_DIR, EPRX_DIR):
        if not directory.is_dir():
            continue
        for path in sorted(directory.iterdir()):
            if not path.is_file():
                continue
            name = path.name
            if name in seen or not _is_requirement_filename(name):
                continue
            seen.add(name)
            if _REQ_ARCHIVE_RE.match(name):
                with zipfile.ZipFile(path) as zf:
                    for csv_name in sorted(n for n in zf.namelist() if n.lower().endswith(".csv")):
                        yield Path(csv_name).name, zf.read(csv_name)
            else:
                yield name, path.read_bytes()


def load_requirements() -> pd.DataFrame:
    dfs: list[pd.DataFrame] = []
    for source_name, raw in _iter_requirement_sources():
        df = parse_requirement_csv(raw, source_name)
        if not df.empty:
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def build_requirement_parquets() -> None:
    """Parse all local requirement CSV/ZIPs and save parquet files."""
    df = load_requirements()
    if df.empty:
        log.info("[EPRX] requirements: no data")
        return

    out_all = EPRX_REQ_DIR / "requirements.parquet"
    df.to_parquet(out_all, index=False)
    log.info(
        "[EPRX] %s: %s rows, tables=%s",
        out_all.name,
        f"{len(df):,}",
        ", ".join(sorted(df["table_type"].dropna().unique())),
    )

    compound = df[df["product_code"] == "4-0"]
    if not compound.empty:
        out = EPRX_REQ_DIR / "requirements_compound.parquet"
        compound.to_parquet(out, index=False)
        log.info("[EPRX] %s: %s rows", out.name, f"{len(compound):,}")

    tertiary2 = df[df["product_code"] == "3-2"]
    if not tertiary2.empty:
        out = EPRX_REQ_DIR / "requirements_tertiary-2.parquet"
        tertiary2.to_parquet(out, index=False)
        log.info("[EPRX] %s: %s rows", out.name, f"{len(tertiary2):,}")


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
    build_requirement_parquets()
