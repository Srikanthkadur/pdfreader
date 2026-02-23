from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Literal

import pandas as pd


KNOWN_HEADER_TOKENS = {
    "asset",
    "class",
    "folio",
    "no",
    "transaction",
    "date",
    "type",
    "quantity",
    "unit",
    "value",
    "purchase",
    "sale",
    "realised",
    "gain",
    "loss",
}


def _norm_cell(x: object) -> str:
    if x is None:
        return ""
    s = str(x)
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _looks_like_header_row(row: Iterable[object]) -> bool:
    joined = " ".join(_norm_cell(c).lower() for c in row)
    if not joined:
        return False
    hits = sum(1 for t in KNOWN_HEADER_TOKENS if t in joined)
    return hits >= 5


def normalize_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Camelot/pdfplumber often return tables where the first row contains the header.
    This normalizes that shape and cleans whitespace/newlines.
    """
    if df is None or df.empty:
        return df

    df = df.copy()
    df = df.applymap(_norm_cell)

    # Drop fully-empty rows/cols
    df = df.loc[~(df == "").all(axis=1)]
    df = df.loc[:, ~(df == "").all(axis=0)]

    if df.empty:
        return df

    first_row = df.iloc[0].tolist()
    if _looks_like_header_row(first_row):
        df.columns = [c if c else f"col_{i}" for i, c in enumerate(first_row)]
        df = df.iloc[1:].reset_index(drop=True)
    else:
        df.columns = [str(c) if str(c).strip() else f"col_{i}" for i, c in enumerate(df.columns)]

    # Extra cleanup
    df.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in df.columns]
    return df


@dataclass(frozen=True)
class ExtractResult:
    method: Literal["camelot", "pdfplumber", "ocr"]
    tables: list[pd.DataFrame]


def extract_with_camelot(
    pdf_path: str,
    *,
    pages: str = "1-end",
    flavor: Literal["lattice", "stream"] = "lattice",
) -> list[pd.DataFrame]:
    try:
        import camelot  # type: ignore
    except Exception:
        return []

    try:
        tables = camelot.read_pdf(pdf_path, pages=pages, flavor=flavor)
    except Exception:
        return []

    out: list[pd.DataFrame] = []
    for t in tables:
        try:
            df = normalize_table(t.df)
            if df is not None and not df.empty:
                out.append(df)
        except Exception:
            continue
    return out


def extract_with_pdfplumber(pdf_path: str, *, pages: str = "1-end") -> list[pd.DataFrame]:
    try:
        import pdfplumber  # type: ignore
    except Exception:
        return []

    def iter_page_numbers(num_pages: int) -> list[int]:
        if pages.strip().lower() in {"all", "1-end", "end"}:
            return list(range(1, num_pages + 1))
        out_pages: list[int] = []
        for part in pages.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                a, b = part.split("-", 1)
                a_i = int(a)
                b_i = num_pages if b.strip().lower() == "end" else int(b)
                out_pages.extend(range(a_i, b_i + 1))
            else:
                out_pages.append(int(part))
        # De-dup while preserving order
        seen: set[int] = set()
        unique: list[int] = []
        for p in out_pages:
            if 1 <= p <= num_pages and p not in seen:
                unique.append(p)
                seen.add(p)
        return unique

    out: list[pd.DataFrame] = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            wanted_pages = iter_page_numbers(len(pdf.pages))
            for pno in wanted_pages:
                page = pdf.pages[pno - 1]

                # Two attempts: line-based extraction first, then text-based.
                settings_list = [
                    {
                        "vertical_strategy": "lines",
                        "horizontal_strategy": "lines",
                        "intersection_tolerance": 5,
                        "snap_tolerance": 3,
                        "join_tolerance": 3,
                        "edge_min_length": 10,
                        "min_words_vertical": 1,
                        "min_words_horizontal": 1,
                        "text_tolerance": 3,
                    },
                    {
                        "vertical_strategy": "text",
                        "horizontal_strategy": "text",
                        "snap_tolerance": 3,
                        "join_tolerance": 3,
                        "text_tolerance": 3,
                    },
                ]

                for table_settings in settings_list:
                    try:
                        tables = page.extract_tables(table_settings=table_settings)
                    except Exception:
                        continue
                    for tbl in tables or []:
                        try:
                            df = normalize_table(pd.DataFrame(tbl))
                            if df is not None and not df.empty:
                                out.append(df)
                        except Exception:
                            continue
                    if out:
                        # Prefer first successful settings for the page
                        break
    except Exception:
        return []

    return out


def extract_tables_auto(
    pdf_path: str,
    *,
    pages: str = "1-end",
    camelot_flavor: Literal["lattice", "stream"] = "lattice",
) -> ExtractResult:
    tables = extract_with_camelot(pdf_path, pages=pages, flavor=camelot_flavor)
    if tables:
        return ExtractResult(method="camelot", tables=tables)

    tables = extract_with_pdfplumber(pdf_path, pages=pages)
    if tables:
        return ExtractResult(method="pdfplumber", tables=tables)

    # OCR is implemented in nitro_table_extractor.ocr to avoid heavy imports by default.
    try:
        from .ocr import extract_with_ocr  # noqa: WPS433

        tables = extract_with_ocr(pdf_path, pages=pages)
        if tables:
            return ExtractResult(method="ocr", tables=tables)
    except Exception:
        pass

    return ExtractResult(method="pdfplumber", tables=[])

