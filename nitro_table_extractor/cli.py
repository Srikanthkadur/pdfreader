from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

import pandas as pd

from .extract import extract_tables_auto, extract_with_camelot, extract_with_pdfplumber


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_tables(
    tables: list[pd.DataFrame],
    *,
    out_dir: Path,
    prefix: str,
    fmt: Literal["csv", "jsonl"] = "csv",
) -> list[Path]:
    _ensure_dir(out_dir)
    written: list[Path] = []

    for i, df in enumerate(tables, start=1):
        if fmt == "csv":
            out_path = out_dir / f"{prefix}_{i:02d}.csv"
            df.to_csv(out_path, index=False)
        else:
            out_path = out_dir / f"{prefix}_{i:02d}.jsonl"
            with out_path.open("w", encoding="utf-8") as f:
                for rec in df.to_dict(orient="records"):
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        written.append(out_path)

    if tables:
        # Combined output
        combined = pd.concat(tables, ignore_index=True, sort=False)
        if fmt == "csv":
            combined_path = out_dir / f"{prefix}_combined.csv"
            combined.to_csv(combined_path, index=False)
        else:
            combined_path = out_dir / f"{prefix}_combined.jsonl"
            with combined_path.open("w", encoding="utf-8") as f:
                for rec in combined.to_dict(orient="records"):
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        written.append(combined_path)

    return written


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract transaction-history tables from NitroPDF statements (Camelot/pdfplumber + OCR fallback).",
    )
    p.add_argument("pdf", help="Path to the PDF statement")
    p.add_argument("--pages", default="1-end", help='Pages to parse, e.g. "1", "1-3", "1,3,5-7", or "1-end"')
    p.add_argument("--out", default="out", help="Output directory")
    p.add_argument("--format", choices=["csv", "jsonl"], default="csv", help="Output format")
    p.add_argument(
        "--method",
        choices=["auto", "camelot", "pdfplumber", "ocr"],
        default="auto",
        help="Extraction method (auto tries camelot -> pdfplumber -> ocr)",
    )
    p.add_argument("--camelot-flavor", choices=["lattice", "stream"], default="lattice", help="Camelot flavor")
    p.add_argument("--dpi", type=int, default=300, help="OCR DPI (only used for --method ocr/auto fallback)")
    p.add_argument("--lang", default="eng", help="Tesseract language (only used for --method ocr/auto fallback)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    pdf_path = str(Path(args.pdf))
    out_dir = Path(args.out)
    fmt: Literal["csv", "jsonl"] = args.format

    if args.method == "auto":
        res = extract_tables_auto(pdf_path, pages=args.pages, camelot_flavor=args.camelot_flavor)
        tables = res.tables
        prefix = f"{Path(pdf_path).stem}_{res.method}"
        _write_tables(tables, out_dir=out_dir, prefix=prefix, fmt=fmt)
        return 0 if tables else 2

    if args.method == "camelot":
        tables = extract_with_camelot(pdf_path, pages=args.pages, flavor=args.camelot_flavor)
        prefix = f"{Path(pdf_path).stem}_camelot"
        _write_tables(tables, out_dir=out_dir, prefix=prefix, fmt=fmt)
        return 0 if tables else 2

    if args.method == "pdfplumber":
        tables = extract_with_pdfplumber(pdf_path, pages=args.pages)
        prefix = f"{Path(pdf_path).stem}_pdfplumber"
        _write_tables(tables, out_dir=out_dir, prefix=prefix, fmt=fmt)
        return 0 if tables else 2

    if args.method == "ocr":
        from .ocr import extract_with_ocr

        tables = extract_with_ocr(pdf_path, pages=args.pages, dpi=args.dpi, lang=args.lang)
        prefix = f"{Path(pdf_path).stem}_ocr"
        _write_tables(tables, out_dir=out_dir, prefix=prefix, fmt=fmt)
        return 0 if tables else 2

    raise AssertionError("unreachable")

