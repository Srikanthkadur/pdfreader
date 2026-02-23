"""
Extract tabular data from NitroPDF-generated mutual fund statement PDFs.

Supports two extraction backends:
  1. pdfplumber (default) - pure-Python, no Java dependency
  2. tabula-py (fallback) - requires a Java runtime

Usage:
    python read_nitropdf_table.py statement.pdf
    python read_nitropdf_table.py statement.pdf --output result.csv
    python read_nitropdf_table.py statement.pdf --output result.xlsx
    python read_nitropdf_table.py statement.pdf --backend tabula
    python read_nitropdf_table.py statement.pdf --pages 1,2,3
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

EXPECTED_COLUMNS = [
    "Asset Class",
    "Folio No.",
    "Transaction Date",
    "Transaction Type",
    "Quantity",
    "Unit Value",
    "Purchase / Sale Value",
    "Realised Gain or Loss",
]


# ---------------------------------------------------------------------------
# pdfplumber backend
# ---------------------------------------------------------------------------

def extract_with_pdfplumber(pdf_path: str, pages: list[int] | None = None) -> pd.DataFrame:
    """Extract tables from a PDF using pdfplumber."""
    import pdfplumber

    all_rows: list[list[str]] = []
    header: list[str] | None = None

    with pdfplumber.open(pdf_path) as pdf:
        page_indices = range(len(pdf.pages)) if pages is None else [p - 1 for p in pages]

        for idx in page_indices:
            if idx < 0 or idx >= len(pdf.pages):
                continue
            page = pdf.pages[idx]

            tables = page.extract_tables(
                table_settings={
                    "vertical_strategy": "lines",
                    "horizontal_strategy": "lines",
                    "snap_tolerance": 5,
                    "join_tolerance": 5,
                    "min_words_vertical": 2,
                    "min_words_horizontal": 1,
                }
            )

            if not tables:
                tables = page.extract_tables(
                    table_settings={
                        "vertical_strategy": "text",
                        "horizontal_strategy": "text",
                        "snap_tolerance": 5,
                        "join_tolerance": 5,
                    }
                )

            for table in tables:
                for i, row in enumerate(table):
                    cleaned = [cell.strip() if cell else "" for cell in row]
                    if not any(cleaned):
                        continue

                    if header is None and _looks_like_header(cleaned):
                        header = cleaned
                        continue

                    if _looks_like_header(cleaned):
                        continue

                    all_rows.append(cleaned)

    if not all_rows:
        print("WARNING: No table rows found. The PDF may use a non-standard layout.")
        return pd.DataFrame()

    if header and len(header) == len(all_rows[0]):
        df = pd.DataFrame(all_rows, columns=header)
    else:
        df = pd.DataFrame(all_rows)
        if df.shape[1] == len(EXPECTED_COLUMNS):
            df.columns = EXPECTED_COLUMNS

    return _post_process(df)


def _looks_like_header(row: list[str]) -> bool:
    """Heuristic: a row is a header if it contains keywords like 'Folio' or 'Transaction'."""
    joined = " ".join(row).lower()
    keywords = ["folio", "transaction", "asset class", "unit value", "quantity"]
    return sum(kw in joined for kw in keywords) >= 2


# ---------------------------------------------------------------------------
# tabula-py backend
# ---------------------------------------------------------------------------

def extract_with_tabula(pdf_path: str, pages: list[int] | None = None) -> pd.DataFrame:
    """Extract tables from a PDF using tabula-py (requires Java)."""
    import tabula

    page_spec = "all" if pages is None else ",".join(str(p) for p in pages)

    dfs = tabula.read_pdf(
        pdf_path,
        pages=page_spec,
        multiple_tables=True,
        lattice=True,
        pandas_options={"header": None},
    )

    if not dfs:
        dfs = tabula.read_pdf(
            pdf_path,
            pages=page_spec,
            multiple_tables=True,
            stream=True,
            pandas_options={"header": None},
        )

    if not dfs:
        print("WARNING: tabula found no tables in the PDF.")
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.dropna(how="all")

    mask = combined.apply(lambda row: not _looks_like_header(row.astype(str).tolist()), axis=1)
    combined = combined[mask].reset_index(drop=True)

    if combined.shape[1] == len(EXPECTED_COLUMNS):
        combined.columns = EXPECTED_COLUMNS

    return _post_process(combined)


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def _post_process(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and type-cast the extracted dataframe."""
    if df.empty:
        return df

    col_map = {c.lower().strip(): c for c in df.columns}

    numeric_cols = ["quantity", "unit value", "purchase / sale value", "realised gain or loss"]
    for nc in numeric_cols:
        matched = col_map.get(nc)
        if matched is None:
            continue
        df[matched] = (
            df[matched]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        df[matched] = pd.to_numeric(df[matched], errors="coerce")

    date_col = col_map.get("transaction date")
    if date_col:
        df[date_col] = pd.to_datetime(
            df[date_col].astype(str).str.strip(),
            dayfirst=True,
            errors="coerce",
        )

    return df


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def save_output(df: pd.DataFrame, output_path: str) -> None:
    ext = Path(output_path).suffix.lower()
    if ext == ".xlsx":
        df.to_excel(output_path, index=False, engine="openpyxl")
    elif ext == ".json":
        df.to_json(output_path, orient="records", indent=2, date_format="iso")
    else:
        df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} rows to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract table data from a NitroPDF mutual-fund statement."
    )
    parser.add_argument("pdf", help="Path to the PDF file")
    parser.add_argument(
        "--output", "-o",
        help="Output file path (.csv, .xlsx, or .json). Prints to stdout if omitted.",
    )
    parser.add_argument(
        "--backend", "-b",
        choices=["pdfplumber", "tabula"],
        default="pdfplumber",
        help="Extraction backend (default: pdfplumber)",
    )
    parser.add_argument(
        "--pages", "-p",
        help="Comma-separated page numbers to process (default: all)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    pdf_path = args.pdf
    if not Path(pdf_path).is_file():
        sys.exit(f"Error: file not found: {pdf_path}")

    pages = None
    if args.pages:
        pages = [int(p.strip()) for p in args.pages.split(",")]

    if args.backend == "tabula":
        df = extract_with_tabula(pdf_path, pages)
    else:
        df = extract_with_pdfplumber(pdf_path, pages)

    if df.empty:
        sys.exit("No data extracted. Try a different --backend or check the PDF.")

    if args.output:
        save_output(df, args.output)
    else:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 200)
        pd.set_option("display.max_colwidth", 40)
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
