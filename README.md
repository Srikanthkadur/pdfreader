# NitroPDF table extraction (Python)

Extracts tabular data from NitroPDF-generated mutual fund statements like the screenshot (transaction history table). The CLI tries:

1. **Camelot** (best for grid/ruled tables)
2. **pdfplumber** (fallback for text-based tables)
3. **Tesseract OCR** (fallback for scanned/image PDFs)

## Install

### System dependencies (Ubuntu/Debian)

```bash
sudo apt-get update
sudo apt-get install -y ghostscript poppler-utils tesseract-ocr
```

If you see OpenCV-related errors, you may also need:

```bash
sudo apt-get install -y libgl1
```

### Python dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Auto mode (recommended):

```bash
python extract_nitro_tables.py "/path/to/statement.pdf" --pages 1-end --out out --method auto --format csv
```

Force a specific method:

```bash
python extract_nitro_tables.py "statement.pdf" --method camelot --camelot-flavor lattice
python extract_nitro_tables.py "statement.pdf" --method pdfplumber
python extract_nitro_tables.py "statement.pdf" --method ocr --dpi 350 --lang eng
```

Outputs are written to `out/` as `*.csv` (or `*.jsonl`) plus a `*_combined.*` file.

## Notes

- **Camelot requires Ghostscript** to be installed.
- **OCR mode** is slower and best-effort; for best accuracy use a higher DPI (300–400).
