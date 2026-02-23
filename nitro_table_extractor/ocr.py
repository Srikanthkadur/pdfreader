from __future__ import annotations

import re
from typing import Iterable

import pandas as pd

DATE_RE = re.compile(r"\b\d{2}/\d{2}/\d{4}\b")


def _norm_word(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s)
    return s


def _tokenize(s: str) -> list[str]:
    s = _norm_word(s).lower()
    s = re.sub(r"[^a-z0-9/ ]+", " ", s)
    return [t for t in s.split() if t]


def _parse_pages_spec(pages: str, num_pages: int) -> list[int]:
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

    seen: set[int] = set()
    unique: list[int] = []
    for p in out_pages:
        if 1 <= p <= num_pages and p not in seen:
            unique.append(p)
            seen.add(p)
    return unique


def _pick_header_line(lines: pd.DataFrame) -> tuple[tuple[int, int, int], pd.DataFrame] | None:
    """
    Returns the (block, par, line) key and the words for that line.
    """
    expected = {
        "asset",
        "class",
        "folio",
        "no",
        "transaction",
        "date",
        "type",
        "quantity",
        "unit",
        "purchase",
        "sale",
        "realised",
        "gain",
        "loss",
    }

    best_key: tuple[int, int, int] | None = None
    best_score = 0

    for key, grp in lines.groupby(["block_num", "par_num", "line_num"], sort=False):
        text = " ".join(_norm_word(t) for t in grp["text"].tolist())
        toks = set(_tokenize(text))
        score = sum(1 for t in expected if t in toks)
        if score > best_score:
            best_score = score
            best_key = (int(key[0]), int(key[1]), int(key[2]))

    if best_key is None or best_score < 6:
        return None

    header_words = lines[
        (lines["block_num"] == best_key[0])
        & (lines["par_num"] == best_key[1])
        & (lines["line_num"] == best_key[2])
    ].copy()
    return best_key, header_words


def _col_starts_from_header(header_words: pd.DataFrame) -> dict[str, float]:
    """
    Determine approximate x-start per column from the OCR header line.
    """
    hw = header_words.copy()
    hw["t"] = hw["text"].astype(str).map(_norm_word).str.lower()

    def x0_of(predicate) -> float | None:
        sub = hw[predicate(hw)]
        if sub.empty:
            return None
        return float(sub["left"].min())

    starts: dict[str, float] = {}

    starts["asset_class"] = x0_of(lambda d: d["t"].isin(["asset", "class"]))
    starts["folio_no"] = x0_of(lambda d: d["t"].isin(["folio", "no", "no."]))
    # Disambiguate the two "Transaction" groups by anchoring on unique tokens.
    starts["transaction_date"] = x0_of(lambda d: d["t"].isin(["date"]))
    starts["transaction_type"] = x0_of(lambda d: d["t"].isin(["type"]))
    starts["quantity"] = x0_of(lambda d: d["t"].isin(["quantity"]))
    starts["unit_value"] = x0_of(lambda d: d["t"].isin(["unit"]))
    starts["purchase_sale_value"] = x0_of(lambda d: d["t"].isin(["purchase", "sale"]))
    starts["realised_gain_loss"] = x0_of(lambda d: d["t"].isin(["realised", "gain", "loss"]))

    # Fill any missing start by monotonic interpolation across known starts.
    ordered_cols = [
        "asset_class",
        "folio_no",
        "transaction_date",
        "transaction_type",
        "quantity",
        "unit_value",
        "purchase_sale_value",
        "realised_gain_loss",
    ]
    known = [(c, starts[c]) for c in ordered_cols if starts.get(c) is not None]
    if len(known) < 4:
        # Too weak to be trustworthy
        return {}

    # Ensure monotonic order by sorting by x0 and re-mapping to expected order if needed.
    known_sorted = sorted(known, key=lambda x: float(x[1]))  # type: ignore[arg-type]
    min_x = float(known_sorted[0][1])  # type: ignore[index]
    max_x = float(known_sorted[-1][1])  # type: ignore[index]
    if max_x - min_x < 100:
        return {}

    # Simple forward-fill using next known start or linear spacing.
    last_x: float | None = None
    for c in ordered_cols:
        x = starts.get(c)
        if x is None:
            continue
        if last_x is not None and x < last_x:
            # Header mis-detected; bail out.
            return {}
        last_x = x

    # Forward/back fill using nearest known values.
    first_known_idx = min(i for i, c in enumerate(ordered_cols) if starts.get(c) is not None)
    last_known_idx = max(i for i, c in enumerate(ordered_cols) if starts.get(c) is not None)

    for i in range(first_known_idx - 1, -1, -1):
        # Place missing columns slightly to the left of first known, evenly
        next_col = ordered_cols[i + 1]
        starts[ordered_cols[i]] = float(starts[next_col]) - 80.0

    for i in range(last_known_idx + 1, len(ordered_cols)):
        prev_col = ordered_cols[i - 1]
        starts[ordered_cols[i]] = float(starts[prev_col]) + 120.0

    # Fill any gaps between known points.
    for i, c in enumerate(ordered_cols):
        if starts.get(c) is not None:
            continue
        # Find prev and next known.
        j = i - 1
        while j >= 0 and starts.get(ordered_cols[j]) is None:
            j -= 1
        k = i + 1
        while k < len(ordered_cols) and starts.get(ordered_cols[k]) is None:
            k += 1
        if j >= 0 and k < len(ordered_cols):
            starts[c] = (float(starts[ordered_cols[j]]) + float(starts[ordered_cols[k]])) / 2.0
        elif j >= 0:
            starts[c] = float(starts[ordered_cols[j]]) + 120.0
        elif k < len(ordered_cols):
            starts[c] = float(starts[ordered_cols[k]]) - 80.0

    # Keep expected order
    return {c: float(starts[c]) for c in ordered_cols}


def _boundaries_from_starts(starts: dict[str, float]) -> tuple[list[str], list[float]]:
    cols = list(starts.keys())
    xs = [starts[c] for c in cols]
    # boundaries are midpoints between consecutive starts
    bounds: list[float] = [-1e9]
    for a, b in zip(xs, xs[1:]):
        bounds.append((a + b) / 2.0)
    bounds.append(1e9)
    return cols, bounds


def _assign_words_to_columns(words: pd.DataFrame, cols: list[str], bounds: list[float]) -> dict[str, str]:
    if words.empty:
        return {c: "" for c in cols}

    row = {c: [] for c in cols}
    for _, w in words.iterrows():
        txt = _norm_word(str(w["text"]))
        if not txt:
            continue
        x_center = float(w["left"]) + float(w["width"]) / 2.0
        # find interval
        idx = 0
        for i in range(len(bounds) - 1):
            if bounds[i] <= x_center < bounds[i + 1]:
                idx = i
                break
        idx = max(0, min(idx, len(cols) - 1))
        row[cols[idx]].append(txt)

    return {c: " ".join(v).strip() for c, v in row.items()}


def _is_new_data_row(r: dict[str, str]) -> bool:
    if DATE_RE.search(r.get("transaction_date", "")):
        return True
    # OCR sometimes pushes date into neighbor column
    for v in r.values():
        if DATE_RE.search(v):
            return True
    return False


def extract_with_ocr(
    pdf_path: str,
    *,
    pages: str = "1-end",
    dpi: int = 300,
    lang: str = "eng",
) -> list[pd.DataFrame]:
    """
    OCR fallback: best-effort extraction using Tesseract word boxes and header-based column
    boundaries. Returns one DataFrame per page that contains a detected header.
    """
    try:
        import pdfplumber  # type: ignore
        import pytesseract  # type: ignore
        from pdf2image import convert_from_path  # type: ignore
    except Exception:
        return []

    out: list[pd.DataFrame] = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            wanted_pages = _parse_pages_spec(pages, len(pdf.pages))
    except Exception:
        return []

    for pno in wanted_pages:
        try:
            images = convert_from_path(pdf_path, dpi=dpi, first_page=pno, last_page=pno)
        except Exception:
            continue
        if not images:
            continue
        img = images[0]

        try:
            df = pytesseract.image_to_data(img, lang=lang, output_type=pytesseract.Output.DATAFRAME, config="--psm 6")
        except Exception:
            continue

        if df is None or df.empty:
            continue

        words = df.copy()
        # Normalize
        words = words.dropna(subset=["text"])
        words["text"] = words["text"].astype(str).map(_norm_word)
        words = words[words["text"] != ""]

        if words.empty:
            continue

        # Confidence filtering if available.
        if "conf" in words.columns:
            words["conf"] = pd.to_numeric(words["conf"], errors="coerce")
            words = words[(words["conf"].isna()) | (words["conf"] >= 30)]

        header_pick = _pick_header_line(words)
        if header_pick is None:
            continue
        (hb, hp, hl), header_words = header_pick

        starts = _col_starts_from_header(header_words)
        if not starts:
            continue
        cols, bounds = _boundaries_from_starts(starts)

        # Iterate lines after header line.
        data_rows: list[dict[str, str]] = []
        last_row: dict[str, str] | None = None

        # Create a stable line ordering key (top then left).
        line_keys: list[tuple[int, int, int]] = []
        for key, grp in words.groupby(["block_num", "par_num", "line_num"], sort=False):
            kb, kp, kl = int(key[0]), int(key[1]), int(key[2])
            # Skip lines before header in same block/paragraph ordering heuristics:
            if (kb, kp, kl) == (hb, hp, hl):
                continue
            # Keep only lines visually below the header line based on y coordinate.
            grp_top = float(grp["top"].min())
            header_top = float(header_words["top"].min())
            if grp_top <= header_top:
                continue
            line_keys.append((kb, kp, kl))

        # Sort lines by their y position.
        def line_sort_key(k: tuple[int, int, int]) -> tuple[float, float]:
            grp = words[(words["block_num"] == k[0]) & (words["par_num"] == k[1]) & (words["line_num"] == k[2])]
            return float(grp["top"].min()), float(grp["left"].min())

        line_keys_sorted = sorted(set(line_keys), key=line_sort_key)

        for key in line_keys_sorted:
            grp = words[(words["block_num"] == key[0]) & (words["par_num"] == key[1]) & (words["line_num"] == key[2])]
            row = _assign_words_to_columns(grp, cols, bounds)

            # Filter obvious noise lines
            non_empty = sum(1 for v in row.values() if v)
            if non_empty < 2:
                continue
            if "statement date" in " ".join(row.values()).lower():
                continue

            if _is_new_data_row(row):
                data_rows.append(row)
                last_row = row
            else:
                # Continuation line: append to asset_class if possible, else to transaction_type
                if last_row is not None:
                    cont = " ".join(v for v in row.values() if v).strip()
                    if cont:
                        last_row["asset_class"] = (last_row.get("asset_class", "") + " " + cont).strip()

        if not data_rows:
            continue

        page_df = pd.DataFrame(data_rows, columns=cols)
        out.append(page_df)

    return out

