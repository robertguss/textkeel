---
name: textkeel
description: Zero-Python, zero-PII local Rust TF-IDF + L2 Multinomial Logistic text classifier with phrase attribution & self-contained offline browser playground export.
---

# `textkeel` — Local Zero-PII TF-IDF + Calibrated Linear Text Classifier

## When to Use
- Train an explainable, FERPA/PII-safe text classifier locally on advising notes, support tickets, exit surveys, or feedback logs without sending text to external LLM APIs or installing Python/scikit-learn.
- Classify single strings (`--text-str`) or batch files (`CSV`, `TSV`, `JSONL`, `Parquet`) in `<1ms` per document with exact top contributing words/bigrams (`--top-phrases`).
- Export any trained `model.tk.json` into a zero-CDN, single-file offline HTML live classifier playground (`textkeel export-browser`) that runs real inference locally in any browser.

## Command Matrix

| Task | Command |
| --- | --- |
| Train word (1,2)-gram TF-IDF + Logistic model | `textkeel train notes.csv --text note_text --label category --out model.tk.json --format agent` |
| Classify a single text locally with phrase drivers | `textkeel classify model.tk.json --text-str "tuition balance hold delayed registration" --format agent` |
| Batch classify a CSV / JSONL / Parquet file | `textkeel classify model.tk.json inbox.parquet --text body --out tagged.jsonl` |
| Export offline zero-CDN browser playground | `textkeel export-browser model.tk.json --out playground.html --title "Advising Note Classifier"` |
| Print / install this skill | `textkeel skill` / `textkeel skill --install` |

## Exit Codes
- `0`: Success (and optional `--min-macro-f1` gate passed).
- `1`: Quality gate (`--min-macro-f1`) failed.
- `2`: I/O, column, or model parse error.
