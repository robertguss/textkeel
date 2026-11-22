# `textkeel`

**Part of the [Keel Data Stack](https://github.com/robertguss/keel-stack) (Tool #17).**

Zero-Python, zero-PII, single-binary Rust **TF-IDF (1,2)-gram + L2 Multinomial Logistic text classifier** with exact phrase attribution and self-contained offline HTML browser playground export (DX + AX).

## Why `textkeel`?
When working with student advising notes, FERPA-protected exit surveys, or sensitive support tickets, sending raw text to external cloud LLM APIs is a privacy non-starter—and shipping scikit-learn/PyTorch containers just to route short text is overkill. `textkeel` trains and scores fast, auditable, calibrated linear text models 100% locally in milliseconds, explains every prediction with its top contributing phrases, and compiles any trained model into a single offline HTML playground.

## Features
- **Pure-Rust Training (`textkeel train`):** Word unigram + bigram tokenizer, sublinear TF-IDF (`1 + ln(tf)`) with L2 row normalization, stratified train/test split, L2-regularized Multinomial Softmax Logistic Regression, macro-F1 / accuracy / per-class precision-recall-F1 metrics, and top discriminative phrases per class.
- **Sub-Millisecond Inference (`textkeel classify`):** Score a single string (`--text-str`) or batch `CSV`, `TSV`, `JSONL`, `Parquet` files with exact phrase-level linear attributions (`tfidf(term) × weight`).
- **Offline Browser Playground Compiler (`textkeel export-browser`):** Embeds the real trained vocabulary, IDF weights, and class weight matrix inside a zero-CDN interactive HTML file for `<1ms` completely local in-browser classification.

## Quick Start

```bash
# Train a local classifier & verify held-out macro-F1
textkeel train advising_notes.csv --text note_text --label category \
  --out notes.tk.json --min-macro-f1 0.80

# Classify a single note with exact phrase attributions
textkeel classify notes.tk.json --text-str "fell behind on Greek quiz due to pastor travel"

# Compile into a zero-CDN offline live browser playground
textkeel export-browser notes.tk.json --out notes_playground.html
```

## Install

```bash
cargo install --git https://github.com/robertguss/textkeel --locked
```

## License
Dual-licensed under MIT or Apache-2.0.
