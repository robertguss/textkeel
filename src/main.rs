//! `textkeel` — Zero-Python, zero-PII local Rust TF-IDF + Multinomial Logistic text classifier.

mod cli;
mod html;
mod io_data;
mod nlp;
mod skill;

use std::fs;
use std::process::ExitCode;
use std::time::Instant;

use clap::Parser;
use cli::{Cli, Command, ResolvedFormat};
use io_data::read_table;
use nlp::{Prediction, TextKeelModel, train_model};
use owo_colors::OwoColorize;

fn main() -> ExitCode {
    let cli = Cli::parse();
    match dispatch(cli) {
        Ok(code) => ExitCode::from(code),
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::from(2)
        }
    }
}

fn dispatch(cli: Cli) -> Result<u8, String> {
    match cli.command {
        Command::Skill { install, format: _ } => {
            skill::handle_skill(install.as_deref())?;
            Ok(0)
        }
        Command::Train {
            input,
            text,
            label,
            out,
            ngrams,
            max_features,
            min_df,
            test_split,
            epochs,
            l2,
            seed,
            min_macro_f1,
            export_browser,
            format,
        } => {
            let t0 = Instant::now();
            let table = read_table(&input)?;
            let text_col = table
                .columns
                .get(&text)
                .ok_or_else(|| format!("text column `{text}` not found"))?;
            let label_col = table
                .columns
                .get(&label)
                .ok_or_else(|| format!("label column `{label}` not found"))?;

            let mut docs = Vec::new();
            for i in 0..table.row_count {
                if let (Some(txt), Some(lbl)) = (&text_col[i], &label_col[i]) {
                    docs.push((txt.clone(), lbl.clone()));
                }
            }

            if let Some(m) = min_macro_f1 {
                if !m.is_finite() || !(0.0..=1.0).contains(&m) {
                    return Err(format!("--min-macro-f1 must be within [0.0, 1.0], got {m}"));
                }
                if test_split == 0.0 {
                    return Err(
                        "--min-macro-f1 requires a held-out test split (--test-split > 0.0)"
                            .to_string(),
                    );
                }
            }
            let model = train_model(
                &docs,
                ngrams,
                max_features,
                min_df,
                test_split,
                epochs,
                l2,
                seed,
            )?;
            if min_macro_f1.is_some() && model.eval.is_in_sample {
                return Err(
                    "--min-macro-f1 cannot pass on an in-sample fallback split (need >=2 rows in at least one class)"
                        .to_string(),
                );
            }
            if let Some(parent) = out.parent()
                && !parent.as_os_str().is_empty()
            {
                fs::create_dir_all(parent).map_err(|e| format!("failed creating dir: {e}"))?;
            }
            fs::write(
                &out,
                serde_json::to_string_pretty(&model)
                    .map_err(|e| format!("serialize error: {e}"))?,
            )
            .map_err(|e| format!("failed writing {}: {e}", out.display()))?;

            if let Some(bp) = &export_browser {
                let rendered = html::render_browser_playground(
                    &model,
                    "textkeel — Local Text Classifier Playground",
                );
                if let Some(parent) = bp.parent()
                    && !parent.as_os_str().is_empty()
                {
                    fs::create_dir_all(parent).map_err(|e| format!("failed creating dir: {e}"))?;
                }
                fs::write(bp, rendered)
                    .map_err(|e| format!("failed writing {}: {e}", bp.display()))?;
                eprintln!("wrote offline browser playground -> {}", bp.display());
            }

            let elapsed_ms = t0.elapsed().as_millis();
            let gate_ok = min_macro_f1.is_none_or(|m| model.eval.macro_f1 >= m);

            let summary = serde_json::json!({
                "tool": "textkeel",
                "version": env!("CARGO_PKG_VERSION"),
                "action": "train",
                "model_out": out.display().to_string(),
                "elapsed_ms": elapsed_ms,
                "classes": model.classes,
                "vocab_size": model.vocab.len(),
                "ngram_max": model.ngram_max,
                "eval": model.eval,
                "gate_passed": gate_ok,
                "top_phrases_per_class": model.top_phrases_per_class,
                "sha256": model.sha256,
            });

            match format.resolve() {
                ResolvedFormat::Json => {
                    println!("{}", serde_json::to_string_pretty(&summary).unwrap())
                }
                ResolvedFormat::Agent => println!("{}", serde_json::to_string(&summary).unwrap()),
                ResolvedFormat::Human => {
                    println!(
                        "{} train {} ({} docs, {} classes, {} features) in {}ms | macro_f1={:.1}%  acc={:.1}%",
                        "textkeel".bold().green(),
                        input.display(),
                        docs.len(),
                        model.classes.len(),
                        model.vocab.len(),
                        elapsed_ms,
                        model.eval.macro_f1 * 100.0,
                        model.eval.accuracy * 100.0
                    );
                    for cm in &model.eval.per_class {
                        println!(
                            "  {:<22} support={:<4} prec={:.1}%  rec={:.1}%  f1={:.1}%",
                            cm.label,
                            cm.support,
                            cm.precision * 100.0,
                            cm.recall * 100.0,
                            cm.f1 * 100.0
                        );
                    }
                    println!("  saved model -> {}", out.display());
                }
            }
            Ok(if gate_ok { 0 } else { 1 })
        }
        Command::Classify {
            model: model_path,
            input,
            text,
            text_str,
            id_col,
            top_phrases,
            out,
            format,
        } => {
            let t0 = Instant::now();
            let model = TextKeelModel::from_file(&model_path)?;
            let preds: Vec<Prediction> = if let Some(s) = text_str {
                vec![model.predict(&s, None, top_phrases)]
            } else if let Some(inp) = input {
                let col_name = text.ok_or_else(|| {
                    "--text <col> is required when classifying a file".to_string()
                })?;
                let table = read_table(&inp)?;
                let text_col = table
                    .columns
                    .get(&col_name)
                    .ok_or_else(|| format!("text column `{col_name}` not found"))?;
                let id_vec = id_col.as_deref().and_then(|c| table.columns.get(c));
                (0..table.row_count)
                    .map(|i| {
                        let txt = text_col[i].as_deref().unwrap_or("");
                        let id = id_vec.and_then(|v| v[i].clone());
                        model.predict(txt, id, top_phrases)
                    })
                    .collect()
            } else {
                return Err(
                    "provide either --text-str \"...\" or an <input> file with --text <col>"
                        .to_string(),
                );
            };

            let elapsed_us = t0.elapsed().as_micros();

            if let Some(out_path) = &out {
                if let Some(parent) = out_path.parent()
                    && !parent.as_os_str().is_empty()
                {
                    fs::create_dir_all(parent).map_err(|e| format!("failed creating dir: {e}"))?;
                }
                let is_csv = out_path
                    .extension()
                    .and_then(|e| e.to_str())
                    .is_some_and(|e| e.eq_ignore_ascii_case("csv"));
                if is_csv {
                    let mut wtr = csv::Writer::from_path(out_path)
                        .map_err(|e| format!("failed creating CSV {}: {e}", out_path.display()))?;
                    wtr.write_record(["id", "predicted_label", "confidence", "top_phrases"])
                        .map_err(|e| format!("CSV header error: {e}"))?;
                    for (idx, p) in preds.iter().enumerate() {
                        let id_s = p.id.clone().unwrap_or_else(|| format!("{idx}"));
                        let ph = p
                            .top_phrases
                            .iter()
                            .map(|d| format!("{} ({:+.4})", d.phrase, d.contribution))
                            .collect::<Vec<_>>()
                            .join(" | ");
                        wtr.write_record([
                            id_s,
                            p.predicted_label.clone(),
                            format!("{:.4}", p.confidence),
                            ph,
                        ])
                        .map_err(|e| format!("CSV write error: {e}"))?;
                    }
                    wtr.flush().map_err(|e| format!("CSV flush error: {e}"))?;
                } else {
                    let mut buf = String::new();
                    for p in &preds {
                        buf.push_str(&serde_json::to_string(p).unwrap());
                        buf.push('\n');
                    }
                    fs::write(out_path, buf)
                        .map_err(|e| format!("failed writing {}: {e}", out_path.display()))?;
                }
                eprintln!(
                    "wrote {} classifications -> {}",
                    preds.len(),
                    out_path.display()
                );
            }

            let payload = serde_json::json!({
                "tool": "textkeel",
                "version": env!("CARGO_PKG_VERSION"),
                "count": preds.len(),
                "elapsed_us": elapsed_us,
                "predictions": preds,
            });
            match format.resolve() {
                ResolvedFormat::Json => {
                    println!("{}", serde_json::to_string_pretty(&payload).unwrap())
                }
                ResolvedFormat::Agent => println!("{}", serde_json::to_string(&payload).unwrap()),
                ResolvedFormat::Human => {
                    println!(
                        "{} classified {} doc(s) in {:.2}ms",
                        "textkeel".bold().green(),
                        preds.len(),
                        (elapsed_us as f64) / 1000.0
                    );
                    for (idx, p) in preds.iter().take(15).enumerate() {
                        let id_s = p.id.clone().unwrap_or_else(|| format!("#{idx}"));
                        let ph = p
                            .top_phrases
                            .iter()
                            .map(|d| format!("{} ({:+.2})", d.phrase, d.contribution))
                            .collect::<Vec<_>>()
                            .join(", ");
                        println!(
                            "  {:<12} {:<20} ({:.1}%)  [{}]",
                            id_s,
                            p.predicted_label,
                            p.confidence * 100.0,
                            ph
                        );
                    }
                }
            }
            Ok(0)
        }
        Command::ExportBrowser { model, out, title } => {
            let m = TextKeelModel::from_file(&model)?;
            let page_title =
                title.unwrap_or_else(|| "textkeel — Local Zero-PII Text Classifier".to_string());
            let html_str = html::render_browser_playground(&m, &page_title);
            if let Some(parent) = out.parent()
                && !parent.as_os_str().is_empty()
            {
                fs::create_dir_all(parent).map_err(|e| format!("failed creating dir: {e}"))?;
            }
            fs::write(&out, html_str)
                .map_err(|e| format!("failed writing {}: {e}", out.display()))?;
            println!("Exported offline browser playground -> {}", out.display());
            Ok(0)
        }
    }
}
