//! Pure-Rust TF-IDF (word 1..n-gram, sublinear TF, smooth IDF, L2 norm)
//! + L2-regularized Multinomial Softmax Logistic Regression (Adam optimizer)
//! + exact per-phrase linear attribution.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::Path;

use chrono::Utc;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const STOP_WORDS: &[&str] = &[
    "the", "and", "for", "that", "with", "this", "from", "have", "been", "were", "will", "would",
    "could", "should", "about", "into", "your", "their", "they", "them", "some", "such", "than",
    "then", "what", "when", "where", "which", "while", "just", "also", "very", "much", "more",
    "most", "only", "other", "over", "there", "here", "these", "those", "being", "doing", "are",
    "was", "has", "had", "not", "but", "can", "all", "our", "out", "she", "him", "her", "his",
    "you", "its", "who", "did", "does", "per",
];

fn r4(v: f64) -> f64 {
    (v * 10_000.0).round() / 10_000.0
}

fn r6(v: f64) -> f64 {
    (v * 1_000_000.0).round() / 1_000_000.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassMetric {
    pub label: String,
    pub support: usize,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalMetrics {
    pub n_train: usize,
    pub n_test: usize,
    #[serde(default)]
    pub is_in_sample: bool,
    pub accuracy: f64,
    pub macro_f1: f64,
    pub weighted_f1: f64,
    pub per_class: Vec<ClassMetric>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhraseWeight {
    pub phrase: String,
    pub weight: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextKeelModel {
    pub schema_version: u32,
    pub tool: String,
    pub tool_version: String,
    pub created_at: String,
    pub ngram_max: usize,
    pub vocab: Vec<String>,
    pub idf: Vec<f64>,
    pub classes: Vec<String>,
    /// Flattened row-major matrix: `classes.len() * vocab.len()`
    pub weights: Vec<Vec<f64>>,
    pub bias: Vec<f64>,
    pub top_phrases_per_class: IndexMap<String, Vec<PhraseWeight>>,
    pub eval: EvalMetrics,
    pub sha256: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct PhraseDriver {
    pub phrase: String,
    pub contribution: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct Prediction {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub predicted_label: String,
    pub confidence: f64,
    pub probabilities: IndexMap<String, f64>,
    pub top_phrases: Vec<PhraseDriver>,
}

/// Extract normalized word unigrams and n-grams up to `ngram_max`.
pub fn extract_ngrams(text: &str, ngram_max: usize) -> Vec<String> {
    let stop: HashSet<&str> = STOP_WORDS.iter().copied().collect();
    let mut words: Vec<String> = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        if ch.is_alphanumeric() || ch == '\'' {
            for lc in ch.to_lowercase() {
                cur.push(lc);
            }
        } else if !cur.is_empty() {
            let trimmed = cur.trim_matches('\'');
            if trimmed.chars().count() >= 2 && !stop.contains(trimmed) {
                words.push(trimmed.to_string());
            }
            cur.clear();
        }
    }
    if !cur.is_empty() {
        let trimmed = cur.trim_matches('\'');
        if trimmed.chars().count() >= 2 && !stop.contains(trimmed) {
            words.push(trimmed.to_string());
        }
    }

    let mut out = words.clone();
    if ngram_max >= 2 && words.len() >= 2 {
        for n in 2..=ngram_max.min(3) {
            if words.len() >= n {
                for win in words.windows(n) {
                    out.push(win.join(" "));
                }
            }
        }
    }
    out
}

fn vectorize_doc(
    text: &str,
    vocab_map: &HashMap<&str, usize>,
    idf: &[f64],
    ngram_max: usize,
) -> Vec<(usize, f64)> {
    let terms = extract_ngrams(text, ngram_max);
    let mut tf: HashMap<usize, usize> = HashMap::new();
    for t in terms {
        if let Some(&idx) = vocab_map.get(t.as_str()) {
            *tf.entry(idx).or_insert(0) += 1;
        }
    }
    if tf.is_empty() {
        return Vec::new();
    }
    let mut vec: Vec<(usize, f64)> = tf
        .into_iter()
        .map(|(idx, cnt)| {
            let sublinear_tf = 1.0 + (cnt as f64).ln();
            (idx, sublinear_tf * idf[idx])
        })
        .collect();
    let norm = vec.iter().map(|&(_, v)| v * v).sum::<f64>().sqrt();
    if norm > 1e-12 {
        for (_, v) in &mut vec {
            *v /= norm;
        }
    }
    vec.sort_by_key(|&(idx, _)| idx);
    vec
}

fn softmax(logits: &[f64]) -> Vec<f64> {
    let max_l = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&z| (z - max_l).exp()).collect();
    let sum: f64 = exps.iter().sum::<f64>().max(1e-12);
    exps.into_iter().map(|e| e / sum).collect()
}

/// Deterministic xorshift64* shuffle for stratified train/test split.
fn det_shuffle<T>(items: &mut [T], seed: u64) {
    let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    for i in (1..items.len()).rev() {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        let r = state.wrapping_mul(0x2545_F491_4F6C_DD1D);
        let j = (r as usize) % (i + 1);
        items.swap(i, j);
    }
}

#[allow(clippy::too_many_arguments)]
pub fn train_model(
    docs: &[(String, String)],
    ngram_max: usize,
    max_features: usize,
    min_df: usize,
    test_split: f64,
    epochs: usize,
    l2: f64,
    seed: u64,
) -> Result<TextKeelModel, String> {
    if docs.len() < 4 {
        return Err(format!(
            "need at least 4 labeled documents to train, got {}",
            docs.len()
        ));
    }
    if ngram_max == 0 {
        return Err("--ngrams must be >= 1".to_string());
    }
    if !l2.is_finite() || l2 < 0.0 {
        return Err(format!("--l2 must be finite and >= 0.0, got {l2}"));
    }
    if !test_split.is_finite() || !(0.0..=0.5).contains(&test_split) {
        return Err(format!(
            "--test-split must be finite and within [0.0, 0.5], got {test_split}"
        ));
    }

    // Collect sorted class names
    let mut class_set: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (idx, (_, lbl)) in docs.iter().enumerate() {
        class_set.entry(lbl.clone()).or_default().push(idx);
    }
    if class_set.len() < 2 {
        return Err("need at least 2 distinct label classes to train".to_string());
    }
    let classes: Vec<String> = class_set.keys().cloned().collect();
    let class_to_idx: HashMap<&str, usize> = classes
        .iter()
        .enumerate()
        .map(|(i, c)| (c.as_str(), i))
        .collect();

    // Stratified deterministic train / test split
    let split = test_split.clamp(0.0, 0.5);
    let mut train_idx = Vec::new();
    let mut test_idx = Vec::new();
    for (c_i, (_, mut indices)) in class_set.into_iter().enumerate() {
        det_shuffle(&mut indices, seed.wrapping_add(c_i as u64 * 97));
        let n_test = if split > 0.0 && indices.len() >= 2 {
            ((indices.len() as f64) * split).round().max(1.0) as usize
        } else {
            0
        };
        let n_test = n_test.min(indices.len().saturating_sub(1));
        test_idx.extend_from_slice(&indices[..n_test]);
        train_idx.extend_from_slice(&indices[n_test..]);
    }
    let mut is_in_sample = false;
    if test_idx.is_empty() {
        // Fallback: evaluate on train set if split == 0.0 (explicitly flagged in eval.is_in_sample)
        test_idx = train_idx.clone();
        is_in_sample = true;
    }

    // Build TF-IDF vocabulary strictly from training documents (zero leakage!)
    let mut df_counts: HashMap<String, usize> = HashMap::new();
    for &i in &train_idx {
        let uniq: HashSet<String> = extract_ngrams(&docs[i].0, ngram_max).into_iter().collect();
        for term in uniq {
            *df_counts.entry(term).or_insert(0) += 1;
        }
    }
    let eff_min_df = min_df.max(1);
    let mut vocab_entries: Vec<(String, usize)> = df_counts
        .into_iter()
        .filter(|(_, df)| *df >= eff_min_df)
        .collect();
    if vocab_entries.is_empty() {
        return Err("vocabulary is empty; try lowering --min-df".to_string());
    }
    vocab_entries.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    vocab_entries.truncate(max_features.max(10));

    let n_train_f = train_idx.len() as f64;
    let vocab: Vec<String> = vocab_entries.iter().map(|(t, _)| t.clone()).collect();
    let idf: Vec<f64> = vocab_entries
        .iter()
        .map(|(_, df)| r6(((1.0 + n_train_f) / (1.0 + (*df as f64))).ln() + 1.0))
        .collect();
    let vocab_map: HashMap<&str, usize> = vocab
        .iter()
        .enumerate()
        .map(|(i, t)| (t.as_str(), i))
        .collect();

    // Vectorize train and test splits
    let train_vecs: Vec<(Vec<(usize, f64)>, usize)> = train_idx
        .iter()
        .map(|&i| {
            let v = vectorize_doc(&docs[i].0, &vocab_map, &idf, ngram_max);
            let y = class_to_idx[docs[i].1.as_str()];
            (v, y)
        })
        .collect();
    let test_vecs: Vec<(Vec<(usize, f64)>, usize)> = test_idx
        .iter()
        .map(|&i| {
            let v = vectorize_doc(&docs[i].0, &vocab_map, &idf, ngram_max);
            let y = class_to_idx[docs[i].1.as_str()];
            (v, y)
        })
        .collect();

    // Balanced class weights so minority classes have equal voice
    let k = classes.len();
    let v_len = vocab.len();
    let mut class_counts = vec![0usize; k];
    for &(_, y) in &train_vecs {
        class_counts[y] += 1;
    }
    let class_weights: Vec<f64> = class_counts
        .iter()
        .map(|&cnt| n_train_f / ((k as f64) * (cnt.max(1) as f64)))
        .collect();

    // Full-batch Adam optimization for L2-regularized Multinomial Logistic Regression
    let mut w = vec![vec![0.0_f64; v_len]; k];
    let mut b = vec![0.0_f64; k];
    let mut mw = vec![vec![0.0_f64; v_len]; k];
    let mut vw = vec![vec![0.0_f64; v_len]; k];
    let mut mb = vec![0.0_f64; k];
    let mut vb = vec![0.0_f64; k];

    let lr = 0.25_f64;
    let beta1 = 0.9_f64;
    let beta2 = 0.999_f64;
    let eps = 1e-8_f64;
    let n_ep = epochs.max(20);

    for step in 1..=n_ep {
        let mut gw = vec![vec![0.0_f64; v_len]; k];
        let mut gb = vec![0.0_f64; k];

        for (x_sparse, y) in &train_vecs {
            let mut logits = b.clone();
            for &(fi, xv) in x_sparse {
                for c in 0..k {
                    logits[c] += w[c][fi] * xv;
                }
            }
            let probs = softmax(&logits);
            let sw = class_weights[*y];
            for c in 0..k {
                let err = (probs[c] - if c == *y { 1.0 } else { 0.0 }) * sw;
                gb[c] += err;
                for &(fi, xv) in x_sparse {
                    gw[c][fi] += err * xv;
                }
            }
        }

        let inv_n = 1.0 / n_train_f;
        let b1_corr = 1.0 - beta1.powi(step as i32);
        let b2_corr = 1.0 - beta2.powi(step as i32);
        let lr_t = lr
            * (0.5 * (1.0 + (std::f64::consts::PI * (step as f64) / (n_ep as f64)).cos()))
                .max(0.05);

        for c in 0..k {
            let g_b = gb[c] * inv_n;
            mb[c] = beta1 * mb[c] + (1.0 - beta1) * g_b;
            vb[c] = beta2 * vb[c] + (1.0 - beta2) * g_b * g_b;
            b[c] -= lr_t * (mb[c] / b1_corr) / ((vb[c] / b2_corr).sqrt() + eps);

            for fi in 0..v_len {
                let g = gw[c][fi] * inv_n + l2 * w[c][fi];
                mw[c][fi] = beta1 * mw[c][fi] + (1.0 - beta1) * g;
                vw[c][fi] = beta2 * vw[c][fi] + (1.0 - beta2) * g * g;
                w[c][fi] -= lr_t * (mw[c][fi] / b1_corr) / ((vw[c][fi] / b2_corr).sqrt() + eps);
            }
        }
    }

    // Round weights cleanly for compact portable JSON
    for row in &mut w {
        for val in row {
            *val = r6(*val);
        }
    }
    for val in &mut b {
        *val = r6(*val);
    }

    // Evaluate on held-out test set
    let mut tp = vec![0usize; k];
    let mut fp = vec![0usize; k];
    let mut fn_cnt = vec![0usize; k];
    let mut support = vec![0usize; k];
    let mut correct = 0usize;

    for (x_sparse, y) in &test_vecs {
        support[*y] += 1;
        let mut logits = b.clone();
        for &(fi, xv) in x_sparse {
            for c in 0..k {
                logits[c] += w[c][fi] * xv;
            }
        }
        let pred = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0)))
            .map(|(i, _)| i)
            .unwrap_or(0);
        if pred == *y {
            correct += 1;
            tp[*y] += 1;
        } else {
            fp[pred] += 1;
            fn_cnt[*y] += 1;
        }
    }

    let mut per_class = Vec::with_capacity(k);
    let mut macro_f1_sum = 0.0_f64;
    let mut weighted_f1_sum = 0.0_f64;
    let mut active_classes = 0usize;
    let n_test_f = test_vecs.len().max(1) as f64;

    for c in 0..k {
        let p = if tp[c] + fp[c] > 0 {
            (tp[c] as f64) / ((tp[c] + fp[c]) as f64)
        } else {
            0.0
        };
        let r = if tp[c] + fn_cnt[c] > 0 {
            (tp[c] as f64) / ((tp[c] + fn_cnt[c]) as f64)
        } else {
            0.0
        };
        let f1 = if p + r > 0.0 {
            2.0 * p * r / (p + r)
        } else {
            0.0
        };
        if support[c] > 0 || fp[c] > 0 {
            macro_f1_sum += f1;
            active_classes += 1;
        }
        if support[c] > 0 {
            weighted_f1_sum += f1 * (support[c] as f64);
        }
        per_class.push(ClassMetric {
            label: classes[c].clone(),
            support: support[c],
            precision: r4(p),
            recall: r4(r),
            f1: r4(f1),
        });
    }

    let eval = EvalMetrics {
        n_train: train_vecs.len(),
        n_test: test_vecs.len(),
        is_in_sample,
        accuracy: r4((correct as f64) / n_test_f),
        macro_f1: r4(macro_f1_sum / (active_classes.max(1) as f64)),
        weighted_f1: r4(weighted_f1_sum / n_test_f),
        per_class,
    };

    // Extract top discriminative phrases per class
    let mut top_phrases_per_class = IndexMap::new();
    for (c, cname) in classes.iter().enumerate() {
        let mut idxs: Vec<usize> = (0..v_len).collect();
        idxs.sort_by(|&a, &b| w[c][b].total_cmp(&w[c][a]));
        let top: Vec<PhraseWeight> = idxs
            .into_iter()
            .take(10)
            .filter(|&i| w[c][i] > 0.0)
            .map(|i| PhraseWeight {
                phrase: vocab[i].clone(),
                weight: r4(w[c][i]),
            })
            .collect();
        top_phrases_per_class.insert(cname.clone(), top);
    }

    let mut hasher = Sha256::new();
    hasher.update(
        format!(
            "{}:{}:{}:{:.4}",
            classes.join(","),
            vocab.len(),
            n_train_f,
            eval.macro_f1
        )
        .as_bytes(),
    );
    let sha256: String = hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();

    Ok(TextKeelModel {
        schema_version: 1,
        tool: "textkeel".to_string(),
        tool_version: env!("CARGO_PKG_VERSION").to_string(),
        created_at: Utc::now().to_rfc3339(),
        ngram_max,
        vocab,
        idf,
        classes,
        weights: w,
        bias: b,
        top_phrases_per_class,
        eval,
        sha256,
    })
}

impl TextKeelModel {
    pub fn from_file(path: &Path) -> Result<Self, String> {
        let raw = fs::read_to_string(path)
            .map_err(|e| format!("failed to read model {}: {e}", path.display()))?;
        let model: Self = serde_json::from_str(&raw)
            .map_err(|e| format!("failed parsing textkeel model: {e}"))?;
        model.validate()?;
        Ok(model)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.classes.is_empty() {
            return Err("invalid textkeel model: `classes` must not be empty".to_string());
        }
        if self.ngram_max == 0 {
            return Err("invalid textkeel model: `ngram_max` must be >= 1".to_string());
        }
        if self.idf.len() != self.vocab.len() {
            return Err(format!(
                "invalid textkeel model: idf.len() ({}) != vocab.len() ({})",
                self.idf.len(),
                self.vocab.len()
            ));
        }
        if self.idf.iter().any(|v| !v.is_finite()) {
            return Err("invalid textkeel model: non-finite values in `idf`".to_string());
        }
        if self.bias.len() != self.classes.len() {
            return Err(format!(
                "invalid textkeel model: bias.len() ({}) != classes.len() ({})",
                self.bias.len(),
                self.classes.len()
            ));
        }
        if self.bias.iter().any(|v| !v.is_finite()) {
            return Err("invalid textkeel model: non-finite values in `bias`".to_string());
        }
        if self.weights.len() != self.classes.len() {
            return Err(format!(
                "invalid textkeel model: weights.len() ({}) != classes.len() ({})",
                self.weights.len(),
                self.classes.len()
            ));
        }
        for (c, row) in self.weights.iter().enumerate() {
            if row.len() != self.vocab.len() {
                return Err(format!(
                    "invalid textkeel model: weights[{c}].len() ({}) != vocab.len() ({})",
                    row.len(),
                    self.vocab.len()
                ));
            }
            if row.iter().any(|v| !v.is_finite()) {
                return Err(format!(
                    "invalid textkeel model: non-finite value in weights[{c}]"
                ));
            }
        }
        Ok(())
    }

    pub fn predict(&self, text: &str, id: Option<String>, top_phrases: usize) -> Prediction {
        let vocab_map: HashMap<&str, usize> = self
            .vocab
            .iter()
            .enumerate()
            .map(|(i, t)| (t.as_str(), i))
            .collect();
        let x_sparse = vectorize_doc(text, &vocab_map, &self.idf, self.ngram_max);
        let k = self.classes.len();
        let mut logits = self.bias.clone();
        for &(fi, xv) in &x_sparse {
            for (c, logit) in logits.iter_mut().enumerate().take(k) {
                *logit += self.weights[c][fi] * xv;
            }
        }
        let probs = softmax(&logits);
        // First-index wins on ties (matches browser JS `if (probs[c] > probs[best])`)
        let pred_idx = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0)))
            .map(|(i, _)| i)
            .unwrap_or(0);

        let mut prob_map = IndexMap::new();
        for (c, cname) in self.classes.iter().enumerate() {
            prob_map.insert(cname.clone(), r6(probs[c]));
        }

        let mut drivers: Vec<PhraseDriver> = x_sparse
            .iter()
            .map(|&(fi, xv)| PhraseDriver {
                phrase: self.vocab[fi].clone(),
                contribution: r4(self.weights[pred_idx][fi] * xv),
            })
            .filter(|d| d.contribution > 1e-4)
            .collect();
        drivers.sort_by(|a, b| b.contribution.total_cmp(&a.contribution));
        drivers.truncate(top_phrases);

        Prediction {
            id,
            predicted_label: self.classes[pred_idx].clone(),
            confidence: r4(probs[pred_idx]),
            probabilities: prob_map,
            top_phrases: drivers,
        }
    }
}
