use std::io::IsTerminal;
use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum OutputFormat {
    Auto,
    Human,
    Json,
    Agent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolvedFormat {
    Human,
    Json,
    Agent,
}

impl OutputFormat {
    pub fn resolve(self) -> ResolvedFormat {
        match self {
            Self::Human => ResolvedFormat::Human,
            Self::Json => ResolvedFormat::Json,
            Self::Agent => ResolvedFormat::Agent,
            Self::Auto => {
                if std::io::stdout().is_terminal() {
                    ResolvedFormat::Human
                } else {
                    ResolvedFormat::Json
                }
            }
        }
    }
}

#[derive(Debug, Parser)]
#[command(
    name = "textkeel",
    version,
    about = "Zero-Python, zero-PII local Rust TF-IDF + Multinomial Logistic text classifier with phrase attribution & offline HTML playground export."
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Train a local word (1,2)-gram TF-IDF + L2 Multinomial Logistic text classifier
    Train {
        /// Labeled dataset (.csv, .tsv, .jsonl, .ndjson, .parquet)
        input: PathBuf,
        /// Text column name
        #[arg(long)]
        text: String,
        /// Class label column name
        #[arg(long)]
        label: String,
        /// Output portable model JSON path (e.g. model.tk.json)
        #[arg(short, long)]
        out: PathBuf,
        /// Maximum n-gram length (1 = unigrams, 2 = unigrams + bigrams)
        #[arg(long, default_value_t = 2)]
        ngrams: usize,
        /// Maximum vocabulary size
        #[arg(long, default_value_t = 2500)]
        max_features: usize,
        /// Minimum document frequency to include a term
        #[arg(long, default_value_t = 1)]
        min_df: usize,
        /// Held-out test fraction for stratified evaluation (0.0 to 0.5)
        #[arg(long, default_value_t = 0.2)]
        test_split: f64,
        /// Training epochs
        #[arg(long, default_value_t = 160)]
        epochs: usize,
        /// L2 regularization strength
        #[arg(long, default_value_t = 1e-3)]
        l2: f64,
        /// Deterministic RNG seed
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// Optional CI gate: minimum held-out macro-F1 (exits 1 if below)
        #[arg(long)]
        min_macro_f1: Option<f64>,
        /// Optional self-contained offline browser playground HTML path
        #[arg(long)]
        export_browser: Option<PathBuf>,
        /// Output format (auto, human, json, agent)
        #[arg(long, value_enum, default_value_t = OutputFormat::Auto)]
        format: OutputFormat,
    },

    /// Classify a single string (--text-str) or batch file locally with phrase attributions
    Classify {
        /// Trained textkeel model (.tk.json)
        model: PathBuf,
        /// Optional batch dataset (.csv, .tsv, .jsonl, .ndjson, .parquet)
        input: Option<PathBuf>,
        /// Text column name when classifying a batch file
        #[arg(long)]
        text: Option<String>,
        /// Single text string to classify directly from the CLI
        #[arg(long)]
        text_str: Option<String>,
        /// Optional ID column to preserve in batch output
        #[arg(long)]
        id_col: Option<String>,
        /// Number of top contributing phrases to return per document
        #[arg(long, default_value_t = 4)]
        top_phrases: usize,
        /// Write batch predictions to JSONL or CSV
        #[arg(short, long)]
        out: Option<PathBuf>,
        /// Output format (auto, human, json, agent)
        #[arg(long, value_enum, default_value_t = OutputFormat::Auto)]
        format: OutputFormat,
    },

    /// Compile a trained model (.tk.json) into a zero-CDN interactive offline HTML browser playground
    ExportBrowser {
        /// Trained textkeel model (.tk.json)
        model: PathBuf,
        /// Output HTML path
        #[arg(short, long)]
        out: PathBuf,
        /// Optional page title
        #[arg(long)]
        title: Option<String>,
    },

    /// Print or install the bundled SKILL.md for AI coding agents
    Skill {
        /// Install SKILL.md into <dir>/textkeel/SKILL.md (defaults to .agents/skills)
        #[arg(long, num_args = 0..=1, default_missing_value = ".agents/skills")]
        install: Option<PathBuf>,
        /// Output format (kept for Keel stack CLI consistency)
        #[arg(long, value_enum, default_value_t = OutputFormat::Auto)]
        format: OutputFormat,
    },
}
