//! Self-contained zero-CDN offline HTML interactive browser classifier playground compiler.

use crate::nlp::TextKeelModel;

fn esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

pub fn render_browser_playground(model: &TextKeelModel, title: &str) -> String {
    // Fully escape `<`, `>`, `&`, U+2028, U+2029 so embedded JSON can never break out
    // of `<script>` (including HTML5 `<!-- <script>` double-escaped state).
    let model_json = serde_json::to_string(model)
        .unwrap_or_else(|_| "{}".to_string())
        .replace('<', "\\u003c")
        .replace('>', "\\u003e")
        .replace('&', "\\u0026")
        .replace('\u{2028}', "\\u2028")
        .replace('\u{2029}', "\\u2029");

    let class_rows: String = model
        .eval
        .per_class
        .iter()
        .map(|cm| {
            let phrases = model
                .top_phrases_per_class
                .get(&cm.label)
                .map(|ps| {
                    ps.iter()
                        .take(5)
                        .map(|p| format!("<span class=\"tag\">{}</span>", esc(&p.phrase)))
                        .collect::<Vec<_>>()
                        .join(" ")
                })
                .unwrap_or_default();
            format!(
                "<tr><td><strong>{}</strong></td><td class=\"num\">{}</td><td class=\"num\">{:.1}%</td><td class=\"num\">{:.1}%</td><td class=\"num\">{:.1}%</td><td>{}</td></tr>",
                esc(&cm.label),
                cm.support,
                cm.precision * 100.0,
                cm.recall * 100.0,
                cm.f1 * 100.0,
                phrases
            )
        })
        .collect();

    format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{}</title>
<style>
  :root {{ --bg:#0b1120; --card:#111a2e; --border:#1f2d47; --fg:#e2e8f0; --muted:#94a3b8; --accent:#38bdf8; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; padding:28px; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; background:var(--bg); color:var(--fg); line-height:1.5; }}
  .wrap {{ max-width:980px; margin:0 auto; }}
  h1 {{ margin:0 0 6px; font-size:22px; }}
  h2 {{ margin:0 0 12px; font-size:15px; }}
  .sub {{ color:var(--muted); font-size:13px; margin:0 0 18px; }}
  .badge {{ display:inline-block; padding:3px 9px; border-radius:999px; font-size:11px; font-weight:700; background:rgba(16,185,129,.16); color:#34d399; border:1px solid rgba(16,185,129,.35); }}
  .card {{ background:var(--card); border:1px solid var(--border); border-radius:12px; padding:18px 20px; margin-bottom:16px; }}
  textarea {{ width:100%; min-height:96px; background:#090f1d; color:var(--fg); border:1px solid #334155; border-radius:8px; padding:12px; font-size:14px; font-family:inherit; resize:vertical; }}
  textarea:focus {{ outline:none; border-color:var(--accent); }}
  .bar-row {{ display:grid; grid-template-columns:180px 1fr 62px; align-items:center; gap:10px; margin-bottom:8px; font-size:13px; }}
  .bar-track {{ height:10px; background:#1e293b; border-radius:999px; overflow:hidden; }}
  .bar-fill {{ height:100%; border-radius:999px; background:#38bdf8; transition:width .12s ease; }}
  .bar-val {{ text-align:right; font-family:ui-monospace,monospace; font-size:12px; }}
  .tag {{ display:inline-block; padding:2px 8px; margin:2px; border-radius:6px; font-size:12px; background:#1e293b; border:1px solid #334155; color:#bae6fd; font-family:ui-monospace,monospace; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  th, td {{ padding:8px 10px; border-bottom:1px solid var(--border); text-align:left; }}
  th {{ color:var(--muted); font-size:11px; text-transform:uppercase; }}
  .num {{ text-align:right; font-family:ui-monospace,monospace; }}
  .foot {{ font-size:11px; color:var(--muted); font-family:ui-monospace,monospace; margin-top:12px; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>{} <span class="badge">100% LOCAL · ZERO PII NETWORK CALLS</span></h1>
  <p class="sub">Compiled by <strong>textkeel v{}</strong> · {} classes · {} TF-IDF features · Held-out Macro-F1: <strong>{:.1}%</strong> · Accuracy: <strong>{:.1}%</strong></p>

  <div class="card">
    <h2>Live Offline Text Classifier (&lt;1ms In-Browser Inference)</h2>
    <textarea id="inputText" placeholder="Paste any note, ticket, or response here to classify locally in your browser..."></textarea>
    <div id="result" style="margin-top:14px"></div>
  </div>

  <div class="card">
    <h2>Held-Out Evaluation &amp; Top Discriminative Phrases</h2>
    <table>
      <thead><tr><th>Class</th><th class="num">Support</th><th class="num">Prec</th><th class="num">Recall</th><th class="num">F1</th><th>Top Learned Phrases</th></tr></thead>
      <tbody>{class_rows}</tbody>
    </table>
  </div>

  <div class="foot">textkeel v{} · model sha256: {}</div>
</div>

<script>
const MODEL = {model_json};
const STOP = new Set(["the","and","for","that","with","this","from","have","been","were","will","would","could","should","about","into","your","their","they","them","some","such","than","then","what","when","where","which","while","just","also","very","much","more","most","only","other","over","there","here","these","those","being","doing","are","was","has","had","not","but","can","all","our","out","she","him","her","his","you","its","who","did","does","per"]);
const VOCAB_IDX = new Map(MODEL.vocab.map((t, i) => [t, i]));

function extractNgrams(text, maxN) {{
  const words = (text.toLowerCase().match(/[\p{{L}}\p{{N}}']+/gu) || [])
    .map(w => w.replace(/^'+|'+$/g, ''))
    .filter(w => [...w].length >= 2 && !STOP.has(w));
  const out = [...words];
  for (let n = 2; n <= Math.min(maxN, 3); n++) {{
    for (let i = 0; i + n <= words.length; i++) {{
      out.push(words.slice(i, i + n).join(' '));
    }}
  }}
  return out;
}}

function classify(text) {{
  const t0 = performance.now();
  const terms = extractNgrams(text, MODEL.ngram_max ?? 2);
  const tf = new Map();
  for (const t of terms) {{
    const idx = VOCAB_IDX.get(t);
    if (idx !== undefined) tf.set(idx, (tf.get(idx) || 0) + 1);
  }}
  const vec = [];
  let normSq = 0;
  for (const [idx, cnt] of tf.entries()) {{
    const v = (1 + Math.log(cnt)) * MODEL.idf[idx];
    vec.push([idx, v]);
    normSq += v * v;
  }}
  const norm = Math.sqrt(normSq);
  if (norm > 1e-12) {{
    for (const item of vec) item[1] /= norm;
  }}

  const logits = MODEL.bias.slice();
  for (const [idx, val] of vec) {{
    for (let c = 0; c < MODEL.classes.length; c++) {{
      logits[c] += MODEL.weights[c][idx] * val;
    }}
  }}
  const maxL = Math.max(...logits);
  const exps = logits.map(z => Math.exp(z - maxL));
  const sumE = Math.max(exps.reduce((a, b) => a + b, 0), 1e-12);
  const probs = exps.map(e => e / sumE);
  let best = 0;
  for (let c = 1; c < probs.length; c++) if (probs[c] > probs[best]) best = c;

  const drivers = vec
    .map(([idx, val]) => ({{ phrase: MODEL.vocab[idx], contrib: MODEL.weights[best][idx] * val }}))
    .filter(d => d.contrib > 1e-4)
    .sort((a, b) => b.contrib - a.contrib)
    .slice(0, 6);

  return {{ label: MODEL.classes[best], confidence: probs[best], probs, drivers, ms: performance.now() - t0 }};
}}

const esc = s => String(s)
  .replace(/&/g, '&amp;')
  .replace(/</g, '&lt;')
  .replace(/>/g, '&gt;')
  .replace(/"/g, '&quot;')
  .replace(/'/g, '&#39;');
const ta = document.getElementById('inputText');
const resEl = document.getElementById('result');
function render() {{
  const txt = ta.value.trim();
  if (!txt) {{ resEl.innerHTML = '<div class="sub">Start typing to see instant class probabilities and phrase attributions.</div>'; return; }}
  const r = classify(txt);
  const order = MODEL.classes.map((c, i) => [c, r.probs[i]]).sort((a, b) => b[1] - a[1]);
  const bars = order.map(([c, p]) =>
    `<div class="bar-row"><div>${{esc(c)}}</div><div class="bar-track"><div class="bar-fill" style="width:${{(p*100).toFixed(1)}}%"></div></div><div class="bar-val">${{(p*100).toFixed(1)}}%</div></div>`
  ).join('');
  const drv = r.drivers.map(d => `<span class="tag">${{esc(d.phrase)}} (+${{d.contrib.toFixed(2)}})</span>`).join(' ');
  resEl.innerHTML = `<div style="margin-bottom:10px">Predicted: <strong>${{esc(r.label)}}</strong> (${{(r.confidence*100).toFixed(1)}}% confidence · ${{r.ms.toFixed(2)}} ms)</div>${{bars}}<div style="margin-top:8px"><span class="sub">Key phrase drivers:</span> ${{drv || '<span class="sub">none</span>'}}</div>`;
}}
ta.addEventListener('input', render);
render();
</script>
</body>
</html>"#,
        esc(title),
        esc(title),
        esc(&model.tool_version),
        model.classes.len(),
        model.vocab.len(),
        model.eval.macro_f1 * 100.0,
        model.eval.accuracy * 100.0,
        esc(&model.tool_version),
        esc(&model.sha256),
    )
}
