"""
compliance_ui.py
────────────────
Flask web app — upload a company project, get a full SDAIA / NCA compliance report.

Run:
    pip install flask
    python compliance_ui.py

Then open: http://127.0.0.1:5000
"""

import os
import json
import uuid
import tempfile
from flask import Flask, render_template_string, request, jsonify, session
from werkzeug.utils import secure_filename
from compliance_checker import check_compliance

app = Flask(__name__)
app.secret_key = os.urandom(32)

UPLOAD_FOLDER  = os.path.join(tempfile.gettimeout if False else tempfile.gettempdir(), "imtithal_uploads")
ALLOWED_EXTS   = {"pdf", "txt"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTS


# ═══════════════════════════════════════════════════════════════════════════════
# HTML TEMPLATE (single-file app — no separate static folder needed)
# ═══════════════════════════════════════════════════════════════════════════════
HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>مستشار الامتثال — سدايا & NCA</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@300;400;500;600;700&family=Tajawal:wght@300;400;500;700&display=swap" rel="stylesheet">
<style>
/* ─── RESET & BASE ─── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg-deep:      #0a0e1a;
  --bg-card:      #111828;
  --bg-card-hover:#161f35;
  --border:       #1e2a4a;
  --border-light: #2a3a5e;
  --text-primary: #eef0f6;
  --text-muted:   #7a8aad;
  --text-dim:     #4e5d80;
  --accent:       #4f8cff;
  --accent-glow:  rgba(79,140,255,0.25);
  --green:        #34d9a0;
  --green-bg:     rgba(52,217,160,0.08);
  --red:          #ff5c6a;
  --red-bg:       rgba(255,92,106,0.08);
  --amber:        #f0b840;
  --amber-bg:     rgba(240,184,64,0.08);
  --radius:       12px;
  --radius-sm:    8px;
}

html { scroll-behavior: smooth; }
body {
  font-family: 'IBM Plex Sans Arabic', 'Tajawal', sans-serif;
  background: var(--bg-deep);
  color: var(--text-primary);
  min-height: 100vh;
  line-height: 1.6;
  overflow-x: hidden;
}

/* ─── AMBIENT BACKGROUND ─── */
.ambient {
  position: fixed; inset: 0; z-index: 0; pointer-events: none;
  background:
    radial-gradient(ellipse 80% 50% at 20% 0%,  rgba(79,140,255,0.06) 0%, transparent 70%),
    radial-gradient(ellipse 60% 40% at 80% 100%, rgba(52,217,160,0.05) 0%, transparent 70%),
    radial-gradient(ellipse 50% 60% at 50% 50%,  rgba(255,92,106,0.03) 0%, transparent 70%);
}

/* ─── LAYOUT ─── */
.shell { position: relative; z-index: 1; max-width: 900px; margin: 0 auto; padding: 40px 24px 80px; }

/* ─── HEADER ─── */
.header { text-align: center; margin-bottom: 48px; }
.header .badge {
  display: inline-flex; align-items: center; gap: 8px;
  background: var(--bg-card); border: 1px solid var(--border);
  border-radius: 20px; padding: 6px 16px; font-size: 0.78rem;
  color: var(--accent); letter-spacing: 0.5px; margin-bottom: 20px;
}
.header .badge .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--green); animation: pulse 2s infinite; }
.header h1 {
  font-family: 'Tajawal', sans-serif;
  font-size: clamp(1.8rem, 4vw, 2.6rem);
  font-weight: 700; color: var(--text-primary);
  line-height: 1.3;
}
.header h1 span { color: var(--accent); }
.header p { color: var(--text-muted); font-size: 0.95rem; margin-top: 10px; max-width: 560px; margin-left: auto; margin-right: auto; }

/* ─── UPLOAD CARD ─── */
.upload-card {
  background: var(--bg-card);
  border: 2px dashed var(--border);
  border-radius: var(--radius);
  padding: 48px 24px;
  text-align: center;
  transition: border-color .3s, background .3s, box-shadow .3s;
  cursor: pointer;
  position: relative;
}
.upload-card:hover,
.upload-card.drag-over {
  border-color: var(--accent);
  background: var(--bg-card-hover);
  box-shadow: 0 0 30px var(--accent-glow);
}
.upload-card .icon-wrap {
  width: 72px; height: 72px; border-radius: 18px;
  background: linear-gradient(135deg, rgba(79,140,255,0.12), rgba(52,217,160,0.08));
  border: 1px solid var(--border-light);
  display: flex; align-items: center; justify-content: center;
  margin: 0 auto 20px; font-size: 1.8rem;
  transition: transform .3s;
}
.upload-card:hover .icon-wrap { transform: translateY(-3px) scale(1.05); }
.upload-card h3 { font-size: 1.05rem; color: var(--text-primary); margin-bottom: 6px; }
.upload-card p { font-size: 0.82rem; color: var(--text-dim); }
.upload-card .file-types { margin-top: 12px; }
.upload-card .file-types span {
  display: inline-block; background: rgba(79,140,255,0.1); color: var(--accent);
  border-radius: 6px; padding: 3px 10px; font-size: 0.72rem; margin: 2px;
  letter-spacing: 0.5px;
}
#fileInput { position: absolute; inset: 0; opacity: 0; cursor: pointer; width: 100%; height: 100%; }

/* ─── FILE PREVIEW BAR ─── */
.file-bar {
  display: none;
  background: var(--bg-card); border: 1px solid var(--border);
  border-radius: var(--radius-sm); padding: 12px 18px;
  align-items: center; gap: 12px; margin-top: 16px;
}
.file-bar.show { display: flex; }
.file-bar .fi { font-size: 1.2rem; }
.file-bar .fname { flex: 1; font-size: 0.88rem; color: var(--text-primary); word-break: break-all; }
.file-bar .fsize { font-size: 0.75rem; color: var(--text-dim); white-space: nowrap; }
.file-bar .remove {
  background: none; border: none; color: var(--text-dim); cursor: pointer;
  font-size: 1.1rem; transition: color .2s;
}
.file-bar .remove:hover { color: var(--red); }

/* ─── SUBMIT BUTTON ─── */
.btn-submit {
  display: block; width: 100%; margin-top: 20px; padding: 14px;
  border: none; border-radius: var(--radius-sm);
  background: linear-gradient(135deg, #3a7bd5, #4f8cff);
  color: #fff; font-family: inherit; font-size: 0.95rem; font-weight: 600;
  cursor: pointer; transition: opacity .25s, transform .15s, box-shadow .3s;
  letter-spacing: 0.3px;
}
.btn-submit:hover { opacity: 0.9; box-shadow: 0 4px 20px var(--accent-glow); }
.btn-submit:active { transform: scale(0.98); }
.btn-submit:disabled { opacity: 0.35; cursor: not-allowed; }

/* ─── SPINNER ─── */
.spinner-wrap { display: none; text-align: center; padding: 48px 0; }
.spinner-wrap.show { display: block; }
.spinner {
  width: 44px; height: 44px; border-radius: 50%;
  border: 3px solid var(--border);
  border-top-color: var(--accent);
  animation: spin .7s linear infinite;
  margin: 0 auto 18px;
}
.spinner-wrap p { color: var(--text-muted); font-size: 0.88rem; }

/* ─── SUMMARY STATS ─── */
.summary-bar {
  display: none; gap: 12px; margin-top: 32px;
}
.summary-bar.show { display: flex; flex-wrap: wrap; }
.stat-card {
  flex: 1; min-width: 140px;
  background: var(--bg-card); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 20px 16px; text-align: center;
  transition: transform .2s;
}
.stat-card:hover { transform: translateY(-2px); }
.stat-card .num { font-size: 1.9rem; font-weight: 700; line-height: 1; }
.stat-card .label { font-size: 0.75rem; color: var(--text-dim); margin-top: 6px; }
.stat-card.green { border-color: rgba(52,217,160,0.3); background: var(--green-bg); }
.stat-card.green .num { color: var(--green); }
.stat-card.red   { border-color: rgba(255,92,106,0.3); background: var(--red-bg); }
.stat-card.red   .num { color: var(--red); }
.stat-card.amber { border-color: rgba(240,184,64,0.3); background: var(--amber-bg); }
.stat-card.amber .num { color: var(--amber); }

/* ─── RESULTS LIST ─── */
.results-section { margin-top: 32px; display: none; }
.results-section.show { display: block; }
.results-section > h2 {
  font-size: 1rem; color: var(--text-muted); margin-bottom: 14px;
  font-weight: 500; letter-spacing: 0.3px;
}

.result-card {
  background: var(--bg-card); border: 1px solid var(--border);
  border-radius: var(--radius); margin-bottom: 10px;
  overflow: hidden; transition: border-color .25s;
}
.result-card:hover { border-color: var(--border-light); }

/* header row */
.rc-head {
  display: flex; align-items: center; gap: 12px;
  padding: 14px 18px; cursor: pointer; user-select: none;
}
.rc-head .verdict-badge {
  display: inline-flex; align-items: center; gap: 5px;
  border-radius: 6px; padding: 4px 10px; font-size: 0.72rem; font-weight: 600;
  white-space: nowrap; flex-shrink: 0;
}
.verdict-badge.green { background: var(--green-bg); color: var(--green); border: 1px solid rgba(52,217,160,0.2); }
.verdict-badge.red   { background: var(--red-bg);   color: var(--red);   border: 1px solid rgba(255,92,106,0.2); }
.verdict-badge.amber { background: var(--amber-bg); color: var(--amber); border: 1px solid rgba(240,184,64,0.2); }
.rc-head .subject { flex: 1; font-size: 0.85rem; color: var(--text-primary); min-width: 0; text-overflow: ellipsis; overflow: hidden; white-space: nowrap; }
.rc-head .chevron { color: var(--text-dim); font-size: 0.7rem; transition: transform .25s; }
.result-card.open .chevron { transform: rotate(180deg); }

/* body (collapsible detail) */
.rc-body {
  max-height: 0; overflow: hidden; transition: max-height .35s ease;
  border-top: 1px solid transparent;
}
.result-card.open .rc-body { max-height: 600px; border-top-color: var(--border); }
.rc-inner { padding: 18px; }
.rc-inner .summary-text { font-size: 0.84rem; color: var(--text-muted); margin-bottom: 14px; line-height: 1.7; }

.detail-group { margin-bottom: 14px; }
.detail-group:last-child { margin-bottom: 0; }
.detail-group h4 { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.8px; color: var(--text-dim); margin-bottom: 6px; }
.detail-group ul { list-style: none; }
.detail-group ul li {
  font-size: 0.8rem; color: var(--text-muted); padding: 5px 0 5px 18px;
  position: relative; line-height: 1.5;
}
.detail-group ul li::before {
  content: '▸'; position: absolute; left: 0; color: var(--accent);
}
.detail-group.issues ul li::before { color: var(--red); }
.detail-group.recs ul li::before { color: var(--green); }
.detail-group.regs ul li::before { color: var(--amber); }

.confidence-row {
  display: flex; align-items: center; gap: 8px; margin-top: 10px;
}
.confidence-row span { font-size: 0.72rem; color: var(--text-dim); }
.confidence-dots { display: flex; gap: 4px; }
.confidence-dots .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--border); }
.confidence-dots .dot.active.high { background: var(--green); }
.confidence-dots .dot.active.mid  { background: var(--amber); }
.confidence-dots .dot.active.low  { background: var(--red); }

/* ─── ANIMATIONS ─── */
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }
@keyframes spin  { to { transform: rotate(360deg); } }
@keyframes fadeUp { from { opacity:0; transform:translateY(12px); } to { opacity:1; transform:translateY(0); } }
.result-card { animation: fadeUp .3s ease both; }
.result-card:nth-child(n+2) { animation-delay: calc((n - 1) * 0.04s); }
</style>
</head>
<body>
<div class="ambient"></div>
<div class="shell">

  <!-- HEADER -->
  <header class="header">
    <div class="badge"><span class="dot"></span> نظام الامتثال الذكي</div>
    <h1>فحص الامتثال مع <span>سدايا</span> والأمن السيبراني</h1>
    <p>أرفع وثيقة مشروع شركتك وسنحلل مدى توافقها مع اللوائح والأنظمة تلقائياً.</p>
  </header>

  <!-- UPLOAD -->
  <div class="upload-card" id="uploadZone">
    <input type="file" id="fileInput" accept=".pdf,.txt">
    <div class="icon-wrap">📁</div>
    <h3>أسقط الملف هنا أو انقر للاختيار</h3>
    <p>قبول الملفات بصيغة PDF و TXT فقط</p>
    <div class="file-types"><span>PDF</span><span>TXT</span></div>
  </div>

  <!-- FILE PREVIEW -->
  <div class="file-bar" id="fileBar">
    <span class="fi">📄</span>
    <span class="fname" id="fileNameDisp"></span>
    <span class="fsize" id="fileSizeDisp"></span>
    <button class="remove" id="removeBtn">✕</button>
  </div>

  <!-- SUBMIT -->
  <button class="btn-submit" id="submitBtn" disabled>بدء فحص الامتثال</button>

  <!-- SPINNER -->
  <div class="spinner-wrap" id="spinnerWrap">
    <div class="spinner"></div>
    <p id="spinnerMsg">جاري تحليل المستند …</p>
  </div>

  <!-- SUMMARY STATS -->
  <div class="summary-bar" id="summaryBar">
    <div class="stat-card green"><div class="num" id="numGreen">0</div><div class="label">متوافق</div></div>
    <div class="stat-card red">  <div class="num" id="numRed">0</div>  <div class="label">غير متوافق</div></div>
    <div class="stat-card amber"><div class="num" id="numAmber">0</div><div class="label">يحتاج مراجعة</div></div>
  </div>

  <!-- RESULTS -->
  <div class="results-section" id="resultsSection">
    <h2>تفاصيل النتائج</h2>
    <div id="resultsList"></div>
  </div>

</div><!-- /shell -->

<script>
// ── DOM refs ──
const zone       = document.getElementById('uploadZone');
const input      = document.getElementById('fileInput');
const fileBar    = document.getElementById('fileBar');
const fileNameD  = document.getElementById('fileNameDisp');
const fileSizeD  = document.getElementById('fileSizeDisp');
const removeBtn  = document.getElementById('removeBtn');
const submitBtn  = document.getElementById('submitBtn');
const spinnerW   = document.getElementById('spinnerWrap');
const spinnerMsg = document.getElementById('spinnerMsg');
const summaryBar = document.getElementById('summaryBar');
const numGreen   = document.getElementById('numGreen');
const numRed     = document.getElementById('numRed');
const numAmber   = document.getElementById('numAmber');
const resultsSec = document.getElementById('resultsSection');
const resultsList= document.getElementById('resultsList');

let currentFile  = null;

// ── drag & drop ──
zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag-over'); });
zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
zone.addEventListener('drop', e => {
  e.preventDefault(); zone.classList.remove('drag-over');
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});
input.addEventListener('change', e => { if (e.target.files.length) handleFile(e.target.files[0]); });

function handleFile(file) {
  const ext = file.name.split('.').pop().toLowerCase();
  if (!['pdf','txt'].includes(ext)) { alert('الصيغة غير مدعومة. استخدم PDF أو TXT.'); return; }
  currentFile = file;
  fileNameD.textContent = file.name;
  fileSizeD.textContent = formatSize(file.size);
  fileBar.classList.add('show');
  submitBtn.disabled = false;
}

removeBtn.addEventListener('click', () => {
  currentFile = null;
  input.value = '';
  fileBar.classList.remove('show');
  submitBtn.disabled = true;
});

function formatSize(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1048576) return (bytes/1024).toFixed(1) + ' KB';
  return (bytes/1048576).toFixed(1) + ' MB';
}

// ── SUBMIT ──
submitBtn.addEventListener('click', async () => {
  if (!currentFile) return;
  submitBtn.disabled = true;
  spinnerW.classList.add('show');
  summaryBar.classList.remove('show');
  resultsSec.classList.remove('show');
  resultsList.innerHTML = '';

  const fd = new FormData();
  fd.append('file', currentFile);

  try {
    spinnerMsg.textContent = 'جاري تحليل المستند وفحص الامتثال …';
    const res = await fetch('/check', { method: 'POST', body: fd });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.error || 'خطأ في الخادم');
    }
    const data = await res.json();
    renderReport(data);
  } catch (err) {
    alert('❌ خطأ: ' + err.message);
  } finally {
    spinnerW.classList.remove('show');
    submitBtn.disabled = false;
  }
});

// ── RENDER REPORT ──
function renderReport(data) {
  const s = data.overall_summary;
  numGreen.textContent  = s['متوافق']       || 0;
  numRed.textContent    = s['غير متوافق']   || 0;
  numAmber.textContent  = s['يحتاج مراجعة'] || 0;
  summaryBar.classList.add('show');

  resultsList.innerHTML = '';
  data.results.forEach((r, i) => {
    resultsList.appendChild(buildCard(r, i));
  });
  resultsSec.classList.add('show');

  // scroll to results
  setTimeout(() => resultsSec.scrollIntoView({ behavior:'smooth', block:'start' }), 200);
}

function verdictClass(v) {
  if (v === 'متوافق')       return 'green';
  if (v === 'غير متوافق')   return 'red';
  return 'amber';
}

function buildCard(r, idx) {
  const card = document.createElement('div');
  card.className = 'result-card';
  const cls = verdictClass(r.verdict);
  const confLevel = r.confidence === 'عالي' ? 'high' : r.confidence === 'متوسط' ? 'mid' : 'low';
  const confActive = confLevel === 'high' ? 3 : confLevel === 'mid' ? 2 : 1;

  card.innerHTML = `
    <div class="rc-head" onclick="this.parentElement.classList.toggle('open')">
      <span class="verdict-badge ${cls}">● ${r.verdict}</span>
      <span class="subject">${escHtml(r.subject)}</span>
      <span class="chevron">▼</span>
    </div>
    <div class="rc-body">
      <div class="rc-inner">
        <p class="summary-text">${escHtml(r.summary)}</p>
        ${r.matched_regulations && r.matched_regulations.length
          ? `<div class="detail-group regs"><h4>اللوائح المرتبطة</h4><ul>${r.matched_regulations.map(x=>'<li>'+escHtml(x)+'</li>').join('')}</ul></div>`
          : ''}
        ${r.issues && r.issues.length
          ? `<div class="detail-group issues"><h4>المخالفات / النقاط</h4><ul>${r.issues.map(x=>'<li>'+escHtml(x)+'</li>').join('')}</ul></div>`
          : ''}
        ${r.recommendations && r.recommendations.length
          ? `<div class="detail-group recs"><h4>التوصيات</h4><ul>${r.recommendations.map(x=>'<li>'+escHtml(x)+'</li>').join('')}</ul></div>`
          : ''}
        ${r.regulation_sources && r.regulation_sources.length
          ? `<div class="detail-group"><h4>المصادر التنظيمية</h4><ul>${r.regulation_sources.map(x=>'<li>'+escHtml(x.file_name)+' — '+escHtml(x.subject)+'</li>').join('')}</ul></div>`
          : ''}
        <div class="confidence-row">
          <span>مستوى الثقة</span>
          <div class="confidence-dots">
            ${[1,2,3].map(n => `<div class="dot ${n <= confActive ? 'active '+confLevel : ''}"></div>`).join('')}
          </div>
          <span>(${escHtml(r.confidence)})</span>
        </div>
      </div>
    </div>`;
  return card;
}

function escHtml(s) {
  return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
</script>
</body>
</html>
"""


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════════════
@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/check", methods=["POST"])
def check():
    if "file" not in request.files:
        return jsonify({"error": "لم يتم إرفاق ملف"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "اسم الملف فارغ"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "صيغة الملف غير مدعومة. استخدم PDF أو TXT"}), 400

    # save to temp
    safe_name = secure_filename(file.filename) or "upload"
    ext       = safe_name.rsplit(".", 1)[-1] if "." in safe_name else "txt"
    tmp_path  = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}.{ext}")
    file.save(tmp_path)

    try:
        report = check_compliance(tmp_path)
        return jsonify(report)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # clean up
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  🚀  Compliance UI starting …")
    print("      Open http://127.0.0.1:5000 in your browser")
    print("=" * 70 + "\n")
    app.run(debug=False, host="0.0.0.0", port=5000)