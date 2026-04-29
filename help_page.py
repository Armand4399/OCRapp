"""
Help/How-To page — simple workflow instructions.
Mounted as a FastAPI router in the main app.
"""

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from app_copy_2 import _jenv

router = APIRouter()


@router.get("/help", response_class=HTMLResponse)
def help_page():
    return HELP_HTML.render()


HELP_HTML = _jenv.from_string(r"""
<!doctype html>
<html>
<head><meta charset="utf-8"><title>How to Use</title>
{{ THEME_INIT | safe }}
<style>
{{ BASE_CSS | safe }}
.help-wrap{ max-width:800px; margin:2rem auto; padding:0 1.5rem; }
.top-bar{ display:flex; align-items:center; gap:1rem; flex-wrap:wrap; margin-bottom:2rem; }
.step{
  background:var(--panel); border:1px solid var(--border-soft); border-radius:8px;
  padding:1.25rem 1.5rem; margin-bottom:1rem; box-shadow:var(--shadow);
}
.step h3{ margin:0 0 .5rem; color:var(--accent-gold); font-size:1rem; }
.step p, .step ul{ margin:0; color:var(--fg); font-size:.92rem; line-height:1.6; }
.step ul{ padding-left:1.2rem; margin-top:.3rem; }
.step li{ margin-bottom:.25rem; }
.step code{
  background:var(--sunken); padding:.1rem .35rem; border-radius:3px;
  font-size:.85rem; color:var(--accent-gold);
}
.section-title{ color:var(--muted); font-size:.8rem; text-transform:uppercase; letter-spacing:.1em; margin:1.5rem 0 .5rem; }
</style>
</head>
<body>
<div class="help-wrap">
  <div class="top-bar">
    <h1 style="margin:0;">How to Use</h1>
    <a href="/" class="nav-btn">Home</a>
    <a href="/sessions" class="nav-btn">Sessions</a>
    <a href="/models" class="nav-btn">Models</a>
    <a href="/search" class="nav-btn">Search</a>
    <span style="margin-left:auto;">{{ THEME_TOGGLE | safe }}</span>
  </div>

  <div class="section-title">Basic Workflow</div>

  <div class="step">
    <h3>1. Upload an Image or PDF</h3>
    <p>From the home page, select a manuscript image (PNG, JPG, TIFF) or a multi-page PDF.
       Choose your model, preprocessing options, and hit <strong>Start OCR</strong>.</p>
    <ul>
      <li><strong>Grayscale</strong> preprocessing is recommended for most models</li>
      <li><strong>Auto-deskew</strong> helps with tilted scans</li>
      <li><strong>Auto-merge split lines</strong> fixes lines that got incorrectly segmented into fragments</li>
    </ul>
  </div>

  <div class="step">
    <h3>2. Review in the Viewer</h3>
    <p>Once processing finishes, you land in the <strong>Viewer</strong>. Here you can:</p>
    <ul>
      <li>Read and edit the OCR text (auto-saves as you type)</li>
      <li>Toggle between Original/Processed/Text/Notes panels</li>
      <li>Switch between <strong>Split</strong> and <strong>Reading</strong> layouts</li>
      <li>Click a polygon on the processed image to jump to that line in the text</li>
      <li>Use <strong>Image Filters</strong> (invert, contrast, brightness) to better read faded manuscripts</li>
      <li>Change polygon colors for better visibility against your manuscript</li>
    </ul>
  </div>

  <div class="step">
    <h3>3. Correct the Text</h3>
    <p>Edit the OCR output directly in the text panel. Each line corresponds to a detected text line
       in the manuscript. Changes save automatically after a short pause.</p>
    <ul>
      <li>Use the <strong>font selector</strong> to switch between monospace, Hebrew serif, or sans-serif</li>
      <li>Adjust <strong>font size</strong> for comfortable reading</li>
      <li>The <strong>Notes</strong> panel (toggle it on) gives you a parallel workspace for annotations</li>
    </ul>
  </div>

  <div class="step">
    <h3>4. Reprocess if Needed</h3>
    <p>If segmentation or OCR quality is poor, use the <strong>Reprocess</strong> dropdown:</p>
    <ul>
      <li><strong>Reprocess this page</strong> — re-runs segmentation + OCR from scratch</li>
      <li><strong>Re-OCR page (keep polygons)</strong> — keeps your edited line boundaries, just re-runs text recognition</li>
      <li><strong>Re-OCR all pages</strong> — same as above but for the entire session</li>
    </ul>
  </div>

  <div class="section-title">Training Workflow</div>

  <div class="step">
    <h3>5. Edit Line Boundaries (Training Page)</h3>
    <p>Click <strong>Open Training</strong> from the viewer to enter the training annotation view.
       Here you can drag polygon vertices to fix line boundaries, delete bad lines, and edit
       per-line ground truth text.</p>
  </div>

  <div class="step">
    <h3>6. Export Ground Truth</h3>
    <p>Set a <strong>script tag</strong> (e.g. "Hebrew_square") on your session, then click
       <strong>Export to Validation Set</strong>. This copies your corrected line crops + text
       into the training data folder, organized by script.</p>
  </div>

  <div class="step">
    <h3>7. Fine-tune a Model</h3>
    <p>Go to <strong>Models</strong>, select a model, assign it the same script tag, and click
       <strong>Train</strong>. The app runs <code>ketos train</code> to fine-tune on your
       ground truth data. Use <strong>Quick Test</strong> to measure improvement.</p>
  </div>

  <div class="section-title">Other Features</div>

  <div class="step">
    <h3>Search Text</h3>
    <p>The <strong>Search</strong> page lets you find any word or phrase across all sessions.
       Results show the matching line with its source image. Click "Show Page Context" to see
       the full page with the line highlighted.</p>
  </div>

  <div class="step">
    <h3>Sessions</h3>
    <p>The <strong>Sessions</strong> page lists all your OCR runs, sorted by most recent.
       You can jump into the viewer or training page, or delete old sessions.</p>
  </div>

  <div class="step">
    <h3>Exports</h3>
    <ul>
      <li><strong>Download merged text</strong> — all pages concatenated into one .txt file</li>
      <li><strong>Export CSV</strong> — structured export with folio, page, line metadata</li>
      <li><strong>Export GT</strong> — zip of line crops + ground truth text for external tools</li>
    </ul>
  </div>

  <div class="section-title">Tips</div>

  <div class="step">
    <h3>Keyboard &amp; Quick Actions</h3>
    <ul>
      <li>Text auto-saves — no need to hit Save manually</li>
      <li>Panel toggle states persist across pages (localStorage)</li>
      <li>Theme, font size, font family, and polygon colors all persist</li>
      <li>Use <code>Ctrl+C</code> in terminal to stop the server</li>
    </ul>
  </div>
</div>
</body>
</html>
""")
