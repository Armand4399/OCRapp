"""
Sessions page — browse, view, and manage existing OCR sessions.
Mounted as a FastAPI router in the main app.
"""

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from app import (
    SESSIONS_BASE_DIR,
    _jenv,
    page_number,
)

router = APIRouter()


def _session_info(sess_dir: Path) -> dict:
    """Read lightweight metadata from a session directory."""
    name = sess_dir.name
    model = ""
    if (sess_dir / "model.txt").exists():
        model = Path((sess_dir / "model.txt").read_text("utf-8").strip()).stem
    script = ""
    if (sess_dir / "script.txt").exists():
        script = (sess_dir / "script.txt").read_text("utf-8").strip()
    preprocess = ""
    if (sess_dir / "preprocess.txt").exists():
        preprocess = (sess_dir / "preprocess.txt").read_text("utf-8").strip()
    page_count = 0
    pages_dir = sess_dir / "pages"
    if pages_dir.exists():
        page_count = len([
            p for p in pages_dir.iterdir()
            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff")
            and p.stem.startswith("page-")
        ])
    segment_only = ""
    if (sess_dir / "segment_only.txt").exists():
        segment_only = (sess_dir / "segment_only.txt").read_text("utf-8").strip()

    thumb_name = ""
    if pages_dir.exists():
        image_exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
        page_imgs = sorted(
            [p for p in pages_dir.iterdir()
             if p.suffix.lower() in image_exts and p.stem.startswith("page-")],
            key=page_number,
        )
        if page_imgs:
            first = page_imgs[0]
            copy_name = f"{first.stem}_orig{first.suffix.lower()}"
            copy_path = sess_dir / copy_name
            if not copy_path.exists():
                try:
                    import shutil
                    shutil.copy(first, copy_path)
                except Exception:
                    copy_name = ""
            thumb_name = copy_name

    return {
        "name": name,
        "model": model,
        "script": script,
        "preprocess": preprocess,
        "page_count": page_count,
        "segment_only": segment_only == "yes",
        "thumb": thumb_name,
    }


@router.get("/sessions", response_class=HTMLResponse)
def sessions_page():
    sessions = []
    if SESSIONS_BASE_DIR.exists():
        for item in SESSIONS_BASE_DIR.iterdir():
            if item.is_dir() and item.name.startswith("hebrocr_") and (item / "pages").exists():
                info = _session_info(item)
                info["_mtime"] = item.stat().st_mtime
                sessions.append(info)
    sessions.sort(key=lambda s: s["_mtime"], reverse=True)
    return SESSIONS_HTML.render(sessions=sessions)


SESSIONS_HTML = _jenv.from_string("""
<!doctype html>
<html>
<head><meta charset="utf-8"><title>Sessions</title>
{{ THEME_INIT | safe }}
<style>
{{ BASE_CSS | safe }}
.sessions-grid{ display:grid; grid-template-columns:repeat(auto-fill, minmax(340px, 1fr)); gap:1rem; margin-top:1rem; }
.session-card{
  background:var(--panel); border:1px solid var(--border-soft); border-radius:8px;
  padding:1.15rem; box-shadow:var(--shadow); transition:box-shadow .2s;
  display:flex; gap:1rem;
}
.session-card:hover{ box-shadow:var(--shadow-strong); }
.session-thumb{
  width:90px; min-width:90px; height:120px; object-fit:cover;
  border-radius:4px; border:1px solid var(--border);
  background:var(--sunken);
}
.session-body{ flex:1; min-width:0; }
.session-name{
  font-family:var(--font-mono); font-size:.92rem; font-weight:700;
  margin-bottom:.5rem; word-break:break-all;
}
.session-meta{ display:grid; grid-template-columns:auto 1fr; gap:.2rem .7rem; font-size:.88rem; }
.meta-label{ color:var(--muted); }
.meta-value{ font-weight:500; }
.session-actions{ display:flex; gap:.5rem; margin-top:.75rem; flex-wrap:wrap; }
.session-actions a, .session-actions button{
  padding:.3rem .7rem; border-radius:4px; font-size:.85rem;
  text-decoration:none; cursor:pointer; font-family:var(--font-display);
  letter-spacing:.04em; border:1px solid var(--border);
}
.session-actions a{ background:var(--btn); color:var(--fg); }
.session-actions a:hover{ background:var(--btn-hover); border-color:var(--accent-gold); }
.del-btn{
  background:transparent; color:#b44; border-color:#b44;
}
.del-btn:hover{ background:rgba(180,60,60,.12); }
.top-bar{ display:flex; align-items:center; gap:1rem; flex-wrap:wrap; }
.empty-msg{ color:var(--muted); margin-top:2rem; }
.badge{
  display:inline-block; font-size:.72rem; padding:.1rem .4rem; border-radius:3px;
  font-weight:600; text-transform:uppercase; letter-spacing:.04em;
  background:rgba(154,120,32,.15); color:var(--accent-gold);
}
</style>
</head>
<body>
<div class="page-centered" style="max-width:1100px;margin:2rem auto;padding:0 1.5rem;">
  <div class="top-bar">
    <h1 style="margin:0;">Sessions</h1>
    <a href="/" class="nav-btn">Home</a>
    <a href="/models" class="nav-btn">Models</a>
    <span style="margin-left:auto;">{{ THEME_TOGGLE | safe }}</span>
  </div>

  {% if sessions %}
  <div class="sessions-grid">
    {% for s in sessions %}
    <div class="session-card">
      {% if s.thumb %}
      <img class="session-thumb" loading="lazy" decoding="async"
           src="/thumb/{{ s.name }}/{{ s.thumb }}?w=180" alt="Preview">
      {% endif %}
      <div class="session-body">
        <div class="session-name">{{ s.name }}</div>
        <div class="session-meta">
          <span class="meta-label">Pages</span>
          <span class="meta-value">{{ s.page_count }}</span>

          <span class="meta-label">Model</span>
          <span class="meta-value">{{ s.model or '—' }}</span>

          {% if s.script %}
          <span class="meta-label">Script</span>
          <span class="meta-value">{{ s.script }}</span>
          {% endif %}

          <span class="meta-label">Preprocess</span>
          <span class="meta-value">{{ s.preprocess or '—' }}{% if s.segment_only %} <span class="badge">seg only</span>{% endif %}</span>
        </div>

        <div class="session-actions">
          <a href="/view?session={{ s.name }}">View</a>
          <a href="/training?session={{ s.name }}">Train</a>
          <form action="/delete_session" method="post" style="display:inline;margin:0;"
                onsubmit="return confirm('Permanently delete session {{ s.name }}?');">
            <input type="hidden" name="session" value="{{ s.name }}">
            <button type="submit" class="del-btn">Delete</button>
          </form>
        </div>
      </div>
    </div>
    {% endfor %}
  </div>
  {% else %}
  <p class="empty-msg">No sessions found. <a href="/">Upload an image</a> to create one.</p>
  {% endif %}
</div>
</body>
</html>
""")
