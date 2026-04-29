"""
Search/Text Explorer page — search OCR text across all sessions.
Mounted as a FastAPI router in the main app.
"""

import json
from pathlib import Path
from typing import Optional, List

from fastapi import APIRouter, Query
from fastapi.responses import HTMLResponse

from app_copy_2 import (
    SESSIONS_BASE_DIR,
    _jenv,
    page_number,
    _find_lines,
    _poly_from_line,
)

router = APIRouter()


def _list_sessions_with_scripts() -> List[dict]:
    """Return lightweight list of sessions with their script tags."""
    sessions = []
    if SESSIONS_BASE_DIR.exists():
        for item in SESSIONS_BASE_DIR.iterdir():
            if item.is_dir() and item.name.startswith("hebrocr_") and (item / "out").exists():
                script = ""
                if (item / "script.txt").exists():
                    script = (item / "script.txt").read_text("utf-8").strip()
                sessions.append({"name": item.name, "script": script})
    sessions.sort(key=lambda s: s["name"], reverse=True)
    return sessions


def _search(query: str, session_filter: str = "", script_filter: str = "", limit: int = 100) -> tuple:
    """Search OCR text across sessions. Returns (results, total_count)."""
    results = []
    total = 0
    q_lower = query.lower()

    for item in sorted(SESSIONS_BASE_DIR.iterdir(), key=lambda p: p.name, reverse=True):
        if not item.is_dir() or not item.name.startswith("hebrocr_"):
            continue
        out_dir = item / "out"
        if not out_dir.exists():
            continue

        # Apply session filter
        if session_filter and item.name != session_filter:
            continue

        # Apply script filter
        if script_filter:
            script_path = item / "script.txt"
            sess_script = script_path.read_text("utf-8").strip() if script_path.exists() else ""
            if sess_script != script_filter:
                continue

        # Search text files
        for txt_file in sorted(out_dir.glob("page-*.txt"), key=page_number):
            try:
                text = txt_file.read_text("utf-8", errors="replace")
            except Exception:
                continue
            page_stem = txt_file.stem
            page_num = page_number(txt_file)
            lines = text.splitlines()

            for line_idx, line_text in enumerate(lines):
                if q_lower in line_text.lower():
                    total += 1
                    if len(results) < limit:
                        # Check if line crop exists
                        crop_name = f"{page_stem}_{line_idx + 1:04d}.png"
                        crop_exists = (item / "gt_preview" / crop_name).exists()

                        # Get polygon from seg JSON if available
                        polygon = None
                        seg_json = item / "pages" / f"{page_stem}.json"
                        if seg_json.exists():
                            try:
                                with open(seg_json, "r", encoding="utf-8") as f:
                                    seg = json.load(f)
                                seg_lines = seg.get("lines") if isinstance(seg.get("lines"), list) else _find_lines(seg)
                                if line_idx < len(seg_lines):
                                    poly = _poly_from_line(seg_lines[line_idx])
                                    if poly:
                                        polygon = [[round(x, 1), round(y, 1)] for x, y in poly]
                            except Exception:
                                pass

                        results.append({
                            "session": item.name,
                            "page_num": page_num,
                            "page_stem": page_stem,
                            "line_idx": line_idx,
                            "line_num": line_idx + 1,
                            "text": line_text,
                            "crop_name": crop_name if crop_exists else "",
                            "polygon": polygon,
                        })

    return results, total


@router.get("/search", response_class=HTMLResponse)
def search_page(q: str = "", session: str = "", script: str = ""):
    sessions = _list_sessions_with_scripts()
    scripts = sorted(set(s["script"] for s in sessions if s["script"]))

    results = []
    total = 0
    if q.strip():
        results, total = _search(q.strip(), session_filter=session, script_filter=script)

    return SEARCH_HTML.render(
        q=q,
        session_filter=session,
        script_filter=script,
        sessions=sessions,
        scripts=scripts,
        results=results,
        total=total,
        limit=100,
    )


SEARCH_HTML = _jenv.from_string(r"""
<!doctype html>
<html>
<head><meta charset="utf-8"><title>Text Explorer</title>
{{ THEME_INIT | safe }}
<style>
{{ BASE_CSS | safe }}
.search-wrap{ max-width:1100px; margin:2rem auto; padding:0 1.5rem; }
.top-bar{ display:flex; align-items:center; gap:1rem; flex-wrap:wrap; margin-bottom:1.5rem; }
.search-form{ display:flex; gap:.5rem; flex-wrap:wrap; align-items:center; margin-bottom:1.5rem; }
.search-form input[type=text]{
  flex:1; min-width:200px; padding:.5rem .75rem; font-size:1rem;
  background:var(--panel); color:var(--fg); border:1px solid var(--border); border-radius:6px;
}
.search-form select{
  padding:.45rem .6rem; background:var(--panel); color:var(--fg);
  border:1px solid var(--border); border-radius:6px; font-size:.85rem;
}
.search-form button{
  padding:.5rem 1rem; font-size:.9rem;
}
.results-meta{ color:var(--muted); font-size:.85rem; margin-bottom:1rem; }
.result-card{
  background:var(--panel); border:1px solid var(--border-soft); border-radius:8px;
  padding:1rem; margin-bottom:.75rem; box-shadow:var(--shadow); transition:box-shadow .2s;
}
.result-card:hover{ box-shadow:var(--shadow-strong); }
.result-header{ display:flex; align-items:center; gap:.75rem; margin-bottom:.5rem; flex-wrap:wrap; }
.result-badge{
  font-size:.72rem; padding:.15rem .4rem; border-radius:3px;
  background:rgba(154,120,32,.15); color:var(--accent-gold); font-weight:600;
}
.result-text{
  direction:rtl; text-align:right; font-family:var(--font-mono); font-size:1rem;
  line-height:1.6; padding:.4rem 0; color:var(--fg);
}
.result-text mark{
  background:rgba(154,120,32,.3); color:var(--fg); border-radius:2px; padding:0 2px;
}
.result-body{ display:flex; gap:1rem; align-items:flex-start; }
.result-crop{
  max-width:280px; max-height:60px; border:1px solid var(--border); border-radius:4px;
  flex-shrink:0;
}
.result-actions{ display:flex; gap:.5rem; align-items:center; margin-top:.5rem; }
.result-actions a, .result-actions button{
  font-size:.8rem; padding:.25rem .5rem; border-radius:4px; text-decoration:none;
  border:1px solid var(--border); background:var(--btn); color:var(--fg); cursor:pointer;
}
.result-actions a:hover, .result-actions button:hover{ background:var(--btn-hover); border-color:var(--accent-gold); }
.expanded-view{
  display:none; margin-top:.75rem; position:relative;
  border:1px solid var(--border); border-radius:6px; overflow:auto; max-height:50vh;
}
.expanded-view.open{ display:block; }
.expanded-view img{ display:block; width:100%; }
.expanded-view svg{
  position:absolute; top:0; left:0; width:100%; height:100%; pointer-events:none;
}
.expanded-view .hl-poly{
  fill:rgba(154,120,32,0.2); stroke:var(--poly-hl-stroke, #c84040);
  stroke-width:3px; vector-effect:non-scaling-stroke;
}
.empty-state{ color:var(--muted); text-align:center; padding:3rem 1rem; }
</style>
</head>
<body>
<div class="search-wrap">
  <div class="top-bar">
    <h1 style="margin:0;">Text Explorer</h1>
    <a href="/" class="nav-btn">Home</a>
    <a href="/models" class="nav-btn">Models</a>
    <a href="/sessions" class="nav-btn">Sessions</a>
    <span style="margin-left:auto;">{{ THEME_TOGGLE | safe }}</span>
  </div>

  <form class="search-form" method="get" action="/search">
    <input type="text" name="q" value="{{ q }}" placeholder="Search OCR text across all sessions…" autofocus>
    <select name="session">
      <option value="">All sessions</option>
      {% for s in sessions %}
      <option value="{{ s.name }}" {% if s.name == session_filter %}selected{% endif %}>{{ s.name }}</option>
      {% endfor %}
    </select>
    {% if scripts %}
    <select name="script">
      <option value="">All scripts</option>
      {% for sc in scripts %}
      <option value="{{ sc }}" {% if sc == script_filter %}selected{% endif %}>{{ sc }}</option>
      {% endfor %}
    </select>
    {% endif %}
    <button type="submit">Search</button>
  </form>

  {% if q %}
  <div class="results-meta">
    {% if total == 0 %}
      No matches for "{{ q }}"
    {% elif total <= limit %}
      {{ total }} match{{ 'es' if total != 1 else '' }} for "{{ q }}"
    {% else %}
      Showing {{ limit }} of {{ total }} matches for "{{ q }}"
    {% endif %}
  </div>

  {% for r in results %}
  <div class="result-card">
    <div class="result-header">
      <span class="result-badge">{{ r.session | replace('hebrocr_', '') }}</span>
      <span class="result-badge">Page {{ r.page_num }} : Line {{ r.line_num }}</span>
    </div>
    <div class="result-body">
      {% if r.crop_name %}
      <img class="result-crop" src="/image/{{ r.session }}/{{ r.crop_name }}" loading="lazy" decoding="async" alt="Line crop">
      {% endif %}
      <div class="result-text">{{ r.text | replace(q, '<mark>' ~ q ~ '</mark>') | safe }}</div>
    </div>
    <div class="result-actions">
      <a href="/view?session={{ r.session }}&page={{ r.page_num }}">Open in Viewer</a>
      {% if r.polygon %}
      <button type="button" onclick="toggleExpand(this, '{{ r.session }}', '{{ r.page_stem }}', {{ r.polygon | tojson }})">Show Page Context</button>
      {% endif %}
    </div>
    <div class="expanded-view"></div>
  </div>
  {% endfor %}

  {% elif not q %}
  <div class="empty-state">
    <p style="font-size:1.1rem;">Search all OCR text across your sessions.</p>
    <p>Enter a word or phrase above to find where it appears in your manuscripts.</p>
  </div>
  {% endif %}
</div>

<script>
function toggleExpand(btn, session, pageStem, polygon) {
  var card = btn.closest('.result-card');
  var view = card.querySelector('.expanded-view');
  if (view.classList.contains('open')) {
    view.classList.remove('open');
    btn.textContent = 'Show Page Context';
    return;
  }
  btn.textContent = 'Hide Page Context';
  // Build content if not already loaded
  if (!view.dataset.loaded) {
    var imgSrc = '/image/' + session + '/' + pageStem + '.png';
    var points = polygon.map(function(p){ return p[0] + ',' + p[1]; }).join(' ');
    view.innerHTML =
      '<img src="' + imgSrc + '" onload="this.parentElement.querySelector(\'svg\').setAttribute(\'viewBox\', \'0 0 \' + this.naturalWidth + \' \' + this.naturalHeight)">' +
      '<svg preserveAspectRatio="xMinYMin meet"><polygon class="hl-poly" points="' + points + '"/></svg>';
    view.dataset.loaded = '1';
  }
  view.classList.add('open');
}
</script>
</body>
</html>
""")
