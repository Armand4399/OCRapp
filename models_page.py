"""
Models page — grid overview + per-model detail view.
Mounted as a FastAPI router in the main app.
"""

import json
import os
import re
import subprocess
import threading
import tempfile
import time
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Form, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response

from app_copy_2 import (
    APP_ROOT,
    MODELS_DIR,
    FINAL_GT_DIR,
    PROGRESS,
    DEVICE,
    list_script_dirs,
    _jenv,
    SESSIONS_BASE_DIR,
)

router = APIRouter()

TRAIN_DIR = APP_ROOT / "training" / "train"
KETOS_BIN = Path.home() / "kraken-env" / "bin" / "ketos"
GT_BASE = APP_ROOT / "training"

_SCRIPTS_FILE = MODELS_DIR / "model_scripts.json"
_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


# ===================== helpers =====================

def _load_model_scripts() -> dict:
    try:
        return json.loads(_SCRIPTS_FILE.read_text("utf-8"))
    except Exception:
        return {}


def _save_model_scripts(data: dict):
    _SCRIPTS_FILE.write_text(json.dumps(data, indent=2), "utf-8")


def _load_model_info(path: Path, scripts_map: dict) -> dict:
    stat = path.stat()
    info: Dict[str, Any] = {
        "name": path.stem,
        "path": str(path),
        "model_type": "unknown",
        "best_accuracy": None,
        "completed_epochs": None,
        "input_spec": "",
        "file_size_mb": round(stat.st_size / (1024 * 1024), 1),
        "script": scripts_map.get(path.stem, ""),
        "charset": "",
        "charset_count": 0,
        "accuracy_history": [],
        "last_modified": datetime.fromtimestamp(stat.st_mtime).strftime("%b %d, %Y"),
        "has_log": (MODELS_DIR / f"{path.stem}_train.log").exists(),
    }
    try:
        from kraken.lib import vgsl
        model = vgsl.TorchVGSLModel.load_model(str(path))
        info["model_type"] = model.model_type or "unknown"
        info["input_spec"] = str(model.input)
        meta = model.user_metadata or {}
        hp = meta.get("hyper_params", {})
        info["completed_epochs"] = hp.get("completed_epochs") or meta.get("completed_epochs")
        acc_list = meta.get("accuracy")
        if isinstance(acc_list, list) and acc_list:
            info["best_accuracy"] = round(max(a[1] for a in acc_list if len(a) >= 2) * 100, 1)
            info["accuracy_history"] = [[a[0], round(a[1] * 100, 2)] for a in acc_list if len(a) >= 2]
        if hasattr(model, "codec") and hasattr(model.codec, "c2l"):
            chars = sorted(model.codec.c2l.keys())
            info["charset"] = " ".join(chars)
            info["charset_count"] = len(chars)
    except Exception:
        pass
    return info


def _count_gt_lines(script: str) -> dict:
    counts = {"train": 0, "val": 0}
    for label, base_dir in [("train", TRAIN_DIR), ("val", FINAL_GT_DIR)]:
        script_dir = base_dir / script
        if script_dir.is_dir():
            counts[label] = len(list(script_dir.glob("*.png")))
        elif not script and base_dir.is_dir():
            counts[label] = len(list(base_dir.glob("*.png")))
    return counts


def _sparkline_points(history: list, width: int = 140, height: int = 32) -> str:
    if not history:
        return ""
    accs = [h[1] for h in history]
    mn, mx = min(accs), max(accs)
    rng = mx - mn if mx != mn else 1
    n = len(accs)
    pts = []
    for i, a in enumerate(accs):
        x = round(i / max(n - 1, 1) * width, 1)
        y = round(height - (a - mn) / rng * height, 1)
        pts.append(f"{x},{y}")
    return " ".join(pts)


def _char_error_rate(ref: str, hyp: str) -> float:
    if not ref:
        return 0.0 if not hyp else 1.0
    n, m = len(ref), len(hyp)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[m] / n


# ===================== routes =====================

# ---- Grid overview ----
@router.get("/models", response_class=HTMLResponse)
def models_page():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scripts_map = _load_model_scripts()
        model_paths = sorted(MODELS_DIR.glob("*.mlmodel"), key=lambda p: p.stem.lower())
        models = [_load_model_info(p, scripts_map) for p in model_paths]
    return MODELS_GRID_HTML.render(models=models, sparkline_points=_sparkline_points)


# ---- Detail view ----
@router.get("/models/{name}", response_class=HTMLResponse)
def model_detail(name: str):
    if not _NAME_RE.match(name):
        return PlainTextResponse("Invalid model name.", status_code=400)
    model_path = MODELS_DIR / f"{name}.mlmodel"
    if not model_path.exists():
        return HTMLResponse("<h3>Model not found.</h3><a href='/models'>Back</a>", status_code=404)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scripts_map = _load_model_scripts()
        m = _load_model_info(model_path, scripts_map)

    scripts = list_script_dirs()
    gt_counts = _count_gt_lines(m["script"]) if m["script"] else {"train": 0, "val": 0}
    models_json = json.dumps(m["accuracy_history"])

    # For comparison dropdown: load basic info of all other models
    all_models = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for p in sorted(MODELS_DIR.glob("*.mlmodel"), key=lambda p: p.stem.lower()):
            if p.stem != name:
                other = _load_model_info(p, scripts_map)
                all_models.append(other)

    return MODEL_DETAIL_HTML.render(
        m=m, scripts=scripts, gt_counts=gt_counts,
        accuracy_json=models_json, other_models=all_models,
        sparkline_points=_sparkline_points,
    )


# ---- API routes ----

@router.post("/set_model_script")
def set_model_script(model_name: str = Form(...), script: str = Form("")):
    data = _load_model_scripts()
    if script:
        data[model_name] = script
    else:
        data.pop(model_name, None)
    _save_model_scripts(data)
    return JSONResponse({"ok": True})


@router.post("/delete_model")
def delete_model(model_name: str = Form(...)):
    if not _NAME_RE.match(model_name):
        return PlainTextResponse("Invalid model name.", status_code=400)
    model_path = MODELS_DIR / f"{model_name}.mlmodel"
    if not model_path.exists():
        return PlainTextResponse("Model not found.", status_code=404)
    model_path.unlink()
    (MODELS_DIR / f"{model_name}_train.log").unlink(missing_ok=True)
    data = _load_model_scripts()
    data.pop(model_name, None)
    _save_model_scripts(data)
    return JSONResponse({"ok": True})


@router.post("/rename_model")
def rename_model(old_name: str = Form(...), new_name: str = Form(...)):
    new_name = new_name.strip()
    if not _NAME_RE.match(old_name) or not _NAME_RE.match(new_name):
        return PlainTextResponse("Invalid name. Use letters, numbers, dashes, underscores.", status_code=400)
    old_path = MODELS_DIR / f"{old_name}.mlmodel"
    new_path = MODELS_DIR / f"{new_name}.mlmodel"
    if not old_path.exists():
        return PlainTextResponse("Model not found.", status_code=404)
    if new_path.exists():
        return PlainTextResponse(f"A model named '{new_name}' already exists.", status_code=409)
    old_path.rename(new_path)
    old_log = MODELS_DIR / f"{old_name}_train.log"
    if old_log.exists():
        old_log.rename(MODELS_DIR / f"{new_name}_train.log")
    data = _load_model_scripts()
    if old_name in data:
        data[new_name] = data.pop(old_name)
        _save_model_scripts(data)
    return JSONResponse({"ok": True, "new_name": new_name})


@router.get("/model_log/{name}")
def model_log(name: str):
    if not _NAME_RE.match(name):
        return PlainTextResponse("Invalid name.", status_code=400)
    log_path = MODELS_DIR / f"{name}_train.log"
    if not log_path.exists():
        return PlainTextResponse("No training log found.", status_code=404)
    return PlainTextResponse(log_path.read_text("utf-8", errors="replace"))


@router.post("/train")
def train(
    model_name: str = Form(...),
    epochs: int = Form(-1),
    lrate: float = Form(0.001),
    batch_size: int = Form(1),
    lag: int = Form(10),
):
    scripts_map = _load_model_scripts()
    script = scripts_map.get(model_name, "")
    if not script:
        return PlainTextResponse("No script assigned to this model.", status_code=400)
    model_path = MODELS_DIR / f"{model_name}.mlmodel"
    if not model_path.exists():
        return PlainTextResponse("Model file not found.", status_code=404)
    train_dir = TRAIN_DIR / script
    val_dir = FINAL_GT_DIR / script
    if not train_dir.is_dir():
        train_dir = TRAIN_DIR
    if not val_dir.is_dir():
        val_dir = FINAL_GT_DIR
    train_files = sorted(train_dir.glob("*.png"))
    val_files = sorted(val_dir.glob("*.png"))
    if not train_files:
        return PlainTextResponse(f"No training data found for script '{script}'.", status_code=400)
    output_stem = f"{model_name}_ft"
    n = 1
    while (MODELS_DIR / f"{output_stem}.mlmodel").exists():
        n += 1
        output_stem = f"{model_name}_ft_{n}"
    job_id = output_stem
    if job_id in PROGRESS and PROGRESS[job_id].get("state") not in ("done", "error", None):
        return PlainTextResponse("Training already in progress.", status_code=409)
    PROGRESS[job_id] = {"total": 0, "done_pages": 0, "state": "preparing", "errors": []}
    threading.Thread(
        target=_train_job,
        args=(str(model_path), output_stem, train_files, val_files, epochs, lrate, batch_size, lag),
        daemon=True,
    ).start()
    return JSONResponse({"ok": True, "job": job_id})


def _train_job(model_path, output_stem, train_files, val_files, epochs, lrate, batch_size, lag):
    output_path = str(MODELS_DIR / output_stem)
    job_id = output_stem
    log_path = MODELS_DIR / f"{output_stem}_train.log"
    val_list_file = None
    try:
        if val_files:
            val_list_file = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, prefix="kraken_val_")
            for vf in val_files:
                val_list_file.write(str(vf) + "\n")
            val_list_file.close()
        cmd = [
            str(KETOS_BIN), "train", "-d", DEVICE,
            "-i", model_path, "-f", "path", "-o", output_path,
            "-N", str(epochs), "-r", str(lrate), "-B", str(batch_size), "--lag", str(lag),
        ]
        if val_list_file:
            cmd += ["-e", val_list_file.name]
        cmd += [str(f) for f in train_files]
        PROGRESS[job_id]["state"] = "training"
        log_lines = []
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            line = line.strip()
            if line:
                log_lines.append(line)
                PROGRESS[job_id]["state"] = line[-120:]
        proc.wait()
        log_path.write_text("\n".join(log_lines), "utf-8")
        if proc.returncode == 0:
            PROGRESS[job_id]["state"] = "done"
        else:
            PROGRESS[job_id]["state"] = f"error: ketos exited with code {proc.returncode}"
    except Exception as e:
        PROGRESS[job_id]["state"] = f"error: {e}"
    finally:
        if val_list_file:
            Path(val_list_file.name).unlink(missing_ok=True)


@router.get("/gt_samples/{script}")
def gt_samples(script: str, limit: int = Query(6)):
    if not _NAME_RE.match(script):
        return JSONResponse([])
    samples = []
    for base_dir in [TRAIN_DIR, FINAL_GT_DIR]:
        d = base_dir / script
        if not d.is_dir():
            d = base_dir
        for png in sorted(d.glob("*.png"))[:limit]:
            gt_txt = png.with_suffix(".gt.txt")
            text = gt_txt.read_text("utf-8", errors="replace").strip() if gt_txt.exists() else ""
            samples.append({"name": png.name, "text": text, "dir": base_dir.name, "path": str(png)})
            if len(samples) >= limit:
                break
        if len(samples) >= limit:
            break
    return JSONResponse(samples)


@router.get("/gt_thumb")
def gt_thumb(path: str, w: int = Query(200)):
    safe = Path(path).resolve()
    if not str(safe).startswith(str(GT_BASE.resolve())):
        return PlainTextResponse("Forbidden", status_code=403)
    if not safe.exists() or safe.suffix.lower() not in (".png", ".jpg", ".jpeg"):
        return PlainTextResponse("Not found", status_code=404)
    try:
        from PIL import Image
        import io
        im = Image.open(safe)
        ratio = w / im.width
        new_h = max(1, int(im.height * ratio))
        im = im.resize((w, new_h), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        buf.seek(0)
        return Response(content=buf.getvalue(), media_type="image/png",
                        headers={"Cache-Control": "public, max-age=86400"})
    except Exception:
        return PlainTextResponse("Error generating thumbnail", status_code=500)


@router.get("/gt_browse/{script}")
def gt_browse(script: str):
    if not _NAME_RE.match(script):
        return JSONResponse({"lines": []})
    from PIL import Image
    lines = []
    for label, base_dir in [("train", TRAIN_DIR), ("val", FINAL_GT_DIR)]:
        d = base_dir / script
        if not d.is_dir():
            d = base_dir
        for png in sorted(d.glob("*.png")):
            gt_txt = png.with_suffix(".gt.txt")
            text = gt_txt.read_text("utf-8", errors="replace").strip() if gt_txt.exists() else ""
            try:
                im = Image.open(png)
                w, h = im.size
                im.close()
            except Exception:
                w, h = 0, 0
            ratio = w / h if h else 0
            flagged = ratio < 2.0 or h > 350 or w < 200 or (len(text) < 3 and w > 500)
            lines.append({
                "path": str(png), "gt_text": text, "width": w, "height": h,
                "size_kb": round(png.stat().st_size / 1024, 1),
                "source": label, "flagged": flagged, "name": png.name,
            })
    return JSONResponse({"lines": lines})


@router.post("/gt_delete")
async def gt_delete(request: Request):
    body = await request.json()
    paths = body.get("paths", [])
    if not paths:
        return JSONResponse({"deleted": 0})
    gt_base_resolved = str(GT_BASE.resolve())
    deleted = 0
    for p in paths:
        safe = Path(p).resolve()
        if not str(safe).startswith(gt_base_resolved):
            continue
        if not safe.exists():
            continue
        safe.unlink(missing_ok=True)
        safe.with_suffix(".gt.txt").unlink(missing_ok=True)
        deleted += 1
    return JSONResponse({"deleted": deleted})


@router.post("/quick_test")
def quick_test(model_name: str = Form(...), num_lines: int = Form(5)):
    scripts_map = _load_model_scripts()
    script = scripts_map.get(model_name, "")
    if not script:
        return JSONResponse({"error": "No script assigned."}, status_code=400)
    model_path = MODELS_DIR / f"{model_name}.mlmodel"
    if not model_path.exists():
        return JSONResponse({"error": "Model not found."}, status_code=404)
    gt_files = []
    for base_dir in [FINAL_GT_DIR, TRAIN_DIR]:
        d = base_dir / script
        if not d.is_dir():
            d = base_dir
        for png in d.glob("*.png"):
            gt_txt = png.with_suffix(".gt.txt")
            if gt_txt.exists() and gt_txt.stat().st_size > 0:
                gt_files.append((png, gt_txt))
    if not gt_files:
        return JSONResponse({"error": f"No GT data with text found for '{script}'."}, status_code=400)
    sample = random.sample(gt_files, min(num_lines, len(gt_files)))
    from app_copy_2 import _get_rec_model, _kr_rpred
    from kraken.containers import Segmentation, BaselineLine
    from PIL import Image
    rec_model = _get_rec_model(str(model_path))
    results = []
    total_chars = 0
    total_errors = 0
    for png_path, gt_path in sample:
        ground_truth = gt_path.read_text("utf-8", errors="replace").strip()
        prediction = ""
        try:
            im = Image.open(png_path)
            w, h = im.size
            # Build a single-line segmentation covering the full image
            seg = Segmentation(
                type="baselines",
                imagename=str(png_path),
                text_direction="horizontal-rl",
                lines=[BaselineLine(
                    id="line_0001",
                    baseline=[[0, h // 2], [w, h // 2]],
                    boundary=[[0, 0], [w, 0], [w, h], [0, h]],
                    tags={"type": "default"},
                )],
                script_detection=False,
                regions={},
                line_orders=[],
            )
            for record in _kr_rpred.rpred(rec_model, im, seg, bidi_reordering=True):
                prediction = record.prediction
                break
        except Exception as e:
            prediction = f"[error: {e}]"
        cer = _char_error_rate(ground_truth, prediction)
        total_chars += len(ground_truth)
        total_errors += int(round(cer * len(ground_truth)))
        results.append({
            "image": str(png_path), "ground_truth": ground_truth,
            "prediction": prediction, "cer": round(cer * 100, 1),
        })
    overall_cer = round(total_errors / max(total_chars, 1) * 100, 1)
    return JSONResponse({"results": results, "overall_cer": overall_cer})


# ===================== templates =====================

# ---- Grid overview (slim cards) ----
MODELS_GRID_HTML = _jenv.from_string(r"""
<!doctype html>
<html>
<head><meta charset="utf-8"><title>Models</title>
{{ THEME_INIT | safe }}
<style>
{{ BASE_CSS | safe }}
.models-grid{ display:grid; grid-template-columns:repeat(auto-fill, minmax(280px, 1fr)); gap:1rem; margin-top:1rem; }
.model-card{
  background:var(--panel); border:1px solid var(--border-soft); border-radius:8px;
  padding:1rem 1.15rem; box-shadow:var(--shadow); transition:box-shadow .2s;
  cursor:pointer; text-decoration:none; color:inherit; display:block;
}
.model-card:hover{ box-shadow:var(--shadow-strong); border-color:var(--accent-gold); }
.card-header{ display:flex; align-items:center; gap:.5rem; margin-bottom:.4rem; }
.card-name{ font-family:var(--font-display); font-size:1.05rem; font-weight:700; }
.model-badge{
  display:inline-block; font-size:.68rem; padding:.1rem .4rem; border-radius:3px;
  font-weight:600; text-transform:uppercase; letter-spacing:.05em;
}
.badge-recognition{ background:rgba(154,120,32,.18); color:var(--accent-gold); }
.badge-segmentation{ background:rgba(80,160,200,.18); color:#5ac; }
.badge-unknown{ background:rgba(128,128,128,.18); color:var(--muted); }
.card-stats{ display:flex; gap:1rem; font-size:.85rem; color:var(--muted); margin-top:.3rem; }
.card-stats strong{ color:var(--fg); }
.sparkline-wrap{ margin-top:.4rem; }
.top-bar{ display:flex; align-items:center; gap:1rem; flex-wrap:wrap; }
</style>
</head>
<body>
<div class="page-centered" style="max-width:1100px;margin:2rem auto;padding:0 1.5rem;">
  <div class="top-bar">
    <h1 style="margin:0;">Models</h1>
    <a href="/" class="nav-btn">Home</a>
    <a href="/sessions" class="nav-btn">Sessions</a>
    <span style="margin-left:auto;">{{ THEME_TOGGLE | safe }}</span>
  </div>

  <div class="models-grid">
  {% for m in models %}
    <a class="model-card" href="/models/{{ m.name }}">
      <div class="card-header">
        <span class="card-name">{{ m.name }}</span>
        <span class="model-badge badge-{{ m.model_type }}">{{ m.model_type }}</span>
      </div>
      <div class="card-stats">
        <span>{% if m.best_accuracy is not none %}<strong>{{ m.best_accuracy }}%</strong> acc{% else %}—{% endif %}</span>
        <span>{{ m.completed_epochs or '—' }} epochs</span>
        <span>{{ m.file_size_mb }} MB</span>
      </div>
      {% if m.accuracy_history %}
      <div class="sparkline-wrap">
        <svg width="100%" height="28" viewBox="0 0 144 28" preserveAspectRatio="none">
          <polyline points="{{ sparkline_points(m.accuracy_history, 140, 24) }}"
                    fill="none" stroke="var(--accent-gold,#9a7820)" stroke-width="1.5"
                    transform="translate(2,2)"/>
        </svg>
      </div>
      {% endif %}
      {% if m.script %}<div style="font-size:.8rem;color:var(--muted);margin-top:.3rem;">{{ m.script }}</div>{% endif %}
    </a>
  {% endfor %}
  </div>

  {% if not models %}
  <p style="color:var(--muted);margin-top:2rem;">No .mlmodel files found in models/</p>
  {% endif %}
</div>
</body>
</html>
""")


# ---- Detail view (single model) ----
MODEL_DETAIL_HTML = _jenv.from_string(r"""
<!doctype html>
<html>
<head><meta charset="utf-8"><title>{{ m.name }} — Model</title>
{{ THEME_INIT | safe }}
<style>
{{ BASE_CSS | safe }}
.detail-wrap{ max-width:900px; margin:2rem auto; padding:0 1.5rem; }
.top-bar{ display:flex; align-items:center; gap:1rem; flex-wrap:wrap; margin-bottom:1.5rem; }
.model-title{
  font-family:var(--font-display); font-size:1.4rem; font-weight:700;
  cursor:pointer; border-bottom:1px dashed transparent;
}
.model-title:hover{ border-bottom-color:var(--muted); }
.model-badge{
  display:inline-block; font-size:.75rem; padding:.12rem .5rem; border-radius:3px;
  font-weight:600; text-transform:uppercase; letter-spacing:.05em;
}
.badge-recognition{ background:rgba(154,120,32,.18); color:var(--accent-gold); }
.badge-segmentation{ background:rgba(80,160,200,.18); color:#5ac; }
.badge-unknown{ background:rgba(128,128,128,.18); color:var(--muted); }

.section{ background:var(--panel); border:1px solid var(--border-soft); border-radius:8px; padding:1.25rem; margin-bottom:1rem; box-shadow:var(--shadow); }
.section h3{ margin:0 0 .6rem; font-size:1rem; }

.stats-grid{ display:grid; grid-template-columns:repeat(auto-fill, minmax(140px, 1fr)); gap:.4rem .8rem; font-size:.9rem; }
.stat-label{ color:var(--muted); }
.stat-value{ font-weight:600; }

.chart-area svg{ display:block; width:100%; }

.charset-display{ font-family:var(--font-mono); font-size:1.15rem; letter-spacing:.3rem; direction:rtl; line-height:1.8; }

.controls-row{ display:flex; flex-wrap:wrap; gap:.5rem; align-items:center; }
.controls-row select{ min-width:12rem; }
.script-status{ font-size:.82rem; }

.train-opts{ display:grid; grid-template-columns:1fr 1fr 1fr 1fr; gap:.4rem; margin:.6rem 0; }
.train-opts label{ font-size:.82rem; color:var(--muted); display:flex; flex-direction:column; gap:.15rem; }
.train-opts input{ width:100%; padding:.3rem .4rem; font-size:.85rem; }

.gt-info{ font-size:.88rem; color:var(--muted); margin:.4rem 0; }
.gt-toolbar{ display:flex; flex-wrap:wrap; gap:.5rem; align-items:center; margin:.6rem 0; font-size:.85rem; }
.gt-toolbar input[type=text]{ padding:.3rem .5rem; }
.gt-grid{ max-height:60vh; overflow-y:auto; border:1px solid var(--border); border-radius:6px; margin-top:.4rem; }
.gt-row{
  display:flex; align-items:center; gap:.6rem; padding:.4rem .6rem;
  border-bottom:1px solid var(--border-soft); font-size:.85rem;
}
.gt-row:hover{ background:rgba(154,120,32,.04); }
.gt-row.flagged{ background:rgba(180,60,60,.06); }
.gt-row input[type=checkbox]{ flex-shrink:0; }
.gt-row img{ max-height:36px; max-width:200px; border:1px solid var(--border); border-radius:2px; flex-shrink:0; }
.gt-row .gt-text{ direction:rtl; font-family:var(--font-mono); font-size:.82rem; flex:1; min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.gt-row .gt-dims{ color:var(--muted); font-size:.78rem; white-space:nowrap; flex-shrink:0; }
.gt-row .gt-source{ font-size:.72rem; padding:.1rem .3rem; border-radius:2px; background:rgba(128,128,128,.12); color:var(--muted); flex-shrink:0; }
.gt-flag{ color:#c84; font-size:.9rem; flex-shrink:0; title:attr(data-reason); }

.test-results table{ width:100%; border-collapse:collapse; margin-top:.5rem; }
.test-results th,.test-results td{ padding:.4rem .5rem; border-bottom:1px solid var(--border); text-align:left; vertical-align:top; }
.test-results th{ color:var(--muted); font-weight:400; font-size:.82rem; }
.test-line-img{ max-height:36px; max-width:220px; border:1px solid var(--border); border-radius:2px; }
.test-text{ direction:rtl; font-family:var(--font-mono); font-size:.85rem; }
.cer-good{ color:#5a5; } .cer-ok{ color:var(--accent-gold); } .cer-bad{ color:#b44; }
.test-summary{ font-weight:600; font-size:1rem; margin-bottom:.3rem; }

.danger-zone{ display:flex; gap:.5rem; margin-top:.5rem; }
.del-btn{ color:#b44; border-color:#b44; background:transparent; }
.del-btn:hover{ background:rgba(180,60,60,.12); }

.compare-select{ margin-bottom:.6rem; }

.rename-input{
  font-family:var(--font-display); font-size:1.4rem; font-weight:700;
  background:var(--sunken); border:1px solid var(--accent-gold); border-radius:3px;
  color:var(--fg); padding:.1rem .4rem; width:18rem;
}
</style>
</head>
<body>
<div class="detail-wrap">
  <div class="top-bar">
    <a href="/models" class="nav-btn">&larr; All Models</a>
    <span class="model-title" id="model-title" onclick="startRename(this, '{{ m.name }}')" title="Click to rename">{{ m.name }}</span>
    <span class="model-badge badge-{{ m.model_type }}">{{ m.model_type }}</span>
    <span style="margin-left:auto;">{{ THEME_TOGGLE | safe }}</span>
  </div>

  <!-- Stats -->
  <div class="section">
    <h3>Overview</h3>
    <div class="stats-grid">
      <div><span class="stat-label">Best accuracy</span><br><span class="stat-value">{% if m.best_accuracy is not none %}{{ m.best_accuracy }}%{% else %}—{% endif %}</span></div>
      <div><span class="stat-label">Epochs</span><br><span class="stat-value">{{ m.completed_epochs or '—' }}</span></div>
      <div><span class="stat-label">Size</span><br><span class="stat-value">{{ m.file_size_mb }} MB</span></div>
      <div><span class="stat-label">Modified</span><br><span class="stat-value">{{ m.last_modified }}</span></div>
      <div><span class="stat-label">Input spec</span><br><span class="stat-value" style="font-size:.82rem;">{{ m.input_spec or '—' }}</span></div>
      {% if m.charset_count %}<div><span class="stat-label">Charset</span><br><span class="stat-value">{{ m.charset_count }} chars</span></div>{% endif %}
    </div>
  </div>

  <!-- Accuracy chart -->
  {% if m.accuracy_history %}
  <div class="section">
    <h3>Accuracy Curve</h3>
    <div class="chart-area">
      <svg width="100%" height="180" viewBox="0 0 400 180" preserveAspectRatio="none">
        <polyline points="{% for h in m.accuracy_history %}{{ (loop.index0 / (m.accuracy_history|length - 1) * 392)|round(1) }},{{ (172 - (h[1] - m.accuracy_history|map(attribute=1)|min) / ((m.accuracy_history|map(attribute=1)|max - m.accuracy_history|map(attribute=1)|min) or 1) * 164)|round(1) }} {% endfor %}"
                  fill="none" stroke="var(--accent-gold)" stroke-width="1.5" transform="translate(4,4)"/>
        <text x="4" y="14" fill="var(--muted)" font-size="10">{{ "%.1f"|format(m.accuracy_history|map(attribute=1)|max) }}%</text>
        <text x="4" y="178" fill="var(--muted)" font-size="10">{{ "%.1f"|format(m.accuracy_history|map(attribute=1)|min) }}%</text>
      </svg>
    </div>

    <!-- Compare -->
    {% if other_models %}
    <div class="compare-select">
      <label style="font-size:.85rem;color:var(--muted);">Compare with:
        <select id="compare-select" onchange="drawComparison()">
          <option value="">— select model —</option>
          {% for o in other_models %}
          {% if o.accuracy_history %}
          <option value="{{ o.name }}" data-history='{{ o.accuracy_history | tojson }}'>{{ o.name }} ({{ o.best_accuracy or '—' }}%)</option>
          {% endif %}
          {% endfor %}
        </select>
      </label>
    </div>
    <div id="compare-chart" style="display:none;">
      <svg id="compare-svg" width="100%" height="180" viewBox="0 0 400 180" preserveAspectRatio="none"></svg>
    </div>
    {% endif %}
  </div>
  {% endif %}

  <!-- Charset -->
  {% if m.charset %}
  <div class="section">
    <h3>Character Set ({{ m.charset_count }})</h3>
    <div class="charset-display">{{ m.charset }}</div>
  </div>
  {% endif %}

  <!-- Script & GT -->
  <div class="section">
    <h3>Script &amp; Training Data</h3>
    <div class="controls-row">
      <select id="script-select" onchange="setScript('{{ m.name }}', this.value)">
        <option value="">— no script —</option>
        {% for s in scripts %}
        <option value="{{ s }}" {% if s == m.script %}selected{% endif %}>{{ s }}</option>
        {% endfor %}
      </select>
      <span class="script-status" id="script-status"></span>
    </div>
    {% if m.script %}
    <div class="gt-info" id="gt-summary">GT: {{ gt_counts.train }} train / {{ gt_counts.val }} val lines</div>

    <div class="gt-toolbar">
      <label><input type="checkbox" id="filter-flagged" onchange="filterGt()"> Flagged only</label>
      <select id="filter-source" onchange="filterGt()">
        <option value="all">All sources</option>
        <option value="train">Train only</option>
        <option value="val">Val only</option>
      </select>
      <input type="text" id="filter-text" placeholder="Search text…" oninput="filterGt()" style="width:10rem;">
      <button type="button" onclick="selectAllFlagged()">Select flagged</button>
      <button type="button" onclick="deleteSelected()" id="delete-sel-btn" disabled class="del-btn">Delete selected (<span id="sel-count">0</span>)</button>
    </div>

    <div class="gt-grid" id="gt-grid">
      <div style="color:var(--muted);padding:1rem;">Loading training data…</div>
    </div>
    {% else %}
    <div class="gt-info">Assign a script to see training data.</div>
    {% endif %}
  </div>

  <!-- Training (recognition only) -->
  {% if m.model_type == 'recognition' %}
  <div class="section">
    <h3>Training</h3>
    <details>
      <summary style="cursor:pointer;color:var(--muted);font-size:.9rem;">Training options</summary>
      <div class="train-opts" id="train-opts">
        <label>Epochs <input type="number" name="epochs" value="-1" title="-1 = early stopping"></label>
        <label>Learning rate <input type="number" name="lrate" value="0.001" step="0.0001"></label>
        <label>Batch size <input type="number" name="batch_size" value="1" min="1"></label>
        <label>Early stop lag <input type="number" name="lag" value="10" min="1"></label>
      </div>
    </details>
    <div class="controls-row" style="margin-top:.5rem;">
      <button type="button" onclick="startTraining('{{ m.name }}')" id="train-btn"
              {% if not m.script %}disabled title="Assign a script first"{% endif %}>
        Train
      </button>
      {% if m.has_log %}
      <button type="button" onclick="viewLog('{{ m.name }}')">View Log</button>
      {% endif %}
    </div>
  </div>

  <!-- Quick Test -->
  <div class="section">
    <h3>Quick Test</h3>
    <p style="font-size:.85rem;color:var(--muted);margin:0 0 .5rem;">Run OCR on random GT lines and compare to ground truth.</p>
    <div class="controls-row">
      <button type="button" onclick="quickTest('{{ m.name }}')" id="test-btn"
              {% if not m.script %}disabled title="Assign a script first"{% endif %}>
        Run Test (5 lines)
      </button>
    </div>
    <div class="test-results" id="test-results"></div>
  </div>
  {% endif %}

  <!-- Danger zone -->
  <div class="section">
    <h3>Manage</h3>
    <div class="danger-zone">
      <button class="del-btn" onclick="deleteModel('{{ m.name }}')">Delete Model</button>
    </div>
  </div>
</div>

<!-- Log modal -->
<div id="log-modal" style="display:none;position:fixed;inset:0;z-index:9998;background:rgba(0,0,0,.6);backdrop-filter:blur(3px);justify-content:center;align-items:center;" onclick="if(event.target===this)this.style.display='none'">
  <div style="background:var(--panel,#1a1a1a);border:1px solid var(--border,#333);border-radius:10px;padding:1.5rem;box-shadow:0 8px 32px rgba(0,0,0,.4);max-width:800px;width:90%;max-height:80vh;overflow-y:auto;">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:.75rem;">
      <h3 style="margin:0;" id="log-title">Training Log</h3>
      <button onclick="document.getElementById('log-modal').style.display='none'" style="font-size:1.2rem;background:none;border:none;color:var(--fg);cursor:pointer;">&times;</button>
    </div>
    <pre id="log-content" style="font-size:.78rem;white-space:pre-wrap;word-break:break-all;max-height:60vh;overflow-y:auto;color:var(--fg);background:var(--sunken);padding:.75rem;border-radius:6px;"></pre>
  </div>
</div>

<!-- Training overlay -->
<div id="train-overlay" style="display:none;position:fixed;inset:0;z-index:9999;background:rgba(0,0,0,.55);backdrop-filter:blur(3px);justify-content:center;align-items:center;">
  <div style="background:var(--panel,#1a1a1a);border:1px solid var(--border,#333);border-radius:10px;padding:2rem 2.5rem;text-align:center;box-shadow:0 8px 32px rgba(0,0,0,.4);max-width:500px;">
    <div style="margin-bottom:1rem;">
      <svg width="40" height="40" viewBox="0 0 40 40" style="animation:tspin 1s linear infinite;">
        <circle cx="20" cy="20" r="16" fill="none" stroke="var(--border,#555)" stroke-width="3"/>
        <circle cx="20" cy="20" r="16" fill="none" stroke="var(--accent-gold,#9a7820)" stroke-width="3" stroke-dasharray="80" stroke-dashoffset="60" stroke-linecap="round"/>
      </svg>
    </div>
    <div id="train-msg" style="font-size:1.05rem;color:var(--fg,#eee);">Training…</div>
    <div id="train-detail" style="font-size:.82rem;color:var(--muted,#888);margin-top:.5rem;word-break:break-all;max-height:3rem;overflow:hidden;"></div>
  </div>
</div>
<style>@keyframes tspin{to{transform:rotate(360deg)}}</style>

<script>
var ACCURACY = {{ accuracy_json | safe }};

function esc(s){ var d=document.createElement('div'); d.textContent=s; return d.innerHTML; }

// ---- Script ----
function setScript(modelName, script) {
  var status = document.getElementById('script-status');
  var body = new URLSearchParams();
  body.set('model_name', modelName);
  body.set('script', script);
  fetch('/set_model_script', {method:'POST', body:body})
    .then(function(r){ return r.json(); })
    .then(function(){
      status.textContent = 'saved';
      setTimeout(function(){ status.textContent = ''; }, 1500);
      var tb = document.getElementById('train-btn');
      var testb = document.getElementById('test-btn');
      if (tb) { tb.disabled = !script; }
      if (testb) { testb.disabled = !script; }
      if (script) loadGtSamples(script);
    });
}

// ---- Rename ----
function startRename(el, oldName) {
  if (el.querySelector('input')) return;
  var orig = el.textContent;
  var input = document.createElement('input');
  input.className = 'rename-input';
  input.value = orig;
  el.textContent = '';
  el.appendChild(input);
  input.focus(); input.select();
  function finish() {
    var newName = input.value.trim();
    if (!newName || newName === orig) { el.textContent = orig; return; }
    var body = new URLSearchParams();
    body.set('old_name', oldName); body.set('new_name', newName);
    fetch('/rename_model', {method:'POST', body:body})
      .then(function(r){ if(!r.ok) return r.text().then(function(t){throw new Error(t);}); return r.json(); })
      .then(function(d){ window.location.href = '/models/' + d.new_name; })
      .catch(function(e){ alert(e.message); el.textContent = orig; });
  }
  input.addEventListener('keydown', function(e){ if(e.key==='Enter')finish(); if(e.key==='Escape')el.textContent=orig; });
  input.addEventListener('blur', finish);
}

// ---- Delete ----
function deleteModel(name) {
  if (!confirm('Permanently delete model "' + name + '"?')) return;
  var body = new URLSearchParams(); body.set('model_name', name);
  fetch('/delete_model', {method:'POST', body:body})
    .then(function(r){ if(!r.ok) throw new Error('Delete failed'); return r.json(); })
    .then(function(){ window.location.href = '/models'; })
    .catch(function(e){ alert(e.message); });
}

// ---- View log ----
function viewLog(name) {
  var modal = document.getElementById('log-modal');
  var content = document.getElementById('log-content');
  document.getElementById('log-title').textContent = 'Training Log: ' + name;
  content.textContent = 'Loading…';
  modal.style.display = 'flex';
  fetch('/model_log/' + encodeURIComponent(name))
    .then(function(r){ return r.text(); })
    .then(function(t){ content.textContent = t; });
}

// ---- Training ----
function startTraining(modelName) {
  if (!confirm('Start fine-tuning ' + modelName + '?')) return;
  var overlay = document.getElementById('train-overlay');
  var msg = document.getElementById('train-msg');
  var detail = document.getElementById('train-detail');
  overlay.style.display = 'flex';
  msg.textContent = 'Starting training…'; detail.textContent = '';
  var body = new URLSearchParams(); body.set('model_name', modelName);
  document.querySelectorAll('#train-opts input').forEach(function(inp){ body.set(inp.name, inp.value); });
  fetch('/train', {method:'POST', body:body})
    .then(function(r){ if(!r.ok) return r.text().then(function(t){throw new Error(t);}); return r.json(); })
    .then(function(d){
      msg.textContent = 'Training ' + modelName + '…';
      var poll = setInterval(function(){
        fetch('/status?session=' + encodeURIComponent(d.job))
          .then(function(r){ return r.json(); })
          .then(function(s){
            detail.textContent = s.state || '';
            if (s.state === 'done' || (s.state && s.state.startsWith('error'))) {
              clearInterval(poll);
              msg.textContent = s.state === 'done' ? 'Training complete!' : 'Training failed';
              setTimeout(function(){ location.reload(); }, 1500);
            }
          });
      }, 2000);
    })
    .catch(function(e){ overlay.style.display = 'none'; alert(e.message); });
}

// ---- Quick test ----
function quickTest(modelName) {
  var btn = document.getElementById('test-btn');
  var results = document.getElementById('test-results');
  btn.disabled = true; btn.textContent = 'Testing…';
  results.innerHTML = '<span style="color:var(--muted)">Running OCR on sample lines…</span>';
  var body = new URLSearchParams(); body.set('model_name', modelName); body.set('num_lines', '5');
  fetch('/quick_test', {method:'POST', body:body})
    .then(function(r){ if(!r.ok) return r.json().then(function(d){throw new Error(d.error);}); return r.json(); })
    .then(function(d){
      var cc = d.overall_cer < 5 ? 'cer-good' : d.overall_cer < 15 ? 'cer-ok' : 'cer-bad';
      var html = '<div class="test-summary">Overall CER: <span class="' + cc + '">' + d.overall_cer + '%</span></div>';
      html += '<table><tr><th>Line</th><th>Ground Truth</th><th>Prediction</th><th>CER</th></tr>';
      d.results.forEach(function(r){
        var rc = r.cer < 5 ? 'cer-good' : r.cer < 15 ? 'cer-ok' : 'cer-bad';
        html += '<tr><td><img class="test-line-img" src="/gt_thumb?path=' + encodeURIComponent(r.image) + '&w=220"></td>';
        html += '<td class="test-text">' + esc(r.ground_truth) + '</td>';
        html += '<td class="test-text">' + esc(r.prediction) + '</td>';
        html += '<td class="' + rc + '">' + r.cer + '%</td></tr>';
      });
      results.innerHTML = html + '</table>';
    })
    .catch(function(e){ results.innerHTML = '<span style="color:#b44">' + e.message + '</span>'; })
    .finally(function(){ btn.disabled = false; btn.textContent = 'Run Test (5 lines)'; });
}

// ---- Compare chart ----
function drawComparison() {
  var sel = document.getElementById('compare-select');
  var chart = document.getElementById('compare-chart');
  var svg = document.getElementById('compare-svg');
  var opt = sel.options[sel.selectedIndex];
  if (!opt.value) { chart.style.display = 'none'; return; }
  var otherHistory = JSON.parse(opt.dataset.history);
  chart.style.display = 'block';
  svg.innerHTML = drawCurve(ACCURACY, 'var(--accent-gold,#9a7820)', 396, 172)
                + drawCurve(otherHistory, '#5ac', 396, 172)
                + '<text x="4" y="14" fill="var(--accent-gold)" font-size="10">{{ m.name }}</text>'
                + '<text x="4" y="26" fill="#5ac" font-size="10">' + opt.value + '</text>';
}

function drawCurve(history, color, w, h) {
  if (!history.length) return '';
  var accs = history.map(function(p){ return p[1]; });
  var allAccs = ACCURACY.map(function(p){return p[1];}).concat(accs);
  var mn = Math.min.apply(null, allAccs), mx = Math.max.apply(null, allAccs);
  var rng = mx - mn || 1;
  var pts = history.map(function(p, i){
    return ((i / Math.max(history.length-1,1) * w) + 2).toFixed(1) + ',' +
           ((h - (p[1]-mn)/rng * h) + 4).toFixed(1);
  }).join(' ');
  return '<polyline points="' + pts + '" fill="none" stroke="' + color + '" stroke-width="1.5"/>';
}

// ---- GT data browser ----
var gtData = [];

function loadGtBrowser(script) {
  var grid = document.getElementById('gt-grid');
  if (!grid) return;
  grid.innerHTML = '<div style="color:var(--muted);padding:1rem;">Loading…</div>';
  fetch('/gt_browse/' + encodeURIComponent(script))
    .then(function(r){ return r.json(); })
    .then(function(d){
      gtData = d.lines;
      var summary = document.getElementById('gt-summary');
      if (summary) {
        var train = gtData.filter(function(l){return l.source==='train';}).length;
        var val = gtData.filter(function(l){return l.source==='val';}).length;
        var flagged = gtData.filter(function(l){return l.flagged;}).length;
        summary.textContent = 'GT: ' + train + ' train / ' + val + ' val lines' + (flagged ? ' (' + flagged + ' flagged)' : '');
      }
      filterGt();
    });
}

function filterGt() {
  var grid = document.getElementById('gt-grid');
  if (!grid) return;
  var flaggedOnly = document.getElementById('filter-flagged') && document.getElementById('filter-flagged').checked;
  var source = document.getElementById('filter-source') ? document.getElementById('filter-source').value : 'all';
  var search = (document.getElementById('filter-text') ? document.getElementById('filter-text').value : '').toLowerCase();
  grid.innerHTML = '';
  var shown = 0;
  gtData.forEach(function(line, idx){
    if (flaggedOnly && !line.flagged) return;
    if (source !== 'all' && line.source !== source) return;
    if (search && line.gt_text.toLowerCase().indexOf(search) === -1 && line.name.toLowerCase().indexOf(search) === -1) return;
    shown++;
    var row = document.createElement('div');
    row.className = 'gt-row' + (line.flagged ? ' flagged' : '');
    row.dataset.idx = idx;
    row.innerHTML =
      '<input type="checkbox" class="gt-check" data-path="' + esc(line.path) + '" onchange="updateSelCount()">' +
      (line.flagged ? '<span class="gt-flag" title="Suspicious: check dimensions">⚠</span>' : '') +
      '<img loading="lazy" src="/gt_thumb?path=' + encodeURIComponent(line.path) + '&w=200">' +
      '<span class="gt-text" title="' + esc(line.gt_text) + '">' + esc(line.gt_text || '(empty)') + '</span>' +
      '<span class="gt-dims">' + line.width + '×' + line.height + '</span>' +
      '<span class="gt-source">' + line.source + '</span>';
    grid.appendChild(row);
  });
  if (!shown) grid.innerHTML = '<div style="color:var(--muted);padding:1rem;">No lines match filters.</div>';
  updateSelCount();
}

function updateSelCount() {
  var checks = document.querySelectorAll('.gt-check:checked');
  var countEl = document.getElementById('sel-count');
  var btn = document.getElementById('delete-sel-btn');
  if (countEl) countEl.textContent = checks.length;
  if (btn) btn.disabled = checks.length === 0;
}

function selectAllFlagged() {
  document.querySelectorAll('.gt-row.flagged .gt-check').forEach(function(c){ c.checked = true; });
  updateSelCount();
}

function deleteSelected() {
  var checks = document.querySelectorAll('.gt-check:checked');
  if (!checks.length) return;
  if (!confirm('Delete ' + checks.length + ' GT line(s)? This cannot be undone.')) return;
  var paths = [];
  checks.forEach(function(c){ paths.push(c.dataset.path); });
  fetch('/gt_delete', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({paths:paths})})
    .then(function(r){ return r.json(); })
    .then(function(d){
      alert('Deleted ' + d.deleted + ' line(s).');
      var pathSet = new Set(paths);
      gtData = gtData.filter(function(l){ return !pathSet.has(l.path); });
      filterGt();
      var summary = document.getElementById('gt-summary');
      if (summary) {
        var train = gtData.filter(function(l){return l.source==='train';}).length;
        var val = gtData.filter(function(l){return l.source==='val';}).length;
        var flagged = gtData.filter(function(l){return l.flagged;}).length;
        summary.textContent = 'GT: ' + train + ' train / ' + val + ' val lines' + (flagged ? ' (' + flagged + ' flagged)' : '');
      }
    });
}

{% if m.script %}
document.addEventListener('DOMContentLoaded', function(){ loadGtBrowser('{{ m.script }}'); });
{% endif %}
</script>
</body>
</html>
""")
