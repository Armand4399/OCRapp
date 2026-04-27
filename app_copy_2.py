import os
import re
import json
import shutil
import tempfile
import zipfile
import threading
import subprocess
import uuid
import time
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

# --- Optional PIL for image ops ---
try:
    from PIL import Image
except ImportError:
    print("Warning: Pillow not found. Image operations may fail.")
    Image = None

# --- Optional cv2/numpy for deskew ---
try:
    import numpy as np
    import cv2
except ImportError:
    print("Warning: opencv-python / numpy not found. Deskew will be unavailable.")
    np = None
    cv2 = None

from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import (
    HTMLResponse, FileResponse, RedirectResponse, PlainTextResponse, JSONResponse
)
from jinja2 import Environment
from pydantic import BaseModel
from datetime import datetime
import csv

app = FastAPI(title="Hebrew OCR Viewer + Training (Kraken 6)")

from fastapi.middleware.gzip import GZipMiddleware
app.add_middleware(GZipMiddleware, minimum_size=500)

# ---------------- Config ----------------
APP_ROOT = Path(__file__).resolve().parent
MODELS_DIR = APP_ROOT / "models"
DEFAULT_MODEL = str(MODELS_DIR / "BiblIA_01_ft_best.mlmodel")
DEFAULT_DPI = 300
PROGRESS: Dict[str, Dict[str, Any]] = {}

# --- Device selection (MPS = Apple GPU, falls back to CPU) ---
import torch as _torch
DEVICE = "mps" if _torch.backends.mps.is_available() else "cpu"

# --- Permanent Data Locations ---
SESSIONS_BASE_DIR = APP_ROOT / "sessions"
FINAL_GT_DIR = APP_ROOT / "training" / "val"
DOWNLOADS_DIR = Path.home() / "Downloads"  # CSV export target

# Ensure base directories exist on startup
SESSIONS_BASE_DIR.mkdir(parents=True, exist_ok=True)
FINAL_GT_DIR.mkdir(parents=True, exist_ok=True)
DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- Kraken Python API ----------------
from kraken import binarization as _kr_binarize, blla as _kr_blla, rpred as _kr_rpred
from kraken.lib.vgsl import TorchVGSLModel as _TorchVGSLModel
from kraken.lib.models import load_any as _load_any

# Model cache (load once, reuse across requests)
_seg_model_cache: Dict[str, Any] = {}
_rec_model_cache: Dict[str, Any] = {}

DEFAULT_SEG_MODEL = str(MODELS_DIR / "blla.mlmodel")

def _get_seg_model(path: str = None) -> Any:
    path = path or DEFAULT_SEG_MODEL
    if path not in _seg_model_cache:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = _TorchVGSLModel.load_model(path)
            m.to(DEVICE)
            _seg_model_cache[path] = m
    return _seg_model_cache[path]

def _get_rec_model(path: str) -> Any:
    if path not in _rec_model_cache:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _rec_model_cache[path] = _load_any(path, device=DEVICE)
    return _rec_model_cache[path]

def _seg_to_json(seg, imagename: str = "") -> dict:
    """Convert a kraken Segmentation object to the JSON dict format used by this app."""
    lines = []
    for i, line in enumerate(seg.lines):
        entry: Dict[str, Any] = {"id": f"line_{i+1:04d}"}
        if line.boundary:
            entry["boundary"] = [[float(x), float(y)] for x, y in line.boundary]
        if line.baseline:
            entry["baseline"] = [[float(x), float(y)] for x, y in line.baseline]
        lines.append(entry)
    regions = {}
    if seg.regions:
        for rtype, rlist in seg.regions.items():
            regions[rtype] = [{"boundary": [[float(x), float(y)] for x, y in r.boundary]} for r in rlist]
    return {
        "type": seg.type,
        "imagename": imagename,
        "text_direction": seg.text_direction,
        "script_detection": seg.script_detection,
        "lines": lines,
        "regions": regions,
        "line_orders": seg.line_orders if seg.line_orders else [],
        "language": getattr(seg, "language", None),
    }

def kraken_segment(img_path: Path, text_direction: str = "horizontal-rl") -> dict:
    """Segment an image, return JSON-serializable dict."""
    im = Image.open(img_path)
    seg_model = _get_seg_model()
    seg = _kr_blla.segment(im, text_direction=text_direction, model=seg_model, device=DEVICE)
    return _seg_to_json(seg, str(img_path))

def kraken_segment_and_ocr(img_path: Path, model_path: str,
                           text_direction: str = "horizontal-rl",
                           bidi_reordering: bool = True) -> Tuple[dict, List[str]]:
    """Segment + OCR an image. Returns (seg_dict, text_lines)."""
    im = Image.open(img_path)
    seg_model = _get_seg_model()
    rec_model = _get_rec_model(model_path)
    seg = _kr_blla.segment(im, text_direction=text_direction, model=seg_model, device=DEVICE)
    text_lines = []
    for record in _kr_rpred.rpred(rec_model, im, seg, bidi_reordering=bidi_reordering):
        text_lines.append(record.prediction)
    return _seg_to_json(seg, str(img_path)), text_lines

def kraken_ocr_lines(img_path: Path, model_path: str, seg_dict: dict,
                     bidi_reordering: bool = True) -> List[str]:
    """Run OCR on an image using existing segmentation JSON. Returns text lines."""
    from kraken.containers import Segmentation, BaselineLine
    im = Image.open(img_path)
    rec_model = _get_rec_model(model_path)
    # Rebuild Segmentation object from dict
    lines = []
    for line_d in seg_dict.get("lines", []):
        bl = line_d.get("baseline", [])
        bnd = line_d.get("boundary", [])
        if bl:
            lines.append(BaselineLine(
                id=line_d.get("id", ""),
                baseline=bl,
                boundary=bnd if bnd else None,
                tags={"type": "default"},
            ))
    seg = Segmentation(
        type="baselines",
        imagename=str(img_path),
        text_direction=seg_dict.get("text_direction", "horizontal-rl"),
        lines=lines,
        script_detection=False,
        regions={},
        line_orders=[],
    )
    text_lines = []
    for record in _kr_rpred.rpred(rec_model, im, seg, bidi_reordering=bidi_reordering):
        text_lines.append(record.prediction)
    return text_lines

def kraken_binarize(img_path: Path, out_path: Path):
    """Binarize an image using kraken's nlbin."""
    im = Image.open(img_path)
    result = _kr_binarize.nlbin(im)
    result.save(out_path)


# ---------------- Helpers ----------------
def list_models() -> List[str]:
    if MODELS_DIR.exists():
        return sorted(str(p) for p in MODELS_DIR.glob("*.mlmodel"))
    return []

def list_script_dirs() -> List[str]:
    scripts: set[str] = set()
    if FINAL_GT_DIR.exists():
        scripts.update(d.name for d in FINAL_GT_DIR.iterdir() if d.is_dir())
    if SESSIONS_BASE_DIR.exists():
        for sf in SESSIONS_BASE_DIR.glob("*/script.txt"):
            name = sf.read_text(encoding="utf-8").strip()
            if name and _SCRIPT_NAME_RE.match(name):
                scripts.add(name)
    return sorted(scripts)

_SCRIPT_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")

def sanitize_script_name(raw: str) -> str:
    s = (raw or "").strip()
    return s if _SCRIPT_NAME_RE.match(s) else ""

def run(cmd: List[str], cwd: Optional[str] = None) -> str:
    proc = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True)
    return proc.stdout

def is_pdf(path: str) -> bool:
    return path.lower().endswith(".pdf")

def page_number(path: Path) -> int:
    try:
        # supports stems like "page-12"
        return int(path.stem.split("-")[-1])
    except Exception:
        return 10**9

def _bbox(points: List[Tuple[float, float]], pad: int = 3) -> Tuple[int, int, int, int]:
    xs = [p[0] for p in points]; ys = [p[1] for p in points]
    return int(min(xs))-pad, int(min(ys))-pad, int(max(xs))+pad, int(max(ys))+pad

def _find_lines(seg: dict) -> List[dict]:
    lines: List[dict] = []
    def has_line_feats(d: dict) -> bool:
        return any(k in d for k in ("baseline","boundary","polygon","coords"))
    def walk(node):
        if isinstance(node, dict):
            if has_line_feats(node): lines.append(node)
            for v in node.values(): walk(v)
        elif isinstance(node, list):
            for v in node: walk(v)
    walk(seg)
    return lines

def _poly_from_line(line: dict) -> Optional[List[Tuple[float, float]]]:
    for key in ("boundary","polygon","coords"):
        v = line.get(key)
        if isinstance(v, list) and v and isinstance(v[0], (list, tuple)):
            return [(float(x), float(y)) for x,y in v]
    return None

def _build_sidebar_cache(sess: Path, bin_dir: Path, pages_dir: Path, image_exts: tuple) -> list:
    """Return sidebar page list, using a cached JSON file when possible."""
    cache_path = sess / "sidebar_cache.json"
    # Check if cache is fresh
    if cache_path.exists():
        cache_mtime = cache_path.stat().st_mtime
        json_files = list(pages_dir.glob("page-*.json"))
        newest_json = max((f.stat().st_mtime for f in json_files), default=0)
        if cache_mtime >= newest_json:
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass  # fall through to rebuild

    # Build sidebar data
    all_pages = []
    all_page_files = sorted(bin_dir.glob("page-*.png"), key=page_number)
    for bin_file in all_page_files:
        page_stem = bin_file.stem
        orig = next((x for x in pages_dir.iterdir() if x.stem == page_stem and x.suffix.lower() in image_exts), None)
        orig_name_to_copy = "not_found.png"
        if orig:
            orig_name_to_copy = f"{page_stem}_orig{orig.suffix.lower()}"
            if not (sess / orig_name_to_copy).exists():
                try:
                    shutil.copy(orig, sess / orig_name_to_copy)
                except Exception:
                    orig_name_to_copy = "not_found.png"
        line_count = 0
        seg_json_path_sidebar = pages_dir / f"{page_stem}.json"
        if seg_json_path_sidebar.exists():
            try:
                with open(seg_json_path_sidebar, "r", encoding="utf-8") as f:
                    seg = json.load(f)
                line_count = len(_find_lines(seg))
            except Exception:
                pass
        all_pages.append({
            "title": page_stem.replace("page-", ""),
            "orig_image_name": orig_name_to_copy,
            "line_count": line_count,
        })

    # Write cache
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(all_pages, f)
    except Exception:
        pass
    return all_pages

def extract_lines_to_gt(bin_img: Path, seg_json: Path, out_dir: Path) -> int:
    if Image is None: return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(seg_json, "r", encoding="utf-8") as fh: seg = json.load(fh)
    # Prefer authoritative top-level lines
    if isinstance(seg.get("lines"), list):
        lines = [l for l in seg["lines"] if isinstance(l, dict)]
    else:
        # Fallback for older sessions
        lines = _find_lines(seg)
    im = Image.open(bin_img); W,H = im.size
    count = 0
    for i, line in enumerate(lines, 1):
        poly = _poly_from_line(line)
        if not poly: continue
        x0,y0,x1,y1 = _bbox(poly, pad=3)
        x0=max(0,min(x0,W-1)); y0=max(0,min(y0,H-1)); x1=max(x0+1,min(x1,W)); y1=max(y0+1,min(y1,H))
        if (x1-x0)<3 or (y1-y0)<3: continue
        out_png = out_dir / f"{bin_img.stem}_{i:04d}.png"
        try: im.crop((x0,y0,x1,y1)).save(out_png)
        except Exception: continue
        gt = out_png.with_suffix(".gt.txt")
        if not gt.exists(): gt.write_text("", encoding="utf-8")
        count += 1
    im.close()
    return count

def _baseline_from_poly(poly: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not poly: return []
    min_x=min(p[0] for p in poly); max_x=max(p[0] for p in poly)
    ys_at_min_x=sorted([p[1] for p in poly if abs(p[0]-min_x)<1])
    ys_at_max_x=sorted([p[1] for p in poly if abs(p[0]-max_x)<1])
    if not ys_at_min_x or not ys_at_max_x: return []
    start_y=(ys_at_min_x[0]+ys_at_min_x[-1])/2; end_y=(ys_at_max_x[0]+ys_at_max_x[-1])/2
    return [(min_x,start_y), (max_x,end_y)]

def _deskew_image(img_path: Path) -> None:
    """Detect and correct skew using projection-profile variance."""
    if cv2 is None or np is None:
        return
    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return
    # Convert to grayscale for angle detection
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    # Downsample 4x for speed
    h, w = gray.shape
    small = cv2.resize(gray, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    _, bw = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Search angles -5 to +5 in 0.5-degree steps
    best_angle = 0.0
    best_var = -1.0
    (sh, sw) = bw.shape
    center = (sw // 2, sh // 2)
    for a10 in range(-50, 51, 5):  # tenths of degrees
        angle = a10 / 10.0
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(bw, M, (sw, sh), flags=cv2.INTER_NEAREST, borderValue=0)
        profile = np.sum(rotated, axis=1, dtype=np.float64)
        v = np.var(profile)
        if v > best_var:
            best_var = v
            best_angle = angle
    # Skip if negligible
    if abs(best_angle) < 0.2:
        return
    # Rotate full-res image
    (fh, fw) = img.shape[:2]
    fc = (fw // 2, fh // 2)
    M = cv2.getRotationMatrix2D(fc, best_angle, 1.0)
    fill = 255 if np.mean(gray) > 128 else 0
    if len(img.shape) == 3:
        border = (fill, fill, fill)
    else:
        border = fill
    result = cv2.warpAffine(img, M, (fw, fh), flags=cv2.INTER_LINEAR, borderValue=border)
    cv2.imwrite(str(img_path), result)

# --------------- Close-up fallback helpers ---------------

def _seg_needs_retry(seg: dict, img_width: int) -> bool:
    """Return True if segmentation looks like a failed close-up detection."""
    lines = _find_lines(seg)
    if not lines:
        return True
    baselines = [l.get("baseline") for l in lines if isinstance(l.get("baseline"), list) and len(l["baseline"]) >= 2]
    if not baselines:
        return True
    import math
    def bl_len(bl):
        return math.hypot(bl[-1][0] - bl[0][0], bl[-1][1] - bl[0][1])
    avg_len = sum(bl_len(bl) for bl in baselines) / len(baselines)
    return avg_len < img_width * 0.30

def _pad_image_for_segmentation(img_path: Path, multiplier: float) -> Tuple[Path, int]:
    """Pad image with white borders top/bottom. Returns (padded_path, pad_top)."""
    im = Image.open(img_path)
    w, h = im.size
    new_h = int(h * multiplier)
    pad_top = (new_h - h) // 2
    mode = im.mode if im.mode in ("L", "RGB", "RGBA") else "L"
    fill = 255 if mode == "L" else (255,) * len(mode)
    canvas = Image.new(mode, (w, new_h), fill)
    canvas.paste(im, (0, pad_top))
    padded_path = img_path.parent / f"{img_path.stem}_padded.png"
    canvas.save(padded_path)
    return padded_path, pad_top

def _adjust_seg_json_for_padding(seg_path: Path, pad_top: int, original_img_path: str) -> None:
    """Shift all Y coordinates in segmentation JSON to undo padding offset."""
    with open(seg_path, "r", encoding="utf-8") as f:
        seg = json.load(f)
    def shift_points(pts):
        if not isinstance(pts, list):
            return
        for pt in pts:
            if isinstance(pt, list) and len(pt) >= 2:
                pt[1] = max(0, pt[1] - pad_top)
    for line in (seg.get("lines") or []):
        if isinstance(line, dict):
            for key in ("baseline", "boundary", "polygon", "coords"):
                if key in line:
                    shift_points(line[key])
    for reg_list in (seg.get("regions") or {}).values():
        if isinstance(reg_list, list):
            for region in reg_list:
                if isinstance(region, dict) and "boundary" in region:
                    shift_points(region["boundary"])
    seg["imagename"] = original_img_path
    with open(seg_path, "w", encoding="utf-8") as f:
        json.dump(seg, f, indent=2)

def _clip_seg_polygons(seg_json_path: Path, img_height: int) -> None:
    """Clip polygon boundaries that extend unreasonably far below their baseline.

    Kraken often stretches the last line's polygon to the image bottom.
    This trims any boundary whose bottom edge exceeds 1.5× the median
    line height below the baseline midpoint.
    """
    with open(seg_json_path, "r", encoding="utf-8") as f:
        seg = json.load(f)
    lines = seg.get("lines") if isinstance(seg.get("lines"), list) else _find_lines(seg)
    if len(lines) < 2:
        return  # not enough lines to compute a meaningful median

    # Collect per-line heights from boundary polygons
    heights = []
    for line in lines:
        bnd = None
        for key in ("boundary", "polygon", "coords"):
            v = line.get(key)
            if isinstance(v, list) and v and isinstance(v[0], (list, tuple)):
                bnd = v
                break
        if bnd:
            ys = [pt[1] for pt in bnd]
            h = max(ys) - min(ys)
            if h > 0:
                heights.append(h)
    if not heights:
        return
    heights.sort()
    median_h = heights[len(heights) // 2]
    max_extent = median_h * 1.5

    changed = False
    for line in lines:
        # Get baseline Y midpoint
        bl = line.get("baseline")
        if isinstance(bl, list) and bl and isinstance(bl[0], (list, tuple)):
            bl_ys = [pt[1] for pt in bl]
            bl_mid_y = sum(bl_ys) / len(bl_ys)
        else:
            continue

        # Check and clip boundary
        for key in ("boundary", "polygon", "coords"):
            bnd = line.get(key)
            if not (isinstance(bnd, list) and bnd and isinstance(bnd[0], (list, tuple))):
                continue
            bottom = max(pt[1] for pt in bnd)
            clip_y = bl_mid_y + max_extent
            if bottom > clip_y:
                for pt in bnd:
                    if pt[1] > clip_y:
                        pt[1] = min(clip_y, img_height)
                changed = True

    if changed:
        if isinstance(seg.get("lines"), list):
            seg["lines"] = lines
        with open(seg_json_path, "w", encoding="utf-8") as f:
            json.dump(seg, f, indent=2)



def _merge_split_lines(lines: List[dict]) -> Tuple[List[dict], List[List[int]]]:
    """Merge line fragments that share the same horizontal band (y-center).

    Groups lines whose vertical midpoints are within 40% of the median line
    height. Fragments of a split line sit at roughly the same y-level, while
    genuinely separate lines are spaced ~1 full line height apart.

    Returns (merged_lines, groups) where groups[i] is a list of original
    line indices that were merged into merged_lines[i].
    """
    if not lines or len(lines) < 3:
        return lines, [[i] for i in range(len(lines))]
    # Compute y_mid and bbox for each line, keeping original index
    entries = []  # (y_mid, orig_idx, line_dict)
    no_poly = []  # (orig_idx, line_dict)
    heights = []
    for orig_idx, line in enumerate(lines):
        poly = _poly_from_line(line)
        if not poly:
            no_poly.append((orig_idx, line))
            continue
        x0, y0, x1, y1 = _bbox(poly, pad=0)
        h = y1 - y0
        y_mid = (y0 + y1) / 2.0
        entries.append((y_mid, orig_idx, line))
        if h > 0:
            heights.append(h)
    if not heights or not entries:
        return lines, [[i] for i in range(len(lines))]
    entries.sort(key=lambda e: e[0])
    heights.sort()
    median_h = heights[len(heights) // 2]
    band_tol = 0.4 * median_h
    # Group into horizontal bands by y_mid proximity
    bands: List[List[int]] = []  # each band is indices into entries
    used = [False] * len(entries)
    for i in range(len(entries)):
        if used[i]:
            continue
        band = [i]
        used[i] = True
        anchor_y = entries[i][0]
        for j in range(i + 1, len(entries)):
            if used[j]:
                continue
            if abs(entries[j][0] - anchor_y) < band_tol:
                band.append(j)
                used[j] = True
            else:
                break
        bands.append(band)
    # Merge each band into a single line
    merged = []
    groups = []  # parallel to merged: which original indices went into each
    for band in bands:
        orig_indices = [entries[idx][1] for idx in band]
        if len(band) == 1:
            merged.append(entries[band[0]][2])
            groups.append(orig_indices)
            continue
        all_pts = []
        base_line = entries[band[0]][2]
        for idx in band:
            poly = _poly_from_line(entries[idx][2])
            if poly:
                all_pts.extend(poly)
        if not all_pts:
            for idx in band:
                merged.append(entries[idx][2])
                groups.append([entries[idx][1]])
            continue
        nx0 = min(p[0] for p in all_pts)
        ny0 = min(p[1] for p in all_pts)
        nx1 = max(p[0] for p in all_pts)
        ny1 = max(p[1] for p in all_pts)
        new_boundary = [(nx0, ny0), (nx1, ny0), (nx1, ny1), (nx0, ny1)]
        new_baseline = _baseline_from_poly(new_boundary)
        merged_line = dict(base_line)
        merged_line["boundary"] = new_boundary
        merged_line["baseline"] = new_baseline
        if "polygon" in merged_line:
            merged_line["polygon"] = new_boundary
        merged.append(merged_line)
        groups.append(orig_indices)
    # Append lines without polygons
    for orig_idx, line in no_poly:
        merged.append(line)
        groups.append([orig_idx])
    return merged, groups

# ---------------- Shared UI (theme + layout) ----------------

BASE_CSS = """
/* ============================================
   THEME VARIABLES — Sana'a editorial theme
   ============================================ */
:root,[data-theme=dark]{
  --bg:#282828;--fg:#e8e0d0;--muted:#9a9080;--ghost:#686058;
  --panel:#303030;--panel-glass:rgba(48,48,48,.82);--sunken:#343434;--overlay:#3a3a3a;
  --border:#4a4840;--border-soft:#424038;
  --btn:#343434;--btn-hover:#3c3c3c;
  --accent:#c89a40;--accent-gold:#d4a830;--accent-hover:#e0b840;
  --red:#c05050;--blue:#6090c8;--purple:#9060c0;
  --shadow:0 1px 2px rgba(0,0,0,.20),0 2px 6px rgba(0,0,0,.15);
  --shadow-strong:0 2px 4px rgba(0,0,0,.25),0 8px 24px rgba(0,0,0,.30);
  --shadow-lifted:0 4px 12px rgba(0,0,0,.30),0 1px 3px rgba(0,0,0,.20);
  --sidebar-w:260px;--sidebar-collapsed-w:56px;
  --font-display:'Spectral SC','Georgia',serif;
  --font-body:'EB Garamond','Georgia',serif;
  --font-arabic:'Noto Naskh Arabic',serif;
  --font-mono:'JetBrains Mono','Courier New',monospace;
}
[data-theme=light]{
  --bg:#f5f0e8;--fg:#1a1409;--muted:#7a6a50;--ghost:#b0a080;
  --panel:#faf7f0;--panel-glass:rgba(250,247,240,.82);--sunken:#ede5d0;--overlay:#e8e0cc;
  --border:#c8b890;--border-soft:#ddd0b0;
  --btn:#faf7f0;--btn-hover:#ede5d0;
  --accent:#6b4c10;--accent-gold:#9a7820;--accent-hover:#8a6318;
  --red:#8b2020;--blue:#1a3d6e;--purple:#4a2060;
  --shadow:0 1px 2px rgba(100,80,20,.06),0 2px 6px rgba(100,80,20,.08);
  --shadow-strong:0 2px 4px rgba(100,80,20,.10),0 8px 24px rgba(100,80,20,.14);
  --shadow-lifted:0 4px 12px rgba(100,80,20,.12),0 1px 3px rgba(100,80,20,.08);
}

/* === Reset & Base === */
*,*::before,*::after{box-sizing:border-box;}
body{background:var(--bg);color:var(--fg);font-family:var(--font-body);font-size:16px;line-height:1.7;margin:0;padding:0;transition:background .2s,color .2s;}

/* === Headings === */
h1,h2,h3{font-family:var(--font-display);font-weight:500;color:var(--fg);letter-spacing:.06em;}
h1{font-size:1.6rem;}
h2{font-size:1.2rem;}

/* === Simple-page header (Index / Progress) === */
.app-header{
  position:sticky;top:0;z-index:50;
  background:var(--panel-glass);backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);
  border-bottom:1px solid var(--border-soft);
  padding:.5rem 1.25rem;display:flex;align-items:center;justify-content:flex-end;gap:.5rem;
}
.page-centered{max-width:900px;margin:2rem auto;padding:0 1.25rem;}

/* === App shell (sidebar + main grid) === */
.app{display:grid;grid-template-columns:var(--sidebar-w) 1fr;transition:grid-template-columns .25s ease;min-height:100vh;}
.app.collapsed{grid-template-columns:var(--sidebar-collapsed-w) 1fr;}
.sidebar{position:sticky;top:0;height:100vh;overflow-y:auto;border-right:1px solid var(--border);background:var(--panel);box-shadow:1px 0 8px var(--border-soft);}
.sidebar::after{content:'';position:absolute;top:0;right:0;width:2px;height:100%;background:linear-gradient(180deg,var(--accent-gold) 0%,transparent 60%);opacity:.4;pointer-events:none;}
.sidebar-header{position:sticky;top:0;z-index:2;background:var(--panel-glass);backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px);border-bottom:1px solid var(--border);display:flex;align-items:center;gap:.5rem;padding:.75rem;}
.sidebar-title{font-weight:600;color:var(--fg);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-family:var(--font-display);font-size:.85rem;letter-spacing:.05em;}
.sidebar-btn{background:var(--btn);color:var(--fg);border:1px solid var(--border);border-radius:4px;padding:.4rem .6rem;cursor:pointer;font-size:.9rem;box-shadow:var(--shadow);}
.sidebar-btn:hover{background:var(--btn-hover);}
.thumbs{padding:.5rem;display:flex;flex-direction:column;gap:.5rem;}
.thumb-item{display:flex;align-items:center;gap:.6rem;padding:.45rem .5rem;border:1px solid var(--border-soft);border-radius:4px;background:var(--bg);transition:border-color .15s,background .15s,box-shadow .15s,transform .15s;}
.thumb-item:hover{border-color:var(--accent-gold);background:var(--btn-hover);box-shadow:var(--shadow);transform:translateY(-1px);}
.thumb-item.active{border-color:var(--accent);background:var(--btn);box-shadow:inset 2px 0 0 var(--accent-gold);}
.thumb-img{width:42px;height:42px;object-fit:cover;border-radius:4px;border:1px solid var(--border);background:var(--panel);}
.thumb-meta{display:flex;flex-direction:column;min-width:0;}
.thumb-title{font-size:.95rem;color:var(--fg);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.thumb-sub{font-size:.8rem;color:var(--muted);}
.thumb-link{all:unset;cursor:pointer;display:block;}
.app.collapsed .sidebar-title,.app.collapsed .thumb-meta,.app.collapsed .sidebar-btn .label{display:none;}
.app.collapsed .sidebar-btn{padding:.4rem;}
/* Sticky toolbar inside .main — negative margins cancel parent padding */
.main{padding:0 1.5rem 2rem;}
.top{
  position:sticky;top:0;z-index:50;
  background:var(--panel-glass);backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);
  border-bottom:1px solid var(--border-soft);
  display:flex;gap:.75rem;align-items:center;flex-wrap:wrap;
  margin:0 -1.5rem 1rem;padding:.75rem 1.5rem;
}

/* === Shared components === */
button{padding:.5rem 1rem;border:1px solid var(--border);border-radius:4px;background:var(--btn);color:var(--fg);cursor:pointer;box-shadow:var(--shadow);font-family:var(--font-display);font-size:.8rem;letter-spacing:.07em;transition:background .15s,border-color .15s,box-shadow .15s,transform .1s;}
button:hover{background:var(--btn-hover);border-color:var(--accent-gold);box-shadow:var(--shadow-lifted);}
button:active{transform:translateY(1px);box-shadow:var(--shadow);}
button[type=submit]{background:var(--accent);color:var(--bg);border-color:var(--accent);}
button[type=submit]:hover{background:var(--accent-hover);border-color:var(--accent-hover);}
input[type=text],input[type=number],input[type=file],select{padding:.45rem .75rem;border:1px solid var(--border);border-radius:4px;background:var(--sunken);color:var(--fg);font-size:1rem;font-family:var(--font-body);transition:border-color .15s,box-shadow .15s;}
input[type=text]:focus,input[type=number]:focus,select:focus{outline:none;border-color:var(--accent-gold);box-shadow:0 0 0 2px rgba(154,120,32,.18);}
textarea{padding:.6rem;line-height:1.5;border:1px solid var(--border);border-radius:4px;background:var(--sunken);color:var(--fg);font-family:var(--font-mono);font-size:14px;white-space:pre-wrap;resize:none;transition:border-color .15s,box-shadow .15s;}
textarea:focus{outline:none;border-color:var(--accent-gold);box-shadow:0 0 0 2px rgba(154,120,32,.18);}
input[type=range]{accent-color:var(--accent);}
label{display:block;margin:.5rem 0 .25rem;font-weight:500;font-family:var(--font-display);font-size:.85rem;letter-spacing:.04em;}
hr{border:0;border-top:1px solid var(--border-soft);margin:2rem 0;}
.top a{padding:.5rem 1rem;border:1px solid var(--border);border-radius:4px;background:var(--btn);color:var(--fg);text-decoration:none;box-shadow:var(--shadow);font-family:var(--font-display);font-size:.8rem;letter-spacing:.07em;transition:background .15s;}
.top a:hover{background:var(--btn-hover);border-color:var(--accent-gold);}
.small{color:var(--muted);font-size:.9rem;}
.hint{color:var(--muted);font-size:.9rem;margin-top:.25rem;}
.coltitle{font-weight:500;margin:0 0 .5rem;font-family:var(--font-display);font-size:.85rem;letter-spacing:.06em;}
.panel{border:1px solid var(--border);border-radius:6px;padding:1rem;background:var(--panel);}
.toast{position:fixed;bottom:16px;left:50%;transform:translateX(-50%);background:var(--panel-glass);backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);border:1px solid var(--border);color:var(--fg);border-radius:6px;padding:.6rem 1.2rem;font-size:.9rem;display:flex;gap:.8rem;align-items:center;z-index:9999;box-shadow:var(--shadow-strong);}
.toast button{background:none;border:1px solid var(--border);border-radius:4px;color:var(--muted);padding:.2rem .6rem;cursor:pointer;font-size:.85rem;box-shadow:none;}
.toast button:hover{color:var(--fg);}

/* === Section label (small caps) === */
.section-label{font-family:var(--font-display);font-size:.72rem;letter-spacing:.1em;color:var(--ghost);text-transform:uppercase;border-bottom:1px solid var(--border-soft);padding-bottom:.2rem;margin-bottom:.6rem;}

/* === Gold accent rule (decorative divider) === */
.gold-rule{width:48px;height:2px;background:var(--accent-gold);margin:0 auto 1.5rem;border:none;}

/* === Theme toggle button (used in toolbars) === */
.theme-btn{
  background:transparent;border:1px solid var(--border);border-radius:4px;
  color:var(--muted);width:34px;height:34px;cursor:pointer;font-size:1rem;
  transition:border-color .15s,color .15s;display:flex;align-items:center;justify-content:center;
  padding:0;letter-spacing:0;
}
.theme-btn:hover{border-color:var(--accent-gold);color:var(--accent-gold);background:transparent;}

/* === Nav buttons (same feel as theme toggle but sized for text) === */
.nav-btn{
  background:transparent;border:1px solid var(--border);border-radius:4px;
  color:var(--muted);height:34px;cursor:pointer;font-size:.82rem;
  transition:border-color .15s,color .15s;display:inline-flex;align-items:center;justify-content:center;
  padding:0 .65rem;text-decoration:none;letter-spacing:.03em;
}
.nav-btn:hover{border-color:var(--accent-gold);color:var(--accent-gold);background:transparent;}

/* === Text selection === */
::selection{background:var(--accent-gold);color:var(--bg);}

/* === Scrollbars (webkit) === */
::-webkit-scrollbar{width:6px;height:6px;}
::-webkit-scrollbar-track{background:var(--sunken);}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px;}
::-webkit-scrollbar-thumb:hover{background:var(--accent-gold);}

/* === View layout modes (applied via data-view-layout on .main) === */
/* --- Reading: images in left column, text in right column --- */
.main[data-view-layout=reading] .panels-top,
.main[data-view-layout=reading] .panels-bottom { display:contents; }
.main[data-view-layout=reading] .page {
  display:grid;
  grid-template-columns:1fr 1fr;
  grid-template-areas:"title title" "meta meta" "orig text" "bin text" "notes notes";
  gap:1rem;
}
.main[data-view-layout=reading] .page > h2    { grid-area:title; }
.main[data-view-layout=reading] .page-meta    { grid-area:meta;  }
.main[data-view-layout=reading] .col-original  { grid-area:orig;  }
.main[data-view-layout=reading] .col-binarized { grid-area:bin;   }
.main[data-view-layout=reading] .col-text      { grid-area:text; align-self:stretch; }
.main[data-view-layout=reading] .col-text .line-editor-wrap { height:100%; min-height:400px; }
.main[data-view-layout=reading] .col-notes     { grid-area:notes; }

/* === View layout picker buttons === */
.layout-btns{display:flex;gap:.25rem;}
.layout-btn{padding:.3rem .55rem;font-size:.8rem;border-radius:4px;box-shadow:none;}
.layout-btn.active{background:var(--accent);color:var(--bg);border-color:var(--accent);}
"""

# Injected into <head> — applies saved theme before first paint (no flash)
THEME_INIT = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,500;0,600;1,400&family=Spectral+SC:wght@400;500&family=Noto+Naskh+Arabic:wght@400;500&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<script>
(function(){
  var t=localStorage.getItem('kraken-theme')||'dark';
  document.documentElement.setAttribute('data-theme',t);
})();
</script>"""

# Inline theme toggle button — drop anywhere inside a toolbar
THEME_TOGGLE = """<button id="kraken-theme-btn" class="theme-btn"
  onclick="(function(){var t=document.documentElement.getAttribute('data-theme')==='dark'?'light':'dark';document.documentElement.setAttribute('data-theme',t);localStorage.setItem('kraken-theme',t);document.getElementById('kraken-theme-btn').textContent=t==='dark'?'\\u2600':'\\u263D';})()"
  title="Toggle light / dark">&#x263D;</button>
<script>(function(){var t=document.documentElement.getAttribute('data-theme')||'dark';var b=document.getElementById('kraken-theme-btn');if(b)b.textContent=t==='dark'?'\\u2600':'\\u263D';})();</script>"""

_jenv = Environment()
_jenv.globals.update(BASE_CSS=BASE_CSS, THEME_INIT=THEME_INIT, THEME_TOGGLE=THEME_TOGGLE)

# ---------------- Templates ----------------
INDEX_HTML = _jenv.from_string("""
<!doctype html>
<html>
<head><meta charset="utf-8"><title>Hebrew OCR</title>
{{ THEME_INIT | safe }}
<style>
  {{ BASE_CSS | safe }}
  /* Index-specific */
  input[type=file],input[type=text],input[type=number],select{ width:100%; }
  ul{ list-style:none; padding:0; }
  li{ margin-bottom:.75rem; padding:.5rem; border:1px solid var(--border-soft); border-radius:4px; display:flex; gap:1rem; align-items:center; background:var(--panel); }
  a{ text-decoration:none; padding:.3rem .6rem; background:var(--btn); border:1px solid var(--border); border-radius:4px; color:var(--fg); font-family:var(--font-display); font-size:.8rem; letter-spacing:.06em; transition:background .15s; }
  a:hover{ background:var(--btn-hover); border-color:var(--accent-gold); }
</style></head>
<body>
<header class="app-header">{{ THEME_TOGGLE | safe }}</header>
<div class="page-centered">
  <h1>Hebrew OCR (Kraken 6)</h1>
  <form action="/start_ocr" method="post" enctype="multipart/form-data">
    <label>Project Name (optional)</label>
    <input type="text" name="project_name" placeholder="e.g. test-1">

    <label>Object ID (optional)</label>
    <input type="text" name="obj_id" placeholder="e.g. MS-1234">
    <div class="hint">Appears in the obj_id column of the CSV export</div>

    <label>File (PDF / PNG / JPG / TIFF)</label>
    <input type="file" name="file" required>

    <label>Model</label>
    <select name="model_path">
      {% for m in models %}<option value="{{ m }}" {% if m==selected %}selected{% endif %}>{{ m|replace(userhome,'~') }}</option>{% endfor %}
      {% if not models %}<option value="{{ selected }}" selected>{{ selected|replace(userhome,'~') }}</option>{% endif %}
    </select>
    <div class="hint">Models are read from {{ userhome }}/kraken-models</div>


    <label>Line Padding</label>
    <input type="number" name="seg_pad" value="20" min="0" max="200">
    <div class="hint">Horizontal padding (pixels) added to each detected line. Prevents clipping at line edges. Default: 20</div>

    <label>Reorder RTL output?</label>
    <select name="reorder"><option value="yes" selected>Yes (recommended)</option><option value="no">No</option></select>

    <label>Image Preprocessing</label>
    <select name="preprocess">
      <option value="grayscale">Grayscale (recommended — matches model training)</option>
      <option value="binarize">Binarize (legacy kraken binarize)</option>
      <option value="original">Original (no preprocessing)</option>
    </select>
    <div class="hint">The BiblIA model was trained on grayscale images. Grayscale is recommended; binarize is kept for comparison.</div>

    <label style="display:flex;align-items:center;gap:.5rem;cursor:pointer;margin-top:.75rem;">
      <input type="checkbox" name="deskew" value="yes"> Auto-deskew
    </label>
    <div class="hint">Straighten tilted scans before segmentation. Recommended for skewed manuscripts.</div>

    <input type="hidden" name="merge_lines" value="no">
    <label style="display:flex;align-items:center;gap:.5rem;cursor:pointer;margin-top:.75rem;">
      <input type="checkbox" name="merge_lines" value="yes" checked> Auto-merge split lines
    </label>
    <div class="hint">Merges adjacent lines that appear falsely split (large horizontal overlap, small vertical gap). Default: on.</div>

    <input type="hidden" name="segment_only" value="no">
    <label style="display:flex;align-items:center;gap:.5rem;cursor:pointer;margin-top:.75rem;">
      <input type="checkbox" name="segment_only" value="yes"> Segment only (skip OCR — for training-data prep)
    </label>
    <div class="hint">Produces line polygons and crops with empty .gt.txt files for manual transcription. The selected Model is used as the segmentation (blla) model; leave default for kraken's built-in blla.</div>

    <button type="submit">Run OCR</button>
  </form>

  <div style="margin-top:1rem;"><a href="/models">View Models &amp; Training</a></div>

  {% if sessions %}
  <div style="margin-top:1rem;"><a href="/sessions">View {{ sessions|length }} Existing Session{{ 's' if sessions|length != 1 }}</a></div>
  {% endif %}
</div>
</body></html>
""")

PROGRESS_HTML = _jenv.from_string("""
<!doctype html>
<html>
<head><meta charset="utf-8"><title>Processing…</title>
{{ THEME_INIT | safe }}
<style>
  {{ BASE_CSS | safe }}
  /* Progress-specific */
  .bar{ width:100%; height:10px; background:var(--sunken); border:1px solid var(--border-soft); border-radius:999px; overflow:hidden; margin:.5rem 0 1rem; }
  .fill{ height:100%; background:linear-gradient(90deg,var(--accent),var(--accent-gold)); width:0%; border-radius:999px; transition:width .4s ease; }
  .log{ font-family:var(--font-mono); background:var(--sunken); border:1px solid var(--border-soft); border-radius:4px; padding:.75rem; max-height:220px; overflow:auto; color:var(--muted); font-size:.85rem; }
  .row{ display:flex; gap:.75rem; align-items:center; }
  .pill{ background:var(--btn); border:1px solid var(--border); border-radius:999px; padding:.2rem .75rem; font-family:var(--font-mono); font-size:.85rem; color:var(--muted); }
  /* Scan animation */
  .scan-container{
    position:relative; max-width:500px; margin:1.5rem auto;
    border:1px solid var(--border-soft); border-radius:6px;
    overflow:hidden; box-shadow:var(--shadow-strong); background:var(--sunken);
  }
  .scan-container img{ width:100%; display:block; transition:opacity .4s; }
  .scan-line{
    position:absolute; left:0; width:100%; height:24px;
    background:linear-gradient(180deg,transparent,rgba(154,120,32,.4),var(--accent-gold),rgba(154,120,32,.4),transparent);
    box-shadow:0 0 24px var(--accent-gold),0 0 8px var(--accent-gold);
    animation:scanDown 2.5s ease-in-out infinite; z-index:2;
  }
  .scan-tint{
    position:absolute; top:0; left:0; width:100%;
    background:linear-gradient(180deg,rgba(154,120,32,.08),rgba(154,120,32,.03));
    animation:tintGrow 2.5s ease-in-out infinite; z-index:1; pointer-events:none;
  }
  @keyframes scanDown{ 0%{top:-3px} 100%{top:100%} }
  @keyframes tintGrow{ 0%{height:0} 100%{height:100%} }
</style>
</head>
<body>
<header class="app-header">{{ THEME_TOGGLE | safe }}</header>
<div class="page-centered">
  <h1>Processing…</h1>
  <div class="scan-container" id="scan-container" style="display:none;">
    <img id="scan-img" src="" alt="Processing…">
    <div class="scan-line"></div>
    <div class="scan-tint"></div>
  </div>
  <div class="row"><div>Session:</div><div class="pill">{{ session }}</div></div>
  <div id="status">Starting…</div>
  <div class="bar"><div id="fill" class="fill"></div></div>
  <div class="log" id="log"></div>
  <script>
    const session = "{{ session }}";
    let currentSrc = '';
    async function poll(){
      try{
        const r = await fetch(`/status?session=${encodeURIComponent(session)}`);
        const j = await r.json();
        const pct = j.total ? Math.round((j.done_pages / j.total) * 100) : 0;
        document.getElementById('status').textContent = `State: ${j.state} | ${j.done_pages} / ${j.total}`;
        document.getElementById('fill').style.width = pct + '%';
        document.getElementById('log').textContent = (j.errors || []).join('\\n');
        if (j.current_image) {
          const newSrc = `/image/${encodeURIComponent(session)}/${encodeURIComponent(j.current_image)}`;
          if (newSrc !== currentSrc) {
            currentSrc = newSrc;
            const img = document.getElementById('scan-img');
            img.style.opacity = '0';
            img.src = newSrc;
            img.onload = function(){ img.style.opacity = '1'; };
          }
          document.getElementById('scan-container').style.display = '';
        }
        if (j.state === 'done') {
          location.href = `/view?session=${encodeURIComponent(session)}`;
          return;
        }
      }catch(e){}
      setTimeout(poll, 1600);
    }
    poll();
  </script>
</div>
</body></html>
""")

# --- VIEW (with sidebar thumbnails, no hover preview) ---
VIEW_HTML = _jenv.from_string("""
<!doctype html>
<html>
<head><meta charset="utf-8"><title>OCR Viewer</title>
{{ THEME_INIT | safe }}
<style>
  {{ BASE_CSS | safe }}
  /* View-specific */
  .panels-top,.panels-bottom{ display:grid; grid-template-columns:1fr 1fr; gap:.75rem; margin-bottom:.75rem; align-items:start; }
  .col-images{ display:flex; flex-direction:column; gap:.75rem; }
  .page{ margin-bottom:2rem; border-radius:6px; padding:1.25rem; background:var(--panel); border:1px solid var(--border-soft); box-shadow:var(--shadow-strong); }
  .image-card{ background:var(--panel); border-radius:6px; border:1px solid var(--border-soft); box-shadow:var(--shadow); padding:1rem; transition:box-shadow .2s; }
  .image-card:hover{ box-shadow:var(--shadow-strong); }
  .imgwrap{ position:relative; max-height:55vh; overflow:auto; border:1px solid var(--border); border-radius:4px; cursor:zoom-in; }
  .imgwrap img{ max-width:100%; display:block; filter: var(--img-filter, none); }
  .filter-toolbar{ margin:.25rem 0; font-size:.9rem; }
  .filter-toolbar summary{ cursor:pointer; color:var(--muted); }
  .filter-controls{ display:flex; flex-wrap:wrap; gap:.75rem; align-items:center; padding:.5rem; }
  .filter-controls label{ display:flex; align-items:center; gap:.35rem; }
  .img-zoom-inner{ position:relative; width:100%; display:block; }
  .binimg{ display:block; width:100%; height:auto; }
  .zoom-badge{ font-weight:normal; font-size:.8rem; color:var(--accent-gold); opacity:0; transition:opacity .25s; }
  .seg-overlay{ position:absolute; top:0; left:0; width:100%; height:100%; z-index:2; pointer-events:none; }
  .seg-overlay.interactive{ pointer-events:auto; cursor:default; }
  .linepoly{ fill:var(--poly-fill, rgba(154,120,32,0.06)); stroke:var(--poly-stroke, var(--accent-gold)); stroke-width:2px; vector-effect:non-scaling-stroke; cursor:pointer; }
  .linepoly.highlighted{ stroke:var(--poly-hl-stroke, var(--red)); stroke-width:4px; fill:var(--poly-hl-fill, rgba(200,80,80,0.15)); }
  .line-editor-wrap{ display:flex; flex-direction:row; height:70vh; min-height:400px; max-height:80vh; }
  .line-editor-wrap .line-gutter{
    width:2.8rem; padding:.6rem .3rem .6rem 0;
    background:var(--sunken); color:var(--ghost); font-family:var(--font-mono);
    font-size:14px; line-height:1.4; text-align:right;
    overflow:hidden; user-select:none; flex-shrink:0;
    border:1px solid var(--border); border-right:none; border-radius:4px 0 0 4px;
  }
  .line-gutter div{ padding-right:.4rem; }
  .line-gutter .active-ln{ color:var(--fg); font-weight:700; }
  .line-editor-wrap textarea{ width:100%; height:70vh; min-height:400px; border-radius:0 4px 4px 0; resize:none; line-height:1.4; }
  .actions{ display:flex; gap:.6rem; align-items:center; margin-top:.5rem; }
  .controls-group{ display:flex; gap:1rem; align-items:center; margin-left:auto; }
  .page-nav a{ padding:.5rem 1rem; border:1px solid var(--border); border-radius:4px; background:var(--btn); color:var(--fg); text-decoration:none; box-shadow:var(--shadow); font-family:var(--font-display);font-size:.8rem;letter-spacing:.07em; }
  .page-nav a:hover{ background:var(--btn-hover); border-color:var(--accent-gold); }
  .dropdown-menu{ display:none; }
  .dropdown-menu.open{ display:block; }
  .dropdown-menu button:hover{ background:var(--btn-hover); }
  @media(max-width:1100px){ .panels-top,.panels-bottom{ grid-template-columns:1fr; } }
</style>
</head>
<body>
  <div id="app" class="app">
    <aside class="sidebar">
      <div class="sidebar-header">
        <button id="toggleSidebar" class="sidebar-btn" title="Toggle Sidebar">☰ <span class="label">Sidebar</span></button>
        <div class="sidebar-title" title="Session">{{ session }}</div>
      </div>
      <div class="thumbs">
        {% for p in pages %}
        <a class="thumb-link" href="/view?session={{ session }}&page={{ p.title }}">
          <div class="thumb-item">
            <img class="thumb-img"
                 loading="lazy" decoding="async"
                 src="/thumb/{{ session }}/{{ p.orig_image_name }}?w=240"
                 alt="{{ p.title }} thumb">
            <div class="thumb-meta">
              <div class="thumb-title">{{ p.title }}</div>
              <div class="thumb-sub">{{ p.text.split('\\n')|length }} lines</div>
            </div>
          </div>
        </a>
        {% endfor %}
      </div>
    </aside>

    <main class="main" id="view-main">
<script>
(function(){
  var vl=localStorage.getItem('kraken-view-layout')||'split';
  document.getElementById('view-main').setAttribute('data-view-layout',vl);
})();
</script>
      <div class="top">
        <form action="/set_script" method="post" id="script-form" style="display:flex;align-items:center;gap:.4rem;">
          <input type="hidden" name="session" value="{{ session }}">
          <input type="hidden" name="return_to" value="view">
          <input type="hidden" name="page" value="{{ page }}">
          <label class="small" for="script-input">Script:</label>
          <input id="script-input" list="scripts-list" name="script" required
                 pattern="[A-Za-z0-9_-]+" title="Letters, numbers, dashes, underscores only"
                 placeholder="e.g. Arabic_Kufic" autocomplete="off"
                 value="{{ current_script or '' }}"
                 style="width:11rem;">
          <datalist id="scripts-list">
            {% for s in scripts %}<option value="{{ s }}">{% endfor %}
          </datalist>
          <button type="submit">Save script</button>
        </form>

        <form action="/download" method="get"><input type="hidden" name="session" value="{{ session }}"><button type="submit">Download merged text</button></form>

        <form action="/export_viewer_csv" method="get">
          <input type="hidden" name="session" value="{{ session }}">
          <button type="submit" {% if not current_script %}disabled title="Set a script above first"{% endif %}>
            Export CSV
          </button>
        </form>

        <form action="/export_gt" method="get">
          <input type="hidden" name="session" value="{{ session }}">
          <button type="submit">Export GT</button>
        </form>

        <form action="/training" method="get">
          <input type="hidden" name="session" value="{{ session }}">
          <input type="hidden" name="page" value="{{ page }}">
          <button type="submit">Open Training</button>
        </form>

        <button type="button" id="push-gt-btn">Push to Training</button>

        <div style="position:relative;display:inline-block;">
          <button type="button" onclick="this.nextElementSibling.classList.toggle('open')">Reprocess &#9662;</button>
          <div class="dropdown-menu" style="display:none;position:absolute;top:100%;left:0;margin-top:4px;background:var(--panel);border:1px solid var(--border);border-radius:6px;box-shadow:var(--shadow-strong);z-index:100;min-width:180px;">
            <button type="button" onclick="this.parentElement.classList.remove('open');reprocess({{ page }})" style="display:block;width:100%;text-align:left;padding:.5rem .75rem;border:none;background:none;color:var(--fg);cursor:pointer;font-size:.85rem;">Reprocess this page</button>
            <button type="button" onclick="this.parentElement.classList.remove('open');reprocess(null)" style="display:block;width:100%;text-align:left;padding:.5rem .75rem;border:none;background:none;color:var(--fg);cursor:pointer;font-size:.85rem;border-top:1px solid var(--border);">Reprocess all pages</button>
            <button type="button" onclick="this.parentElement.classList.remove('open');reocr({{ page }})" style="display:block;width:100%;text-align:left;padding:.5rem .75rem;border:none;background:none;color:var(--fg);cursor:pointer;font-size:.85rem;border-top:1px solid var(--border);">Re-OCR this page (keep polygons)</button>
            <button type="button" onclick="this.parentElement.classList.remove('open');reocrAll()" style="display:block;width:100%;text-align:left;padding:.5rem .75rem;border:none;background:none;color:var(--fg);cursor:pointer;font-size:.85rem;border-top:1px solid var(--border);">Re-OCR all pages (keep polygons)</button>
          </div>
        </div>

        <a href="/">New OCR</a>
        <a href="/models">Models</a>
<span class="small">Session: {{ session }}{% if model_name %} | Model: {{ model_name }}{% endif %}</span>

<div class="controls-group">
  <label><input type="checkbox" id="toggle-original" checked> Original</label>
  <label><input type="checkbox" id="toggle-binarized" checked> {{ preprocess_label }}</label>
  <label><input type="checkbox" id="toggle-text" checked> Text</label>
  <label><input type="checkbox" id="toggle-notes"> Notes</label>
  <span style="width:1px;height:1.2em;background:var(--border);display:inline-block;margin:0 .25rem;align-self:center;"></span>
  <div class="layout-btns" title="Panel layout">
    <button class="layout-btn" data-vl="split"   onclick="setViewLayout('split')"   title="Split — images top, text bottom">Split</button>
    <button class="layout-btn" data-vl="reading" onclick="setViewLayout('reading')" title="Reading — images left, text right">Reading</button>
  </div>
  <span style="width:1px;height:1.2em;background:var(--border);display:inline-block;margin:0 .25rem;align-self:center;"></span>
  {{ THEME_TOGGLE | safe }}
</div>
</div>   <!-- end of .top -->
<details class="filter-toolbar" id="img-filters">
  <summary>Image filters &amp; polygon colors</summary>
  <div class="filter-controls">
    <label>Invert <input type="range" id="f-invert" min="0" max="1" step="1" value="0"></label>
    <label>Contrast <input type="range" id="f-contrast" min="50" max="300" step="5" value="100"><span id="f-contrast-v">100%</span></label>
    <label>Brightness <input type="range" id="f-brightness" min="50" max="200" step="5" value="100"><span id="f-brightness-v">100%</span></label>
    <label>Saturate <input type="range" id="f-saturate" min="0" max="300" step="10" value="100"><span id="f-saturate-v">100%</span></label>
    <label>Grayscale <input type="range" id="f-grayscale" min="0" max="100" step="10" value="0"><span id="f-grayscale-v">0%</span></label>
    <button type="button" id="f-reset">Reset</button>
  </div>
  <div class="filter-controls" style="border-top:1px solid var(--border); margin-top:.4rem; padding-top:.5rem;">
    <label>Line color <input type="color" id="poly-stroke" value="#9a7820" style="width:32px;height:24px;padding:0;border:1px solid var(--border);border-radius:3px;cursor:pointer;"></label>
    <label>Highlight <input type="color" id="poly-highlight" value="#c84040" style="width:32px;height:24px;padding:0;border:1px solid var(--border);border-radius:3px;cursor:pointer;"></label>
    <span style="color:var(--muted);font-size:.8rem;">Presets:</span>
    <button type="button" class="poly-preset" data-stroke="#9a7820" data-hl="#c84040" style="font-size:.75rem;padding:2px 6px;">Gold/Red</button>
    <button type="button" class="poly-preset" data-stroke="#00ff88" data-hl="#ff3366" style="font-size:.75rem;padding:2px 6px;">Green/Pink</button>
    <button type="button" class="poly-preset" data-stroke="#00ccff" data-hl="#ffaa00" style="font-size:.75rem;padding:2px 6px;">Cyan/Orange</button>
    <button type="button" class="poly-preset" data-stroke="#ff00ff" data-hl="#00ff00" style="font-size:.75rem;padding:2px 6px;">Magenta/Green</button>
    <button type="button" class="poly-preset" data-stroke="#ffffff" data-hl="#ff0000" style="font-size:.75rem;padding:2px 6px;">White/Red</button>
  </div>
</details>
<div class="page-nav" style="display:flex; align-items:center; gap:1rem; margin:0.5rem 0 1rem 0;">
  <a href="/view?session={{ session }}&page={{ page-1 if page>1 else 1 }}">Prev</a>
  <span class="small">Page {{ page }} / {{ total_pages }}</span>
  <a href="/view?session={{ session }}&page={{ page+1 if page<total_pages else total_pages }}">Next</a>
</div>

      {% set p = current_page %}
<div class="page" id="{{ p.title }}">
  <h2>Page {{ p.title }}</h2>
  <div class="page-meta" style="display:flex; gap:1rem; flex-wrap:wrap; margin-bottom:1rem; padding:.75rem; background:var(--bg); border:1px solid var(--border); border-radius:8px;">
    <label style="display:flex; flex-direction:column; gap:.25rem; font-size:.85rem; color:var(--muted);">Folio <input type="text" name="folio" class="page-meta-input" value="{{ p.folio }}" placeholder="e.g. 1" style="padding:.3rem .5rem; background:var(--panel); color:var(--fg); border:1px solid var(--border); border-radius:6px; width:80px;"></label>
    <label style="display:flex; flex-direction:column; gap:.25rem; font-size:.85rem; color:var(--muted);">Side <input type="text" name="side" class="page-meta-input" value="{{ p.side }}" placeholder="e.g. recto" style="padding:.3rem .5rem; background:var(--panel); color:var(--fg); border:1px solid var(--border); border-radius:6px; width:80px;"></label>
    <label style="display:flex; flex-direction:column; gap:.25rem; font-size:.85rem; color:var(--muted);">Part <input type="text" name="part" class="page-meta-input" value="{{ p.part }}" placeholder="e.g. 1" style="padding:.3rem .5rem; background:var(--panel); color:var(--fg); border:1px solid var(--border); border-radius:6px; width:60px;"></label>
    <span class="page-meta-status small" style="align-self:flex-end;"></span>
  </div>
  <!-- Top: images side by side -->
  <div class="panels-top">
    <div class="image-card col-original">
      <div class="coltitle" style="display:flex; align-items:center; justify-content:space-between;">
        <span>Original</span>
        <span class="zoom-badge" id="zoom-badge-orig">100%</span>
      </div>
      <div class="imgwrap" id="imgwrap-orig">
        <div class="img-zoom-inner">
          <img src="/image/{{ session }}/{{ p.orig_image_name }}" loading="lazy" decoding="async" alt="{{ p.title }} original">
        </div>
      </div>
    </div>

    <div class="image-card col-binarized">
      <div class="coltitle" style="display:flex; align-items:center; justify-content:space-between;">
        <span>{{ preprocess_label }}</span>
        <span class="zoom-badge" id="zoom-badge-bin">100%</span>
      </div>
      <div class="imgwrap" id="imgwrap-bin">
        <div class="img-zoom-inner">
          <img class="binimg" id="bin-img"
               src="/image/{{ session }}/{{ p.bin_image_name }}"
               loading="lazy" decoding="async"
               alt="{{ p.title }} binarized">
          {% if seg_lines %}
          <svg class="seg-overlay interactive" id="seg-overlay"
               preserveAspectRatio="xMinYMin meet">
            {% for line in seg_lines %}
            <polygon class="linepoly"
                     data-line-idx="{{ line.idx }}"
                     points="{{ line.points_str }}"></polygon>
            {% endfor %}
          </svg>
          {% endif %}
        </div>
      </div>
    </div>
  </div>

  <!-- Bottom: text + notes side by side -->
  <div class="panels-bottom">
    <div class="image-card col-text">
      <div class="coltitle" style="display:flex; align-items:center; justify-content:space-between;">
        <span>Text (editable)</span>
        <div class="font-size-control" style="font-weight:normal; font-size:.85rem; color:var(--muted); display:flex; align-items:center; gap:.6rem;">
          <select class="font-family-input" style="padding:2px 4px; background:var(--panel); color:var(--fg); border:1px solid var(--border); border-radius:3px; font-size:.85rem;">
            <option value="monospace">Mono</option>
            <option value="'SBL Hebrew', 'Frank Ruehl CLM', 'Noto Serif Hebrew', serif">Hebrew Serif</option>
            <option value="'Noto Sans Hebrew', 'Arial Hebrew', sans-serif">Hebrew Sans</option>
            <option value="serif">Serif</option>
            <option value="sans-serif">Sans</option>
          </select>
          <input type="number" class="font-size-input" value="14" min="8" max="36" step="1"
                 style="width:50px; padding:2px 4px; background:var(--panel); color:var(--fg); border:1px solid var(--border); border-radius:3px;"> px
        </div>
      </div>

      <form action="/save" method="post">
  <input type="hidden" name="session" value="{{ session }}">
  <input type="hidden" name="txt_name" value="{{ p.txt_name }}">
  <input type="hidden" name="page" value="{{ page }}">
  <div class="line-editor-wrap">
    <div class="line-gutter" id="line-gutter"></div>
    <textarea name="text" class="editable-textarea" id="main-textarea">{{ p.text }}</textarea>
  </div>
  <div class="actions">
    <button type="submit">Save</button>
    <span class="small">Saved to: {{ p.txt_name }}</span>
  </div>
</form>
    </div>

    <div class="image-card col-notes" style="display:none;">
      <div class="coltitle"><span>Notes</span></div>
      <div class="line-editor-wrap">
        <div class="line-gutter" id="notes-gutter"></div>
        <textarea class="notes-textarea" id="notes-textarea" style="width:100%; height:100%; padding:.6rem; line-height:1.5; border:1px solid var(--border); border-radius:0 10px 10px 0; background:var(--panel); color:var(--fg); font-family:monospace; font-size:14px; resize:none; white-space:pre-wrap;">{{ p.notes }}</textarea>
      </div>
      <span class="notes-status small" style="margin-top:.25rem; display:block;"></span>
    </div>
  </div>
</div>
    </main>
  </div>

<script>
(function() {
  const app = document.getElementById('app');
  const toggleSidebarBtn = document.getElementById('toggleSidebar');
  const savedCollapsed = localStorage.getItem('viewerSidebarCollapsed') === '1';
  if (savedCollapsed) app.classList.add('collapsed');

  toggleSidebarBtn.addEventListener('click', () => {
    app.classList.toggle('collapsed');
    localStorage.setItem('viewerSidebarCollapsed', app.classList.contains('collapsed') ? '1' : '0');
  });

  const qs = sel => document.querySelector(sel);
  const qsa = sel => Array.from(document.querySelectorAll(sel));
  const setVisible = (cls, on) => qsa('.' + cls).forEach(el => el.style.display = on ? '' : 'none');
  const cbOrig = qs('#toggle-original'), cbBin = qs('#toggle-binarized'), cbText = qs('#toggle-text'), cbNotes = qs('#toggle-notes');

  function applyLayout() {
    const vl = (document.getElementById('view-main') || {}).dataset.viewLayout || 'split';
    setVisible('col-original', cbOrig.checked);
    setVisible('col-binarized', cbBin.checked);
    setVisible('col-text', cbText.checked);
    setVisible('col-notes', cbNotes.checked);

    const panelsTop = qs('.panels-top');
    const panelsBottom = qs('.panels-bottom');

    if (vl === 'reading') {
      // CSS handles the grid via display:contents + grid-template-areas.
      // Clear any inline styles that would fight it.
      if (panelsTop)    { panelsTop.style.display = ''; panelsTop.style.gridTemplateColumns = ''; }
      if (panelsBottom) { panelsBottom.style.display = ''; panelsBottom.style.gridTemplateColumns = ''; }
    } else {
      // Split: manage container grid columns based on visible panels.
      const showOrig = cbOrig.checked, showBin = cbBin.checked;
      if (panelsTop) {
        panelsTop.style.display = (showOrig || showBin) ? '' : 'none';
        const topParts = [];
        if (showOrig) topParts.push('1fr');
        if (showBin) topParts.push('1fr');
        panelsTop.style.gridTemplateColumns = topParts.join(' ') || '1fr';
      }
      const showText = cbText.checked, showNotes = cbNotes.checked;
      if (panelsBottom) {
        panelsBottom.style.display = (showText || showNotes) ? '' : 'none';
        const botParts = [];
        if (showText) botParts.push('1fr');
        if (showNotes) botParts.push('1fr');
        panelsBottom.style.gridTemplateColumns = botParts.join(' ') || '1fr';
      }
    }
  }
  window._krakenApplyLayout = applyLayout;

  const applyVisibility = () => { applyLayout(); };

  function savePanelState() {
    localStorage.setItem('kraken-panels', JSON.stringify({
      orig: cbOrig.checked, bin: cbBin.checked,
      text: cbText.checked, notes: cbNotes.checked
    }));
  }
  function restorePanelState() {
    try {
      const s = JSON.parse(localStorage.getItem('kraken-panels') || '{}');
      if ('orig'  in s) cbOrig.checked  = s.orig;
      if ('bin'   in s) cbBin.checked   = s.bin;
      if ('text'  in s) cbText.checked  = s.text;
      if ('notes' in s) cbNotes.checked = s.notes;
    } catch(e) {}
  }
  [cbOrig, cbBin, cbText, cbNotes].forEach(cb => cb.addEventListener('change', () => { applyVisibility(); savePanelState(); }));

  const textareas = qsa('.editable-textarea'), fontSizeInputs = qsa('.font-size-input');
  const fontFamilyInputs = qsa('.font-family-input');
  const gutters = qsa('.line-gutter');

  function applyFontSize(newSize) {
    const n = parseInt(newSize, 10);
    if (n >= 8 && n <= 36) {
      textareas.forEach(ta => ta.style.fontSize = n + 'px');
      gutters.forEach(g => g.style.fontSize = n + 'px');
      fontSizeInputs.forEach(input => input.value = n);
      localStorage.setItem('editableFontSize', String(n));
    }
  }

  function applyFontFamily(family) {
    textareas.forEach(ta => ta.style.fontFamily = family);
    fontFamilyInputs.forEach(sel => sel.value = family);
    localStorage.setItem('editableFontFamily', family);
  }

  fontSizeInputs.forEach(input => input.addEventListener('input', (event) => applyFontSize(event.target.value)));
  fontFamilyInputs.forEach(sel => sel.addEventListener('change', (event) => applyFontFamily(event.target.value)));

  function initializeSettings() {
    const savedFontSize = parseInt(localStorage.getItem('editableFontSize') ?? '14', 10);
    applyFontSize(savedFontSize);
    const savedFamily = localStorage.getItem('editableFontFamily');
    if (savedFamily) applyFontFamily(savedFamily);
    restorePanelState();
    applyVisibility();
  }
  initializeSettings();
})();
</script>
<script>
(function(){
  const KEY = 'kraken-img-filters';
  const ids = ['invert','contrast','brightness','saturate','grayscale'];
  const defaults = { invert:0, contrast:100, brightness:100, saturate:100, grayscale:0 };
  const saved = Object.assign({}, defaults, JSON.parse(localStorage.getItem(KEY) || '{}'));
  function compose(v){
    return `invert(${v.invert}) contrast(${v.contrast}%) brightness(${v.brightness}%) saturate(${v.saturate}%) grayscale(${v.grayscale}%)`;
  }
  function apply(v){
    document.documentElement.style.setProperty('--img-filter', compose(v));
    ids.forEach(k => {
      const el = document.getElementById(`f-${k}`); if (el) el.value = v[k];
      const lbl = document.getElementById(`f-${k}-v`); if (lbl) lbl.textContent = (k==='invert' ? (v[k]?'on':'off') : v[k]+'%');
    });
    localStorage.setItem(KEY, JSON.stringify(v));
  }
  apply(saved);
  ids.forEach(k => {
    const el = document.getElementById(`f-${k}`);
    if (!el) return;
    el.addEventListener('input', () => { saved[k] = Number(el.value); apply(saved); });
  });
  const reset = document.getElementById('f-reset');
  if (reset) reset.addEventListener('click', () => apply(Object.assign({}, defaults)));
})();
</script>
<script>
(function(){
  const POLY_KEY = 'kraken-poly-colors';
  const polyDefaults = { stroke: '#9a7820', highlight: '#c84040' };
  const polySaved = Object.assign({}, polyDefaults, JSON.parse(localStorage.getItem(POLY_KEY) || '{}'));

  function hexToRgba(hex, alpha) {
    const r = parseInt(hex.slice(1,3), 16);
    const g = parseInt(hex.slice(3,5), 16);
    const b = parseInt(hex.slice(5,7), 16);
    return `rgba(${r},${g},${b},${alpha})`;
  }

  function applyPolyColors(colors) {
    const style = document.documentElement.style;
    style.setProperty('--poly-stroke', colors.stroke);
    style.setProperty('--poly-fill', hexToRgba(colors.stroke, 0.06));
    style.setProperty('--poly-hl-stroke', colors.highlight);
    style.setProperty('--poly-hl-fill', hexToRgba(colors.highlight, 0.15));
    const strokeInput = document.getElementById('poly-stroke');
    const hlInput = document.getElementById('poly-highlight');
    if (strokeInput) strokeInput.value = colors.stroke;
    if (hlInput) hlInput.value = colors.highlight;
    localStorage.setItem(POLY_KEY, JSON.stringify(colors));
  }

  applyPolyColors(polySaved);

  const strokeInput = document.getElementById('poly-stroke');
  const hlInput = document.getElementById('poly-highlight');
  if (strokeInput) strokeInput.addEventListener('input', () => { polySaved.stroke = strokeInput.value; applyPolyColors(polySaved); });
  if (hlInput) hlInput.addEventListener('input', () => { polySaved.highlight = hlInput.value; applyPolyColors(polySaved); });

  document.querySelectorAll('.poly-preset').forEach(btn => {
    btn.addEventListener('click', () => {
      polySaved.stroke = btn.dataset.stroke;
      polySaved.highlight = btn.dataset.hl;
      applyPolyColors(polySaved);
    });
  });
})();
</script>
<script>
(function () {
  const form = document.querySelector('form[action="/save"]');
  if (!form) return;

  const textarea = form.querySelector('textarea[name="text"]');
  const session = form.querySelector('input[name="session"]').value;
  const pageInput = form.querySelector('input[name="page"]');
  const txtName = form.querySelector('input[name="txt_name"]').value;

  let dirty = false;
  let debounceTimer = null;

  function markDirty() { dirty = true; }

  textarea.addEventListener('input', () => {
    markDirty();
    if (debounceTimer) clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => { autosave('debounce'); }, 1500);
  });

  async function autosave(reason) {
    if (!dirty) return true;
    try {
      const fd = new FormData();
      fd.set('session', session);
      fd.set('txt_name', txtName);
      fd.set('page', pageInput ? pageInput.value : '1');
      fd.set('text', textarea.value);

      await fetch('/save', {
        method: 'POST',
        body: fd,
        headers: { 'X-Requested-With': 'fetch', 'Accept': 'application/json' },
        keepalive: true,
        redirect: 'manual'
      });

      dirty = false;
      return true;
    } catch (e) {
      console.warn('Autosave failed (' + reason + '):', e);
      return false;
    }
  }

  function isInternalViewLink(a) {
    if (!a || !a.href) return false;
    try {
      const u = new URL(a.href, location.origin);
      return u.pathname === '/view' && u.searchParams.get('session') === session;
    } catch { return false; }
  }

  document.addEventListener('click', async (e) => {
    const a = e.target.closest('a');
    if (!a || !isInternalViewLink(a)) return;
    if (!dirty) return;
    e.preventDefault();
    const ok = await autosave('nav');
    location.href = a.href; // navigate either way
  });

  function beaconAutosave() {
    if (!dirty) return;

    const params = new URLSearchParams();
    params.set('session', session);
    params.set('txt_name', txtName);
    params.set('page', pageInput ? pageInput.value : '1');
    params.set('text', textarea.value);

    const blob = new Blob([params.toString()], { type: 'application/x-www-form-urlencoded;charset=UTF-8' });

    if (!(navigator.sendBeacon && navigator.sendBeacon('/save', blob))) {
      fetch('/save', {
        method: 'POST',
        body: params,
        headers: { 'X-Requested-With': 'fetch', 'Accept': 'application/json' },
        keepalive: true,
        redirect: 'manual'
      }).finally(() => { dirty = false; });
    } else {
      dirty = false;
    }
  }

  window.addEventListener('pagehide', beaconAutosave, { capture: true });
  window.addEventListener('beforeunload', beaconAutosave, { capture: true });

  function flashSaved() {
    let el = document.getElementById('autosave-toast');
    if (!el) {
      el = document.createElement('div');
      el.id = 'autosave-toast';
      el.style.position = 'fixed';
      el.style.right = '12px';
      el.style.bottom = '12px';
      el.style.padding = '8px 12px';
      el.style.background = 'rgba(0,0,0,0.7)';
      el.style.color = '#fff';
      el.style.borderRadius = '8px';
      el.style.fontSize = '12px';
      el.style.zIndex = '9999';
      document.body.appendChild(el);
    }
    el.textContent = 'Saved';
    el.style.opacity = '1';
    setTimeout(() => { el.style.transition = 'opacity 600ms'; el.style.opacity = '0'; }, 600);
  }

  const _autosave = autosave;
  autosave = async (reason) => {
    const ok = await _autosave(reason);
    if (ok) flashSaved();
    return ok;
  };
})();
</script>
<script>
// --- Synchronized image zoom + pan (Ctrl+scroll to zoom, double-click to reset) ---
(function () {
  const wrapOrig = document.getElementById('imgwrap-orig');
  const wrapBin  = document.getElementById('imgwrap-bin');
  const badges   = [document.getElementById('zoom-badge-orig'), document.getElementById('zoom-badge-bin')].filter(Boolean);
  const wraps    = [wrapOrig, wrapBin].filter(Boolean);
  if (!wraps.length) return;

  let zoom    = 1;
  let syncing = false;

  function scalableOf(w) { return w.querySelector('.img-zoom-inner') || w.querySelector('img'); }

  function setBadge(z) {
    const pct = Math.round(z * 100) + '%';
    badges.forEach(b => { b.textContent = pct; b.style.opacity = z === 1 ? '0' : '1'; });
  }

  function syncScrollFrom(src) {
    if (syncing) return;
    syncing = true;
    const maxSL = src.scrollWidth  - src.clientWidth;
    const maxST = src.scrollHeight - src.clientHeight;
    const fracX = maxSL > 0 ? src.scrollLeft / maxSL : 0;
    const fracY = maxST > 0 ? src.scrollTop  / maxST : 0;
    wraps.forEach(w => {
      if (w === src) return;
      w.scrollLeft = fracX * (w.scrollWidth  - w.clientWidth);
      w.scrollTop  = fracY * (w.scrollHeight - w.clientHeight);
    });
    syncing = false;
  }

  function setZoom(z, cx, cy, srcWrap) {
    const prev = zoom;
    zoom = Math.max(0.25, Math.min(8, z));
    const ratio = zoom / prev;
    wraps.forEach(w => {
      const sc = scalableOf(w);
      sc.style.width  = zoom === 1 ? '' : (zoom * 100) + '%';
      w.style.cursor  = zoom === 1 ? 'zoom-in' : 'grab';
    });
    if (cx !== undefined && srcWrap && zoom !== prev) {
      const sl = srcWrap.scrollLeft, st = srcWrap.scrollTop;
      srcWrap.scrollLeft = (sl + cx) * ratio - cx;
      srcWrap.scrollTop  = (st + cy) * ratio - cy;
      syncScrollFrom(srcWrap);
    }
    setBadge(zoom);
  }

  wraps.forEach(w => {
    w.addEventListener('wheel', (e) => {
      if (!e.ctrlKey && !e.metaKey) return;
      e.preventDefault();
      // Normalize deltaY to approximate pixels so trackpad and mouse wheel feel the same
      let delta = e.deltaY;
      if (e.deltaMode === 1) delta *= 30;  // lines → pixels
      if (e.deltaMode === 2) delta *= 800; // pages → pixels
      const r = w.getBoundingClientRect();
      setZoom(zoom * Math.pow(0.997, delta), e.clientX - r.left, e.clientY - r.top, w);
    }, { passive: false });

    w.addEventListener('dblclick', () => {
      zoom = 1;
      wraps.forEach(w2 => {
        scalableOf(w2).style.width = '';
        w2.style.cursor = 'zoom-in';
        w2.scrollLeft = 0;
        w2.scrollTop  = 0;
      });
      setBadge(1);
    });

    w.addEventListener('scroll', () => { if (!syncing) syncScrollFrom(w); });
    w.title = 'Ctrl+scroll to zoom · Double-click to reset';
  });
})();
</script>
<script>
// --- Line numbers + polygon highlight ---
(function () {
  const textarea = document.getElementById('main-textarea');
  const gutter   = document.getElementById('line-gutter');
  const svg      = document.getElementById('seg-overlay');
  const binImg   = document.getElementById('bin-img');
  if (!textarea || !gutter) return;

  // Set SVG viewBox from image's natural dimensions once loaded
  if (svg && binImg) {
    function initViewBox() {
      if (binImg.naturalWidth)
        svg.setAttribute('viewBox', `0 0 ${binImg.naturalWidth} ${binImg.naturalHeight}`);
    }
    if (binImg.complete) initViewBox();
    else binImg.addEventListener('load', initViewBox);
  }

  let lastLine = -1;

  function currentLineNumber() {
    return textarea.value.substring(0, textarea.selectionStart).split('\\n').length;
  }

  function highlightPoly(lineNum) {
    if (!svg) return;
    svg.querySelectorAll('.linepoly.highlighted').forEach(p => p.classList.remove('highlighted'));
    const poly = svg.querySelector(`.linepoly[data-line-idx="${lineNum}"]`);
    if (poly) {
      poly.classList.add('highlighted');
      const wrap = document.getElementById('imgwrap-bin');
      if (wrap) {
        const wrapRect = wrap.getBoundingClientRect();
        const polyRect = poly.getBoundingClientRect();
        if (polyRect.top < wrapRect.top || polyRect.bottom > wrapRect.bottom) {
          poly.scrollIntoView({ block: 'nearest' });
        }
      }
    }
  }

  function updateGutter() {
    const lineNum = currentLineNumber();
    const lines = textarea.value.split('\\n');

    // Rebuild gutter only when line count changes
    if (gutter.children.length !== lines.length) {
      gutter.innerHTML = lines.map((_, i) => `<div>${i + 1}</div>`).join('');
    }

    // Highlight active line number
    Array.from(gutter.children).forEach((el, i) => {
      el.classList.toggle('active-ln', i + 1 === lineNum);
    });

    // Sync gutter scroll with textarea
    gutter.scrollTop = textarea.scrollTop;

    // Highlight polygon only when line actually changed
    if (lineNum !== lastLine) {
      lastLine = lineNum;
      highlightPoly(lineNum);
    }
  }

  textarea.addEventListener('input',  updateGutter);
  textarea.addEventListener('click',  updateGutter);
  textarea.addEventListener('keyup',  updateGutter);
  textarea.addEventListener('scroll', () => { gutter.scrollTop = textarea.scrollTop; });

  // Polygon click → jump to that line in textarea
  if (svg) {
    svg.addEventListener('click', (e) => {
      const poly = e.target.closest('.linepoly');
      if (!poly) return;
      const lineIdx = parseInt(poly.dataset.lineIdx, 10);
      if (!lineIdx) return;
      const lines = textarea.value.split('\\n');
      let offset = 0;
      for (let i = 0; i < lineIdx - 1 && i < lines.length; i++)
        offset += lines[i].length + 1;
      textarea.focus();
      textarea.setSelectionRange(offset, offset);
      updateGutter();
    });
  }

  updateGutter(); // initial render

  // --- Notes gutter: mirror line numbers from main textarea ---
  const notesTA = document.getElementById('notes-textarea');
  const notesGutter = document.getElementById('notes-gutter');
  if (notesTA && notesGutter) {
    function updateNotesGutter() {
      // Use the main textarea's line count for the gutter
      const mainLines = textarea.value.split('\\n');
      const notesLines = notesTA.value.split('\\n');
      const lineCount = Math.max(mainLines.length, notesLines.length);
      if (notesGutter.children.length !== lineCount) {
        notesGutter.innerHTML = Array.from({length: lineCount}, (_, i) => `<div>${i + 1}</div>`).join('');
      }
      notesGutter.scrollTop = notesTA.scrollTop;
    }
    notesTA.addEventListener('input', updateNotesGutter);
    notesTA.addEventListener('scroll', () => { notesGutter.scrollTop = notesTA.scrollTop; });
    // Also update when main textarea changes (line count may change)
    textarea.addEventListener('input', updateNotesGutter);
    updateNotesGutter();
  }
})();
</script>
<script>
// --- Push to Training ---
(function () {
  const btn = document.getElementById('push-gt-btn');
  if (!btn) return;

  const form     = document.querySelector('form[action="/save"]');
  const session  = form.querySelector('input[name="session"]').value;
  const txtName  = form.querySelector('input[name="txt_name"]').value;
  const pageStem = txtName.replace(/^.*[\\/]/, '').replace(/\\.txt$/, '');

  btn.addEventListener('click', async () => {
    if (!confirm(`Push corrected viewer text to training GT for ${pageStem}?\\nThis overwrites existing GT text for this page.`)) return;
    btn.disabled = true;
    btn.textContent = 'Pushing…';
    try {
      const fd = new FormData();
      fd.set('session', session);
      fd.set('page_stem', pageStem);
      const res  = await fetch('/api/push_to_gt', { method: 'POST', body: fd });
      const data = await res.json();
      const t = document.createElement('div');
      t.className = 'toast';
      t.innerHTML = `<span>${data.updated} lines pushed to training</span><button onclick="this.parentNode.remove()">✕</button>`;
      document.body.appendChild(t);
      setTimeout(() => t.remove(), 4000);
    } catch (e) {
      alert('Push failed: ' + e.message);
    } finally {
      btn.disabled = false;
      btn.textContent = 'Push to Training';
    }
  });
})();
</script>
<script>
// --- Page Metadata auto-save ---
(function() {
    const metaBar = document.querySelector('.page-meta');
    if (!metaBar) return;
    const inputs = metaBar.querySelectorAll('.page-meta-input');
    const statusEl = metaBar.querySelector('.page-meta-status');
    let debounce = null;

    const notesTa = document.getElementById('notes-textarea');
    const notesStatus = document.querySelector('.notes-status');

    function savePageMeta() {
        const fd = new FormData();
        fd.set('session', '{{ session }}');
        fd.set('page_stem', '{{ current_page.page_stem }}');
        inputs.forEach(inp => fd.set(inp.name, inp.value));
        if (notesTa) fd.set('notes', notesTa.value);
        fetch('/api/save_page_meta', { method: 'POST', body: fd })
            .then(r => {
                if (r.ok) {
                    if (statusEl) { statusEl.textContent = 'Saved'; setTimeout(() => statusEl.textContent = '', 1500); }
                    if (notesStatus) { notesStatus.textContent = 'Saved'; setTimeout(() => notesStatus.textContent = '', 1500); }
                }
            })
            .catch(() => { if (statusEl) statusEl.textContent = 'Save failed'; });
    }

    inputs.forEach(inp => {
        inp.addEventListener('input', () => {
            if (debounce) clearTimeout(debounce);
            debounce = setTimeout(savePageMeta, 800);
        });
    });
    if (notesTa) {
        notesTa.addEventListener('input', () => {
            if (debounce) clearTimeout(debounce);
            debounce = setTimeout(savePageMeta, 800);
        });
    }
})();
</script>
<script>
function setViewLayout(v) {
  var m = document.getElementById('view-main');
  if (m) m.setAttribute('data-view-layout', v);
  localStorage.setItem('kraken-view-layout', v);
  document.querySelectorAll('.layout-btn').forEach(function(b){
    b.classList.toggle('active', b.dataset.vl === v);
  });
  if (window._krakenApplyLayout) window._krakenApplyLayout();
}
(function(){
  var vl = localStorage.getItem('kraken-view-layout') || 'split';
  setViewLayout(vl);
})();
</script>

<div id="reprocess-overlay" style="display:none;position:fixed;inset:0;z-index:9999;background:rgba(0,0,0,.55);backdrop-filter:blur(3px);justify-content:center;align-items:center;">
  <div style="background:var(--panel,#1a1a1a);border:1px solid var(--border,#333);border-radius:10px;padding:2rem 2.5rem;text-align:center;box-shadow:0 8px 32px rgba(0,0,0,.4);">
    <div style="margin-bottom:1rem;">
      <svg width="40" height="40" viewBox="0 0 40 40" style="animation:reproc-spin 1s linear infinite;">
        <circle cx="20" cy="20" r="16" fill="none" stroke="var(--border,#555)" stroke-width="3"/>
        <circle cx="20" cy="20" r="16" fill="none" stroke="var(--accent-gold,#9a7820)" stroke-width="3" stroke-dasharray="80" stroke-dashoffset="60" stroke-linecap="round"/>
      </svg>
    </div>
    <div id="reprocess-msg" style="font-size:1.05rem;color:var(--fg,#eee);">Reprocessing…</div>
    <div id="reprocess-detail" style="font-size:.85rem;color:var(--muted,#888);margin-top:.4rem;"></div>
  </div>
</div>
<style>@keyframes reproc-spin{to{transform:rotate(360deg)}}</style>
<script>
document.addEventListener('click', function(e) {
  document.querySelectorAll('.dropdown-menu.open').forEach(function(m) {
    if (!m.parentElement.contains(e.target)) m.classList.remove('open');
  });
});
function reprocess(page) {
  var label = page ? 'page ' + page : 'all pages';
  if (!confirm('Re-run segmentation and OCR on ' + label + '?')) return;
  var overlay = document.getElementById('reprocess-overlay');
  var msg = document.getElementById('reprocess-msg');
  var detail = document.getElementById('reprocess-detail');
  overlay.style.display = 'flex';
  msg.textContent = 'Reprocessing ' + label + '…';
  detail.textContent = '';
  var session = '{{ session }}';
  var body = new URLSearchParams();
  body.set('session', session);
  if (page) body.set('page', page);
  fetch('/reprocess', {method:'POST', body:body})
    .then(function(r){ return r.json(); })
    .then(function(){
      var poll = setInterval(function(){
        fetch('/status?session=' + encodeURIComponent(session))
          .then(function(r){ return r.json(); })
          .then(function(d){
            detail.textContent = d.state || '';
            if (d.state === 'done' || (d.state && d.state.startsWith('error'))) {
              clearInterval(poll);
              location.reload();
            }
          });
      }, 1200);
    })
    .catch(function(e){
      overlay.style.display = 'none';
      alert('Reprocess failed: ' + e);
    });
}
function reocr(page) {
  if (!confirm('Re-run OCR on page ' + page + ' using current polygons? (Segmentation preserved)')) return;
  var overlay = document.getElementById('reprocess-overlay');
  var msg = document.getElementById('reprocess-msg');
  var detail = document.getElementById('reprocess-detail');
  overlay.style.display = 'flex';
  msg.textContent = 'Re-OCR page ' + page + '…';
  detail.textContent = '';
  var session = '{{ session }}';
  var body = new URLSearchParams();
  body.set('session', session);
  body.set('page', page);
  fetch('/reocr', {method:'POST', body:body})
    .then(function(r){ return r.json(); })
    .then(function(){
      var poll = setInterval(function(){
        fetch('/status?session=' + encodeURIComponent(session))
          .then(function(r){ return r.json(); })
          .then(function(d){
            detail.textContent = d.state || '';
            if (d.state === 'done' || (d.state && d.state.startsWith('error'))) {
              clearInterval(poll);
              location.reload();
            }
          });
      }, 1200);
    })
    .catch(function(e){
      overlay.style.display = 'none';
      alert('Re-OCR failed: ' + e);
    });
}
function reocrAll() {
  if (!confirm('Re-run OCR on ALL pages using current polygons? (Segmentation preserved)')) return;
  var overlay = document.getElementById('reprocess-overlay');
  var msg = document.getElementById('reprocess-msg');
  var detail = document.getElementById('reprocess-detail');
  overlay.style.display = 'flex';
  msg.textContent = 'Re-OCR all pages…';
  detail.textContent = '';
  var session = '{{ session }}';
  var total = {{ total_pages }};
  var current = 1;
  function doNext() {
    if (current > total) { location.reload(); return; }
    detail.textContent = 'Page ' + current + ' / ' + total;
    var body = new URLSearchParams();
    body.set('session', session);
    body.set('page', current);
    fetch('/reocr', {method:'POST', body:body})
      .then(function(r){ return r.json(); })
      .then(function(){
        // Poll until this page is done
        var poll = setInterval(function(){
          fetch('/status?session=' + encodeURIComponent(session))
            .then(function(r){ return r.json(); })
            .then(function(d){
              detail.textContent = 'Page ' + current + ' / ' + total + ' — ' + (d.state || '');
              if (d.state === 'done' || (d.state && d.state.startsWith('error'))) {
                clearInterval(poll);
                current++;
                doNext();
              }
            });
        }, 800);
      })
      .catch(function(e){
        overlay.style.display = 'none';
        alert('Re-OCR failed on page ' + current + ': ' + e);
      });
  }
  doNext();
}
</script>
</body>
</html>
""")


# --- TRAIN (PAGINATED, single-click select, draggable handles, delete) ---
TRAIN_HTML = _jenv.from_string("""
<!doctype html>
<html>
<head><meta charset="utf-8"><title>Training</title>
{{ THEME_INIT | safe }}
<style>
  {{ BASE_CSS | safe }}
  /* Train-specific */
  .toggle-btn{ padding:.5rem 1rem; border:1px solid var(--border); border-radius:4px; background:var(--btn); color:var(--fg); cursor:pointer; }
  .toggle-btn:hover{ background:var(--btn-hover); border-color:var(--accent-gold); }
  .toggle-btn.active{ background:var(--btn-hover); border-color:var(--accent); box-shadow:0 0 0 2px rgba(154,120,32,.25); }
  .grid{ display:grid; grid-template-columns:2fr 3fr; gap:1.25rem; align-items:start; }
  /* Image/polygon panel stays visible while line list scrolls independently */
  .grid > .panel:first-child{ position:sticky; top:.5rem; align-self:start; }
  .grid > .panel[id^="lines-panel-"]{ max-height:85vh; overflow-y:auto; }
  .imgwrap{ max-height:80vh; overflow:auto; border:1px solid var(--border); border-radius:4px; position:relative; }
  .binimg{ display:block; max-width:100%; height:auto; position:relative; z-index:1; filter: var(--img-filter, none); }
  img.thumb{ filter: var(--img-filter, none); }
  .magnifier-view img{ filter: var(--img-filter, none); }
  .filter-toolbar{ margin:.25rem 0; font-size:.9rem; }
  .filter-toolbar summary{ cursor:pointer; color:var(--muted); }
  .filter-controls{ display:flex; flex-wrap:wrap; gap:.75rem; align-items:center; padding:.5rem; }
  .filter-controls label{ display:flex; align-items:center; gap:.35rem; }
  .overlay{ position:absolute; inset:0; width:100%; height:100%; z-index:2; }
  .linepoly{ fill:rgba(154,120,32,0.06); stroke:var(--accent-gold); stroke-width:3px; vector-effect:non-scaling-stroke; cursor:pointer; }
  .linepoly:hover{ fill:rgba(200,154,64,0.2); }
  .linepoly.selected{ fill:rgba(200,154,64,0.25); stroke:var(--accent); }
  .linepoly.highlighted{ stroke:var(--red); stroke-width:5px; fill:rgba(200,80,80,0.25); }
  .handle{ fill:var(--panel); stroke:var(--accent-gold); stroke-width:2px; cursor:move; vector-effect:non-scaling-stroke; }
  .lineitem{ display:flex; gap:.75rem; align-items:flex-start; margin-bottom:.75rem; border-bottom:1px dashed var(--border-soft); padding-bottom:.75rem; }
  .thumb{ max-width:160px; border:1px solid var(--border); border-radius:4px; }
  .edit-area{ flex:1; display:flex; flex-direction:column; }
  .magnifier-view{ display:none; height:100px; width:100%; margin-bottom:.75rem; padding:5px; background:var(--sunken); border:1px solid var(--border); border-radius:4px; box-sizing:border-box; }
  .magnifier-view img{ width:100%; height:100%; object-fit:contain; }
  textarea{ flex:1; width:100%; height:80px; padding:.5rem; }
  .savebtn{ padding:.4rem .8rem; }
  .count{ margin:.5rem 0 0; color:var(--muted); }
</style>
</head>
<body>
  <div id="app" class="app">

    <aside class="sidebar">
      <div class="sidebar-header">
        <button id="toggleSidebar" class="sidebar-btn" title="Toggle Sidebar">☰ <span class="label">Sidebar</span></button>
        <div class="sidebar-title" title="Session">{{ session }}</div>
      </div>
      <div class="thumbs">
        {% for p in all_pages %}
        <a class="thumb-link" href="/training?session={{ session }}&page={{ p.title }}">
          <div class="thumb-item {% if p.title == current_page %}active{% endif %}">
            <img class="thumb-img" src="/thumb/{{ session }}/{{ p.orig_image_name }}?w=120" alt="{{ p.title }} thumb" loading="lazy" decoding="async">
            <div class="thumb-meta">
              <div class="thumb-title">{{ p.title }}</div>
              <div class="thumb-sub">{{ p.line_count }} lines</div>
            </div>
          </div>
        </a>
        {% endfor %}
      </div>
    </aside>

    <main class="main">
      <div class="top">
        <form action="/set_script" method="post" id="script-form" style="display:flex;align-items:center;gap:.4rem;">
          <input type="hidden" name="session" value="{{ session }}">
          <input type="hidden" name="return_to" value="training">
          <input type="hidden" name="page" value="{{ current_page_num }}">
          <label class="small" for="script-input">Script:</label>
          <input id="script-input" list="scripts-list" name="script" required
                 pattern="[A-Za-z0-9_-]+" title="Letters, numbers, dashes, underscores only"
                 placeholder="e.g. Arabic_Kufic" autocomplete="off"
                 value="{{ current_script or '' }}"
                 style="width:11rem;">
          <datalist id="scripts-list">
            {% for s in scripts %}<option value="{{ s }}">{% endfor %}
          </datalist>
          <button type="submit">Save script</button>
          <span class="small" id="script-status"></span>
        </form>

        <form action="/export_final_gt" method="post" id="export-form">
          <input type="hidden" name="session" value="{{ session }}">
          <button type="submit" {% if not current_script %}disabled title="Set a script above first"{% endif %}>
            Export to Validation Set
          </button>
        </form>
        <span class="small" id="export-status" style="margin-left:-0.5rem;"></span>

        <form action="/view" method="get"><input type="hidden" name="session" value="{{ session }}"><input type="hidden" name="page" value="{{ page.title }}"><button type="submit">Back to Viewer</button></form>

        <button type="button" onclick="reprocess({{ current_page_num }})">Reprocess Page</button>
        <button type="button" onclick="reprocess(null)">Reprocess All</button>
        <button type="button" onclick="reocr({{ current_page_num }})">Re-OCR Page</button>

        <a href="/">New OCR</a>
        <a href="/models">Models</a>
        <span class="small">Session: {{ session }}</span>
        <span style="margin-left:auto;">{{ THEME_TOGGLE | safe }}</span>
      </div>

      <div style="display:flex; align-items:center; gap:1rem; margin:0.25rem 0 0.75rem 0;">
        {% if current_page_num > 1 %}
        <a href="/training?session={{ session }}&page={{ current_page_num - 1 }}" style="color:var(--accent);">← Prev</a>
        {% else %}
        <span style="color:var(--muted);">← Prev</span>
        {% endif %}
        <span class="small">Page {{ current_page_num }} / {{ total_pages }}</span>
        {% if current_page_num < total_pages %}
        <a href="/training?session={{ session }}&page={{ current_page_num + 1 }}" style="color:var(--accent);">Next →</a>
        {% else %}
        <span style="color:var(--muted);">Next →</span>
        {% endif %}
      </div>

      <div class="training-page" id="page-{{ page.title }}" style="display:block;">
        <h2>Page {{ page.title }}</h2>
        <div class="top" style="margin-bottom:1rem; justify-content:flex-start;">
          <button class="toggle-btn" id="edit-toggle-{{ page.title }}">Edit Polygons</button>
          <button class="save-btn" id="add-seg-{{ page.title }}" type="button" style="display:none;">Add Segment</button>
          <button class="save-btn" id="save-seg-{{ page.title }}" type="button" style="display:none;">Save Segmentation</button>
          <span class="small" id="save-status-{{ page.title }}" style="margin-left:1rem;"></span>
        </div>
        <details class="filter-toolbar" id="img-filters">
          <summary>Image filters (viewing only — never baked into exports)</summary>
          <div class="filter-controls">
            <label>Invert <input type="range" id="f-invert" min="0" max="1" step="1" value="0"></label>
            <label>Contrast <input type="range" id="f-contrast" min="50" max="300" step="5" value="100"><span id="f-contrast-v">100%</span></label>
            <label>Brightness <input type="range" id="f-brightness" min="50" max="200" step="5" value="100"><span id="f-brightness-v">100%</span></label>
            <label>Saturate <input type="range" id="f-saturate" min="0" max="300" step="10" value="100"><span id="f-saturate-v">100%</span></label>
            <label>Grayscale <input type="range" id="f-grayscale" min="0" max="100" step="10" value="0"><span id="f-grayscale-v">0%</span></label>
            <button type="button" id="f-reset">Reset</button>
          </div>
        </details>
        <div class="grid">
          <div class="panel">
            <div class="imgwrap" id="imgwrap-{{ page.title }}">
              <div class="img-stack" style="position:relative; display:block; width:100%;">
                <img class="binimg" id="img-{{ page.title }}"
                     src="/image/{{ session }}/{{ page.bin_image_name }}"
                     loading="lazy" decoding="async">
                <svg class="overlay" id="svg-{{ page.title }}" data-page-id="{{ page.title }}" viewBox="0 0 {{ page.img_w }} {{ page.img_h }}" preserveAspectRatio="xMinYMin meet">
                  {% for line in page.lines %}<polygon class="linepoly" data-line-id="{{ page.title }}-{{ line.idx }}" points="{{ line.points_str }}"></polygon>{% endfor %}
                </svg>
              </div>
            </div>
            <p class="count small">{{ page.lines|length }} lines detected. In Edit Mode, single-click a polygon to select and drag its vertices. Press 'Delete' to remove.</p>
          </div>
          <div class="panel" id="lines-panel-{{ page.title }}">
            {% for line in page.lines %}
            <div class="lineitem" id="{{ page.title }}-{{ line.idx }}">
              <img class="thumb" data-src="{% if line.thumb_missing %}/api/crop_line?session={{ session }}&page_stem={{ page.title }}&line_idx={{ line.idx }}{% else %}/image/{{ session }}/{{ line.thumb_name }}?v={{ render_ts }}{% endif %}" src="data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7" loading="lazy">
              <div class="edit-area">
                 <div class="magnifier-view">
                   <img data-src="{% if line.thumb_missing %}/api/crop_line?session={{ session }}&page_stem={{ page.title }}&line_idx={{ line.idx }}{% else %}/image/{{ session }}/{{ line.thumb_name }}?v={{ render_ts }}{% endif %}" src="data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7" loading="lazy" decoding="async" alt="Magnified line segment">
                 </div>
                  <form class="line-form" action="/save_line" method="post" style="display:flex;gap:.5rem;align-items:flex-start">
                    <input type="hidden" name="session" value="{{ session }}"><input type="hidden" name="gt_name" value="{{ line.gt_rel }}">
                    <textarea name="text" data-line-id="{{ page.title }}-{{ line.idx }}" class="line-text">{{ line.text }}</textarea>
                    <button class="savebtn" type="submit">Save</button>
                  </form>
              </div>
            </div>
            {% endfor %}
          </div>
        </div>
      </div>
      </main>
  </div> <script>
(function() {

    // --- 0. Lazy-load line item images via IntersectionObserver ---
    const lazyObserver = new IntersectionObserver((entries) => {
        entries.forEach(e => {
            if (e.isIntersecting) {
                const thumb = e.target.querySelector('img.thumb[data-src]');
                if (thumb) { thumb.src = thumb.dataset.src; delete thumb.dataset.src; }
                lazyObserver.unobserve(e.target);
            }
        });
    }, { rootMargin: '300px' });
    document.querySelectorAll('.lineitem').forEach(el => lazyObserver.observe(el));

    // --- 0b. Viewing-only image filters (shared localStorage key with /view) ---
    (function(){
      const KEY = 'kraken-img-filters';
      const ids = ['invert','contrast','brightness','saturate','grayscale'];
      const defaults = { invert:0, contrast:100, brightness:100, saturate:100, grayscale:0 };
      const saved = Object.assign({}, defaults, JSON.parse(localStorage.getItem(KEY) || '{}'));
      function compose(v){
        return `invert(${v.invert}) contrast(${v.contrast}%) brightness(${v.brightness}%) saturate(${v.saturate}%) grayscale(${v.grayscale}%)`;
      }
      function apply(v){
        document.documentElement.style.setProperty('--img-filter', compose(v));
        ids.forEach(k => {
          const el = document.getElementById(`f-${k}`); if (el) el.value = v[k];
          const lbl = document.getElementById(`f-${k}-v`); if (lbl) lbl.textContent = (k==='invert' ? (v[k]?'on':'off') : v[k]+'%');
        });
        localStorage.setItem(KEY, JSON.stringify(v));
      }
      apply(saved);
      ids.forEach(k => {
        const el = document.getElementById(`f-${k}`);
        if (!el) return;
        el.addEventListener('input', () => { saved[k] = Number(el.value); apply(saved); });
      });
      const reset = document.getElementById('f-reset');
      if (reset) reset.addEventListener('click', () => apply(Object.assign({}, defaults)));
    })();

    // --- 1. Export form script ---
    const exportForm = document.getElementById('export-form');
    const exportStatus = document.getElementById('export-status');
    if (exportForm) {
      exportForm.addEventListener('submit', async (e) => {
          e.preventDefault();
          exportStatus.textContent = 'Exporting...';
          try {
              const response = await fetch('/export_final_gt', { method: 'POST', body: new FormData(exportForm) });
              const text = await response.text();
              exportStatus.textContent = response.ok ? `✅ ${text}` : `❌ Error: ${text}`;
          } catch (err) {
              exportStatus.textContent = `❌ Network Error.`;
          }
      });
    }

    // --- 2. Polygon editing script ---
    function initPage(pageId) {
        const svg = document.getElementById(`svg-${pageId}`);
        if (!svg) { 
            console.warn(`Could not find SVG for page ${pageId}`);
            return; 
        }
        
        const linesPanel = document.getElementById(`lines-panel-${pageId}`);
        const toggleBtn = document.getElementById(`edit-toggle-${pageId}`);
        const saveBtn = document.getElementById(`save-seg-${pageId}`);
        const addBtn = document.getElementById(`add-seg-${pageId}`);
        const statusEl = document.getElementById(`save-status-${pageId}`);

        // O(1) focus tracking — no querySelectorAll scans
        let activeMagnifier = null, activePoly = null;

        linesPanel.addEventListener('focusin', (e) => {
            const ta = e.target.closest('textarea');
            if (!ta) return;
            const lineId = ta.dataset.lineId;
            const poly = document.querySelector(`.linepoly[data-line-id='${lineId}']`);
            const magnifier = ta.closest('.edit-area').querySelector('.magnifier-view');
            if (!poly || !magnifier) return;
            // Hide previous, show new (O(1) instead of O(N))
            if (activeMagnifier && activeMagnifier !== magnifier) activeMagnifier.style.display = 'none';
            magnifier.style.display = 'block';
            activeMagnifier = magnifier;
            // Lazy-load magnifier image on first focus
            const magImg = magnifier.querySelector('img[data-src]');
            if (magImg) { magImg.src = magImg.dataset.src; delete magImg.dataset.src; }
            if (activePoly && activePoly !== poly) activePoly.classList.remove('highlighted');
            poly.classList.add('highlighted');
            activePoly = poly;
        });
        linesPanel.addEventListener('focusout', () => {
            // Defer so focusin on the next textarea fires first
            setTimeout(() => {
                if (activePoly && !linesPanel.contains(document.activeElement)) {
                    activePoly.classList.remove('highlighted');
                    activePoly = null;
                    if (activeMagnifier) { activeMagnifier.style.display = 'none'; activeMagnifier = null; }
                }
            }, 0);
        });

        let isEditMode = false, selectedPoly = null, handles = [];
        function cleanupHandles() { handles.forEach(h => h.remove()); handles = []; }
        function deselectAll() {
            svg.querySelectorAll('.linepoly.selected').forEach(p => p.classList.remove('selected'));
            cleanupHandles(); selectedPoly = null;
        }
        function selectPolygon(poly) {
            if (!isEditMode) return;
            deselectAll(); selectedPoly = poly; poly.classList.add('selected'); createHandles(poly);
        }
        function createHandles(poly) {
            cleanupHandles();
            const points = poly.getAttribute('points').trim().split(/\\s+/).map(p => p.split(',').map(parseFloat));
            points.forEach((point, i) => {
                const handle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                handle.setAttribute('cx', point[0]);
                handle.setAttribute('cy', point[1]);
                handle.setAttribute('r', '8');
                handle.setAttribute('class', 'handle');
                handle.dataset.index = i;
                svg.appendChild(handle);
                handles.push(handle);
                handle.addEventListener('mousedown', (e) => {
                    e.stopPropagation();
                    const index = parseInt(handle.dataset.index, 10);
                    function onMouseMove(moveEvent) {
                        const rect = svg.getBoundingClientRect();
                        const vb   = svg.viewBox.baseVal;
                        const svgP = {
                            x: (moveEvent.clientX - rect.left) * (vb.width  / rect.width),
                            y: (moveEvent.clientY - rect.top)  * (vb.height / rect.height)
                        };
                        handle.setAttribute('cx', svgP.x);
                        handle.setAttribute('cy', svgP.y);
                        const arr = selectedPoly.getAttribute('points').trim().split(/\\s+/);
                        arr[index] = `${svgP.x},${svgP.y}`;
                        selectedPoly.setAttribute('points', arr.join(' '));
                    }
                    function onMouseUp() {
                        document.removeEventListener('mousemove', onMouseMove);
                        document.removeEventListener('mouseup', onMouseUp);
                    }
                    document.addEventListener('mousemove', onMouseMove);
                    document.addEventListener('mouseup', onMouseUp);
                });
            });
        }
        function deleteSelectedPolygon() {
            if (isEditMode && selectedPoly) {
                const lineId = selectedPoly.dataset.lineId;
                document.getElementById(lineId)?.remove();
                selectedPoly.remove();
                cleanupHandles();
                selectedPoly = null;
            }
        }
        toggleBtn.addEventListener('click', () => {
            isEditMode = !isEditMode;
            toggleBtn.classList.toggle('active', isEditMode);
            saveBtn.style.display = isEditMode ? 'inline-block' : 'none';
            if (addBtn) addBtn.style.display = isEditMode ? 'inline-block' : 'none';
            if (!isEditMode) deselectAll();
        });

        // Add Segment: inserts a new rectangular polygon the user can reshape then save
        if (addBtn) {
            addBtn.addEventListener('click', () => {
                if (!isEditMode) return;
                const vb = svg.viewBox.baseVal;
                const w = vb.width, h = vb.height;
                const pw = Math.max(40, w * 0.30);
                const ph = Math.max(20, h * 0.04);
                const cx = w * 0.50, cy = h * 0.50;
                const x0 = cx - pw / 2, y0 = cy - ph / 2;
                const x1 = cx + pw / 2, y1 = cy + ph / 2;
                const points = `${x0},${y0} ${x1},${y0} ${x1},${y1} ${x0},${y1}`;
                const poly = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
                poly.setAttribute('class', 'linepoly');
                poly.setAttribute('points', points);
                poly.dataset.lineId = `${pageId}-new-${Date.now()}`;
                svg.appendChild(poly);
                selectPolygon(poly);
                statusEl.textContent = 'New segment added — drag handles to reshape, then Save Segmentation.';
            });
        }
        svg.addEventListener('click', (e) => {
            if (isEditMode && e.target.classList.contains('linepoly')) {
                e.stopPropagation();
                selectPolygon(e.target);
            } else if (isEditMode && e.target === svg) {
                deselectAll();
            }
        });
        document.addEventListener('keydown', (e) => {
            if ((e.key === 'Delete' || e.key === 'Backspace') && selectedPoly) {
                e.preventDefault();
                deleteSelectedPolygon();
            }
        });

        // Save Segmentation Button
        saveBtn.addEventListener('click', async () => {
            try {
                statusEl.textContent = 'Saving…';
                const polygons = Array.from(svg.querySelectorAll('.linepoly')).map(p =>
                    p.getAttribute('points').trim().split(/\\s+/).map(pair => pair.split(',').map(Number))
                );
                const currentTexts = Array.from(document.querySelectorAll(`#lines-panel-${pageId} textarea`)).map(ta => ta.value);
                // Note: {{ page.title }} is available from the template
                const payload = { session: "{{ session }}", page_id: "{{ page.title }}", polygons, texts: currentTexts };

                const resp = await fetch('/update_segmentation', {
                    method: 'POST', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const txt = await resp.text();
                if (!resp.ok) {
                    console.error('Save seg error:', resp.status, txt);
                    statusEl.textContent = `Error: ${txt}`;
                    alert(`Save failed: ${txt}`);
                    return;
                }
                statusEl.textContent = 'Saved successfully! Refreshing…';
                // This reload is now *fast* because of pagination
                setTimeout(() => { location.reload(); }, 1200);
            } catch (err) {
                console.error('Save seg exception:', err);
                statusEl.textContent = 'Network error.';
                alert('Network error while saving segmentation.');
            }
        });
    }
    
    // --- 3. /save_line — delegated event handling + autosave ---
    // Initialize original text tracking for all textareas (one-time, no listeners)
    document.querySelectorAll('form[action="/save_line"]').forEach(form => {
        const ta = form.querySelector('.line-text');
        const btn = form.querySelector('.savebtn');
        if (ta) ta.dataset.originalText = ta.value;
        if (btn) btn.disabled = true;
    });

    // Tiny "Saved" toast
    function flashSaved() {
        let el = document.getElementById('autosave-toast');
        if (!el) {
            el = document.createElement('div');
            el.id = 'autosave-toast';
            Object.assign(el.style, {position:'fixed',right:'12px',bottom:'12px',padding:'8px 12px',background:'rgba(0,0,0,0.7)',color:'#fff',borderRadius:'8px',fontSize:'12px',zIndex:'9999'});
            document.body.appendChild(el);
        }
        el.textContent = 'Saved';
        el.style.opacity = '1';
        setTimeout(() => { el.style.transition = 'opacity 600ms'; el.style.opacity = '0'; }, 600);
    }

    async function autosaveForm(form) {
        if (!form._dirty) return true;
        const fd = new FormData(form);
        try {
            await fetch('/save_line', { method: 'POST', body: fd, headers: {'X-Requested-With':'fetch','Accept':'application/json'}, keepalive: true, redirect: 'manual' });
            form._dirty = false;
            const btn = form.querySelector('.savebtn');
            const ta = form.querySelector('.line-text');
            if (ta) ta.dataset.originalText = ta.value;
            if (btn) { btn.textContent = 'Saved!'; btn.disabled = true; setTimeout(() => { btn.textContent = 'Save'; }, 1500); }
            flashSaved();
            return true;
        } catch (e) { console.warn('Autosave failed:', e); return false; }
    }

    // Delegated input handler — updates button state + debounced autosave
    document.body.addEventListener('input', (e) => {
        const ta = e.target.closest('textarea.line-text');
        if (!ta) return;
        const form = ta.closest('form[action="/save_line"]');
        if (!form) return;
        const btn = form.querySelector('.savebtn');
        if (btn) btn.disabled = (ta.value === ta.dataset.originalText);
        form._dirty = true;
        if (form._debounce) clearTimeout(form._debounce);
        form._debounce = setTimeout(() => autosaveForm(form), 800);
    });

    // Delegated submit handler
    document.body.addEventListener('submit', async (e) => {
        const form = e.target.closest('form[action="/save_line"]');
        if (!form) return;
        e.preventDefault();
        form._dirty = true; // ensure autosave runs even if just clicking Save
        await autosaveForm(form);
    });

    // Save dirty forms before internal navigation
    function getAllLineForms() { return Array.from(document.querySelectorAll('form[action="/save_line"]')); }
    function isInternalNav(a) {
        if (!a || !a.href) return false;
        try { const u = new URL(a.href, location.origin); return (u.pathname === '/training' || u.pathname === '/view') && u.searchParams.has('session'); }
        catch { return false; }
    }
    document.addEventListener('click', async (e) => {
        const a = e.target.closest('a');
        if (!a || !isInternalNav(a)) return;
        const dirty = getAllLineForms().filter(f => f._dirty);
        if (!dirty.length) return;
        e.preventDefault();
        await Promise.all(dirty.map(f => autosaveForm(f)));
        location.href = a.href;
    });

    // Beacon save on tab close
    function beaconAutosaveAll() {
        for (const form of getAllLineForms().filter(f => f._dirty)) {
            const params = new URLSearchParams();
            params.set('session', form.querySelector('input[name="session"]')?.value || '');
            params.set('gt_name', form.querySelector('input[name="gt_name"]')?.value || '');
            params.set('text', form.querySelector('textarea[name="text"]')?.value || '');
            const blob = new Blob([params.toString()], { type: 'application/x-www-form-urlencoded;charset=UTF-8' });
            if (!(navigator.sendBeacon && navigator.sendBeacon('/save_line', blob))) {
                fetch('/save_line', { method: 'POST', body: params, headers: {'X-Requested-With':'fetch','Accept':'application/json'}, keepalive: true, redirect: 'manual' });
            }
            form._dirty = false;
        }
    }
    window.addEventListener('pagehide', beaconAutosaveAll, { capture: true });
    window.addEventListener('beforeunload', beaconAutosaveAll, { capture: true });

    // --- 4. Sidebar Toggle script ---
    const app = document.getElementById('app');
    const toggleSidebarBtn = document.getElementById('toggleSidebar');
    if (app && toggleSidebarBtn) {
      const savedCollapsed = localStorage.getItem('viewerSidebarCollapsed') === '1';
      if (savedCollapsed) app.classList.add('collapsed');

      toggleSidebarBtn.addEventListener('click', () => {
          app.classList.toggle('collapsed');
          localStorage.setItem('viewerSidebarCollapsed', app.classList.contains('collapsed') ? '1' : '0');
      });
    }

    // --- 5. NEW: Initialize the *single* active page ---
    initPage("{{ page.title }}");

})();
</script>
<script>
/** Simple virtualizer for a vertical list of .thumb-item inside .thumbs */
(function(){
  const container = document.querySelector('.thumbs');
  if(!container) return;
  const ROW_H = 64;            // approx row height
  const BUFFER = 20;           // rows above/below viewport
  const items = Array.from(container.children);
  const total = items.length;
  if (total === 0) return;

  // Wrap originals into a shadow list
  const shadow = document.createDocumentFragment();
  items.forEach(el => shadow.appendChild(el));
  container.innerHTML = "";

  const spacerTop = document.createElement('div');
  const viewport = document.createElement('div');
  const spacerBot = document.createElement('div');
  container.append(spacerTop, viewport, spacerBot);

  function render(){
    const vh = container.clientHeight || window.innerHeight;
    const scrollTop = container.scrollTop;
    const start = Math.max(0, Math.floor(scrollTop / ROW_H) - BUFFER);
    const end = Math.min(total, Math.ceil((scrollTop + vh) / ROW_H) + BUFFER);
    spacerTop.style.height = (start * ROW_H) + "px";
    spacerBot.style.height = ((total - end) * ROW_H) + "px";
    viewport.innerHTML = "";
    for(let i=start; i<end; i++) viewport.appendChild(items[i]);
  }

  container.style.overflowY = 'auto';
  container.style.maxHeight = 'calc(100vh - 56px)';
  container.addEventListener('scroll', () => requestAnimationFrame(render), {passive:true});
  spacerTop.style.height = "0px";
  spacerBot.style.height = (total * ROW_H) + "px";
  render();
})();
</script>

<div id="reprocess-overlay" style="display:none;position:fixed;inset:0;z-index:9999;background:rgba(0,0,0,.55);backdrop-filter:blur(3px);justify-content:center;align-items:center;">
  <div style="background:var(--panel,#1a1a1a);border:1px solid var(--border,#333);border-radius:10px;padding:2rem 2.5rem;text-align:center;box-shadow:0 8px 32px rgba(0,0,0,.4);">
    <div style="margin-bottom:1rem;">
      <svg width="40" height="40" viewBox="0 0 40 40" style="animation:reproc-spin 1s linear infinite;">
        <circle cx="20" cy="20" r="16" fill="none" stroke="var(--border,#555)" stroke-width="3"/>
        <circle cx="20" cy="20" r="16" fill="none" stroke="var(--accent-gold,#9a7820)" stroke-width="3" stroke-dasharray="80" stroke-dashoffset="60" stroke-linecap="round"/>
      </svg>
    </div>
    <div id="reprocess-msg" style="font-size:1.05rem;color:var(--fg,#eee);">Reprocessing…</div>
    <div id="reprocess-detail" style="font-size:.85rem;color:var(--muted,#888);margin-top:.4rem;"></div>
  </div>
</div>
<style>@keyframes reproc-spin{to{transform:rotate(360deg)}}</style>
<script>
document.addEventListener('click', function(e) {
  document.querySelectorAll('.dropdown-menu.open').forEach(function(m) {
    if (!m.parentElement.contains(e.target)) m.classList.remove('open');
  });
});
function reprocess(page) {
  var label = page ? 'page ' + page : 'all pages';
  if (!confirm('Re-run segmentation and OCR on ' + label + '?')) return;
  var overlay = document.getElementById('reprocess-overlay');
  var msg = document.getElementById('reprocess-msg');
  var detail = document.getElementById('reprocess-detail');
  overlay.style.display = 'flex';
  msg.textContent = 'Reprocessing ' + label + '…';
  detail.textContent = '';
  var session = '{{ session }}';
  var body = new URLSearchParams();
  body.set('session', session);
  if (page) body.set('page', page);
  fetch('/reprocess', {method:'POST', body:body})
    .then(function(r){ return r.json(); })
    .then(function(){
      var poll = setInterval(function(){
        fetch('/status?session=' + encodeURIComponent(session))
          .then(function(r){ return r.json(); })
          .then(function(d){
            detail.textContent = d.state || '';
            if (d.state === 'done' || (d.state && d.state.startsWith('error'))) {
              clearInterval(poll);
              location.reload();
            }
          });
      }, 1200);
    })
    .catch(function(e){
      overlay.style.display = 'none';
      alert('Reprocess failed: ' + e);
    });
}
function reocr(page) {
  if (!confirm('Re-run OCR on page ' + page + ' using current polygons? (Segmentation preserved)')) return;
  var overlay = document.getElementById('reprocess-overlay');
  var msg = document.getElementById('reprocess-msg');
  var detail = document.getElementById('reprocess-detail');
  overlay.style.display = 'flex';
  msg.textContent = 'Re-OCR page ' + page + '…';
  detail.textContent = '';
  var session = '{{ session }}';
  var body = new URLSearchParams();
  body.set('session', session);
  body.set('page', page);
  fetch('/reocr', {method:'POST', body:body})
    .then(function(r){ return r.json(); })
    .then(function(){
      var poll = setInterval(function(){
        fetch('/status?session=' + encodeURIComponent(session))
          .then(function(r){ return r.json(); })
          .then(function(d){
            detail.textContent = d.state || '';
            if (d.state === 'done' || (d.state && d.state.startsWith('error'))) {
              clearInterval(poll);
              location.reload();
            }
          });
      }, 1200);
    })
    .catch(function(e){
      overlay.style.display = 'none';
      alert('Re-OCR failed: ' + e);
    });
}
</script>
</body>
</html>
""")

# ---------------- Routes ----------------

# --- Background OCR Job ---
def _process_job(session: str, model_path: str, dpi: int, reorder: str, upload_path: Path, scale: float = 0.55, preprocess: str = "grayscale", deskew: str = "no", merge_lines: str = "no", segment_only: str = "no", seg_pad: int = 20):
    sess=SESSIONS_BASE_DIR/session; pages_dir=sess/"pages"; bin_dir=sess/"bin"; out_dir=sess/"out"
    gt_prev = sess / "gt_preview"
    for d in (pages_dir,bin_dir,out_dir,gt_prev): d.mkdir(parents=True,exist_ok=True)
    (sess/"model.txt").write_text(model_path,encoding="utf-8")
    (sess/"scale.txt").write_text(str(scale),encoding="utf-8")

    src_path = pages_dir / f"source_upload{upload_path.suffix}"
    try:
        shutil.move(upload_path, src_path)
    except Exception as e:
        PROGRESS[session]["state"] = f"error: failed to move upload: {e}"
        upload_path.unlink(missing_ok=True)
        return

    try:
        if is_pdf(str(src_path)):
            PROGRESS[session]["state"]=f"pdftoppm {src_path.name}";
            run(["pdftoppm","-png","-r",str(dpi),str(src_path),str(pages_dir/"page")]); src_path.unlink(missing_ok=True)
        elif not src_path.name.startswith("page-"):
            # Ensure it has a page-like name for consistency
            dest=pages_dir/f"page-1{src_path.suffix.lower()}"; shutil.move(src_path,dest)
    except Exception as e: PROGRESS[session]["state"]=f"error: {e}"; return

    page_imgs=sorted([p for p in pages_dir.iterdir() if p.suffix.lower() in (".png",".jpg",".jpeg",".tif",".tiff") and p.stem.startswith("page-")],key=page_number)
    PROGRESS[session]["total"]=len(page_imgs)
    if not page_imgs: PROGRESS[session]["state"]="error: no pages"; return

    for idx,p in enumerate(page_imgs,1):
        base=p.stem; bin_img=bin_dir/f"{base}.png"; out_txt=out_dir/f"{base}.txt"; seg_json=pages_dir/f"{base}.json"
        PROGRESS[session]["current_image"] = p.name

        # --- Phase 1: Preprocess, segment, OCR ---
        try:
            if preprocess == "binarize":
                PROGRESS[session]["state"]=f"binarize {p.name}"; kraken_binarize(p, bin_img)
            elif preprocess == "grayscale":
                PROGRESS[session]["state"]=f"grayscale {p.name}"; Image.open(p).convert("L").save(bin_img)
            else:
                PROGRESS[session]["state"]=f"copy {p.name}"; shutil.copy(p, bin_img)
            # --- Deskew (optional, after preprocess, before segmentation) ---
            if deskew == "yes" and cv2 is not None:
                PROGRESS[session]["state"]=f"deskew {p.name}"
                _deskew_image(bin_img)
            # --- Segment (+ OCR unless segment_only) ---
            if segment_only == "yes":
                PROGRESS[session]["state"]=f"segment-only {p.name}"
                seg_data = kraken_segment(bin_img)
                with open(seg_json, "w", encoding="utf-8") as _f:
                    json.dump(seg_data, _f, indent=2)
                # Retry with padding if close-up detection looks bad
                _im_tmp = Image.open(bin_img); _img_w = _im_tmp.size[0]; _im_tmp.close()
                if _seg_needs_retry(seg_data, _img_w):
                    PROGRESS[session]["state"]=f"retry-padded {p.name}"
                    for _mult in (4, 3, 6):
                        _padded = None
                        try:
                            _padded, _pad_top = _pad_image_for_segmentation(bin_img, _mult)
                            seg_data = kraken_segment(_padded)
                            if not _seg_needs_retry(seg_data, _img_w):
                                with open(seg_json, "w", encoding="utf-8") as _f:
                                    json.dump(seg_data, _f, indent=2)
                                _adjust_seg_json_for_padding(seg_json, _pad_top, str(bin_img))
                                break
                        except Exception:
                            pass
                        finally:
                            if _padded and _padded.exists():
                                _padded.unlink(missing_ok=True)
            else:
                PROGRESS[session]["state"]=f"segment+ocr {p.name}"
                seg_data, text_lines = kraken_segment_and_ocr(
                    bin_img, model_path, bidi_reordering=(reorder == "yes"))
                # Save seg JSON
                with open(seg_json, "w", encoding="utf-8") as _f:
                    json.dump(seg_data, _f, indent=2)
                # Save text
                out_txt.write_text("\n".join(text_lines) + "\n", encoding="utf-8")
                # --- Retry with padding if close-up detection looks bad ---
                _im_tmp = Image.open(bin_img); _img_w = _im_tmp.size[0]; _im_tmp.close()
                if _seg_needs_retry(seg_data, _img_w):
                    PROGRESS[session]["state"]=f"retry-padded {p.name}"
                    for _mult in (4, 3, 6):
                        _padded = None
                        try:
                            _padded, _pad_top = _pad_image_for_segmentation(bin_img, _mult)
                            seg_data, text_lines = kraken_segment_and_ocr(
                                _padded, model_path, bidi_reordering=(reorder == "yes"))
                            if not _seg_needs_retry(seg_data, _img_w):
                                with open(seg_json, "w", encoding="utf-8") as _f:
                                    json.dump(seg_data, _f, indent=2)
                                _adjust_seg_json_for_padding(seg_json, _pad_top, str(bin_img))
                                out_txt.write_text("\n".join(text_lines) + "\n", encoding="utf-8")
                                break
                        except Exception:
                            pass
                        finally:
                            if _padded and _padded.exists():
                                _padded.unlink(missing_ok=True)
            # --- Merge split lines (optional): merge JSON + rewrite text ---
            did_merge = False
            if merge_lines == "yes" and seg_json.exists():
                try:
                    with open(seg_json, "r", encoding="utf-8") as _f:
                        seg_data = json.load(_f)
                    raw_lines = seg_data.get("lines") if isinstance(seg_data.get("lines"), list) else _find_lines(seg_data)
                    merged, merge_groups = _merge_split_lines(raw_lines)
                    if len(merged) < len(raw_lines):
                        seg_data["lines"] = merged
                        with open(seg_json, "w", encoding="utf-8") as _f:
                            json.dump(seg_data, _f, indent=2)
                        # Rebuild out_txt: combine text lines that were merged
                        if out_txt.exists():
                            text_lines = out_txt.read_text("utf-8").splitlines()
                            merged_text = []
                            for group in merge_groups:
                                parts = [text_lines[i] for i in group if i < len(text_lines)]
                                merged_text.append(" ".join(parts))
                            out_txt.write_text("\n".join(merged_text) + "\n", encoding="utf-8")
                        did_merge = True
                except Exception as me:
                    PROGRESS[session]["errors"].append(f"merge-lines {p.name}: {me!s}")
        except Exception as e:
            PROGRESS[session]["errors"].append(f"{p.name}: {e!s}")
            PROGRESS[session]["done_pages"]=idx
            continue

        # --- Clip oversized last-line polygons ---
        if seg_json.exists():
            _im_clip = Image.open(bin_img); _clip_h = _im_clip.size[1]; _im_clip.close()
            _clip_seg_polygons(seg_json, _clip_h)

        # --- Phase 2: Crop line previews for training ---
        try:
            if not seg_json.exists():
                print(f"Skipping gt-preview for {base}, no seg file")
                continue

            PROGRESS[session]["state"]=f"cropping lines {p.name}"
            with open(seg_json, "r", encoding="utf-8") as f: seg = json.load(f)
            if isinstance(seg.get("lines"), list):
                lines = [l for l in seg["lines"] if isinstance(l, dict)]
            else:
                lines = _find_lines(seg)
            im = Image.open(bin_img); W,H = im.size
            full_page_text_lines = []
            if out_txt.exists():
                full_page_text_lines = out_txt.read_text("utf-8").splitlines()

            for i, line in enumerate(lines, 1):
                poly = _poly_from_line(line)
                if not poly: continue

                x0,y0,x1,y1 = _bbox(poly, 3)
                x0=max(0,min(x0,W-1)); y0=max(0,min(y0,H-1)); x1=max(x0+1,min(x1,W)); y1=max(y0+1,min(y1,H))
                if (x1 - x0) < 3 or (y1 - y0) < 3: continue

                thumb = gt_prev / f"{base}_{i:04d}.png"; gt = thumb.with_suffix(".gt.txt")
                if not thumb.exists():
                    try: im.crop((x0, y0, x1, y1)).save(thumb)
                    except Exception as e:
                        print(f"Failed to crop line {i} for {base}: {e}")
                        continue

                if not gt.exists():
                    # Use the corresponding text line from out_txt
                    current_text = ""
                    if (i - 1) < len(full_page_text_lines):
                        current_text = full_page_text_lines[i-1]
                    gt.write_text(current_text, encoding="utf-8")

            im.close()

        except Exception as e:
            PROGRESS[session]["errors"].append(f"gt-preview failed for {p.name}: {e!s}")

        PROGRESS[session]["done_pages"]=idx

    PROGRESS[session]["state"]="done"

# --- GET / — Home ---
@app.get("/", response_class=HTMLResponse)
def index():
    models = list_models()
    existing_sessions = []
    if SESSIONS_BASE_DIR.exists():
        for item in SESSIONS_BASE_DIR.iterdir():
            if item.is_dir() and item.name.startswith("hebrocr_"):
                 if (item / "pages").exists():
                      existing_sessions.append(item.name)
    existing_sessions.sort(reverse=True)
    return INDEX_HTML.render(
        models=models,
        selected=(models[0] if models else DEFAULT_MODEL),
        userhome=str(Path.home()),
        sessions=existing_sessions,
    )

# --- POST /start_ocr — Upload file and launch OCR job ---
@app.post("/start_ocr", response_class=HTMLResponse)
async def start_ocr(
    file: UploadFile=File(...),
    model_path: str=Form(DEFAULT_MODEL),
    reorder: str=Form("yes"),
    preprocess: str=Form("grayscale"),
    deskew: str=Form("no"),
    merge_lines: str=Form("no"),
    project_name: Optional[str]=Form(None),
    obj_id: Optional[str]=Form(None),
    segment_only: str=Form("no"),
    seg_pad: int=Form(20),
):
    # session: hebrocr_<project>_<uuid4short>
    project = (project_name or "proj").strip().replace(" ", "-")
    session = f"hebrocr_{project}_{str(uuid.uuid4())[:8]}"
    (SESSIONS_BASE_DIR / session).mkdir()
    if obj_id and obj_id.strip():
        (SESSIONS_BASE_DIR / session / "obj_id.txt").write_text(obj_id.strip(), encoding="utf-8")
    (SESSIONS_BASE_DIR / session / "preprocess.txt").write_text(preprocess, encoding="utf-8")
    (SESSIONS_BASE_DIR / session / "deskew.txt").write_text(deskew, encoding="utf-8")
    (SESSIONS_BASE_DIR / session / "merge_lines.txt").write_text(merge_lines, encoding="utf-8")
    (SESSIONS_BASE_DIR / session / "segment_only.txt").write_text(segment_only, encoding="utf-8")
    (SESSIONS_BASE_DIR / session / "seg_pad.txt").write_text(str(seg_pad), encoding="utf-8")
    PROGRESS[session] = {"total":0, "done_pages":0, "state":"upload", "errors":[]}
    upload_tmp = (SESSIONS_BASE_DIR / session / f"upload_tmp_{uuid.uuid4()}{Path(file.filename).suffix}")

    try:
        with open(upload_tmp, "wb") as f:
            while chunk := await file.read(1024*1024): f.write(chunk)
    except Exception as e:
        PROGRESS[session]["state"] = f"error: failed to write upload: {e}"
        return PROGRESS_HTML.render(session=session)

    threading.Thread(target=_process_job, args=(session,model_path,DEFAULT_DPI,reorder,upload_tmp,0.55,preprocess,deskew,merge_lines,segment_only,seg_pad), daemon=True).start()
    return PROGRESS_HTML.render(session=session)

# --- POST /reprocess — Re-run segmentation+OCR on existing session images ---
@app.post("/reprocess")
def reprocess(session: str = Form(...), page: Optional[int] = Form(None)):
    if "/" in session or ".." in session:
        return PlainTextResponse("Invalid session name.", status_code=400)
    sess = SESSIONS_BASE_DIR / session
    if not sess.exists():
        return PlainTextResponse("Session not found.", status_code=404)
    if session in PROGRESS and PROGRESS[session].get("state") not in ("done", None):
        return PlainTextResponse("A job is already running for this session.", status_code=409)
    model_path = (sess / "model.txt").read_text().strip() if (sess / "model.txt").exists() else DEFAULT_MODEL
    try:
        saved_scale = float((sess / "scale.txt").read_text().strip())
    except Exception:
        saved_scale = 0.55
    preprocess = (sess / "preprocess.txt").read_text().strip() if (sess / "preprocess.txt").exists() else "grayscale"
    deskew = (sess / "deskew.txt").read_text().strip() if (sess / "deskew.txt").exists() else "no"
    merge_lines = (sess / "merge_lines.txt").read_text().strip() if (sess / "merge_lines.txt").exists() else "no"
    segment_only = (sess / "segment_only.txt").read_text().strip() if (sess / "segment_only.txt").exists() else "no"
    try:
        seg_pad = int((sess / "seg_pad.txt").read_text().strip())
    except Exception:
        seg_pad = 20
    reorder = "yes"
    PROGRESS[session] = {"total": 0, "done_pages": 0, "state": "reprocess", "errors": []}
    threading.Thread(
        target=_reprocess_job,
        args=(session, model_path, saved_scale, preprocess, deskew, merge_lines, segment_only, reorder, page, seg_pad),
        daemon=True,
    ).start()
    return JSONResponse({"ok": True, "session": session})

def _reprocess_job(session: str, model_path: str, scale: float, preprocess: str, deskew: str, merge_lines: str, segment_only: str, reorder: str, only_page: Optional[int] = None, seg_pad: int = 20):
    sess = SESSIONS_BASE_DIR / session
    pages_dir = sess / "pages"
    bin_dir = sess / "bin"
    out_dir = sess / "out"
    gt_prev = sess / "gt_preview"
    for d in (bin_dir, out_dir, gt_prev):
        d.mkdir(parents=True, exist_ok=True)
    image_exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    all_page_imgs = sorted(
        [p for p in pages_dir.iterdir() if p.suffix.lower() in image_exts and p.stem.startswith("page-")],
        key=page_number,
    )

    if only_page is not None and 1 <= only_page <= len(all_page_imgs):
        page_imgs = [all_page_imgs[only_page - 1]]
    else:
        page_imgs = all_page_imgs

    # Clean old outputs for targeted pages
    for p in page_imgs:
        base = p.stem
        (bin_dir / f"{base}.png").unlink(missing_ok=True)
        (out_dir / f"{base}.txt").unlink(missing_ok=True)
        (pages_dir / f"{base}.json").unlink(missing_ok=True)
        for f in gt_prev.glob(f"{base}_*"):
            f.unlink(missing_ok=True)
        for f in sess.glob(f"{base}_orig*"):
            f.unlink(missing_ok=True)
        for f in sess.glob(f"{base}_bin*"):
            f.unlink(missing_ok=True)
    (sess / "sidebar_cache.json").unlink(missing_ok=True)
    PROGRESS[session]["total"] = len(page_imgs)
    if not page_imgs:
        PROGRESS[session]["state"] = "error: no pages"
        return

    for idx, p in enumerate(page_imgs, 1):
        base = p.stem
        bin_img = bin_dir / f"{base}.png"
        out_txt = out_dir / f"{base}.txt"
        seg_json = pages_dir / f"{base}.json"
        PROGRESS[session]["current_image"] = p.name

        try:
            if preprocess == "binarize":
                PROGRESS[session]["state"] = f"binarize {p.name}"
                kraken_binarize(p, bin_img)
            elif preprocess == "grayscale":
                PROGRESS[session]["state"] = f"grayscale {p.name}"
                Image.open(p).convert("L").save(bin_img)
            else:
                PROGRESS[session]["state"] = f"copy {p.name}"
                shutil.copy(p, bin_img)

            if deskew == "yes" and cv2 is not None:
                PROGRESS[session]["state"] = f"deskew {p.name}"
                _deskew_image(bin_img)

            if segment_only == "yes":
                PROGRESS[session]["state"] = f"segment-only {p.name}"
                seg_data = kraken_segment(bin_img)
                with open(seg_json, "w", encoding="utf-8") as _f:
                    json.dump(seg_data, _f, indent=2)
                _im_tmp = Image.open(bin_img); _img_w = _im_tmp.size[0]; _im_tmp.close()
                if _seg_needs_retry(seg_data, _img_w):
                    PROGRESS[session]["state"] = f"retry-padded {p.name}"
                    for _mult in (4, 3, 6):
                        _padded = None
                        try:
                            _padded, _pad_top = _pad_image_for_segmentation(bin_img, _mult)
                            seg_data = kraken_segment(_padded)
                            if not _seg_needs_retry(seg_data, _img_w):
                                with open(seg_json, "w", encoding="utf-8") as _f:
                                    json.dump(seg_data, _f, indent=2)
                                _adjust_seg_json_for_padding(seg_json, _pad_top, str(bin_img))
                                break
                        except Exception:
                            pass
                        finally:
                                if _padded and _padded.exists():
                                    _padded.unlink(missing_ok=True)
            else:
                PROGRESS[session]["state"] = f"segment+ocr {p.name}"
                seg_data, text_lines = kraken_segment_and_ocr(
                    bin_img, model_path, bidi_reordering=(reorder == "yes"))
                with open(seg_json, "w", encoding="utf-8") as _f:
                    json.dump(seg_data, _f, indent=2)
                out_txt.write_text("\n".join(text_lines) + "\n", encoding="utf-8")
                _im_tmp = Image.open(bin_img); _img_w = _im_tmp.size[0]; _im_tmp.close()
                if _seg_needs_retry(seg_data, _img_w):
                    PROGRESS[session]["state"] = f"retry-padded {p.name}"
                    for _mult in (4, 3, 6):
                        _padded = None
                        try:
                            _padded, _pad_top = _pad_image_for_segmentation(bin_img, _mult)
                            seg_data, text_lines = kraken_segment_and_ocr(
                                _padded, model_path, bidi_reordering=(reorder == "yes"))
                            if not _seg_needs_retry(seg_data, _img_w):
                                with open(seg_json, "w", encoding="utf-8") as _f:
                                    json.dump(seg_data, _f, indent=2)
                                _adjust_seg_json_for_padding(seg_json, _pad_top, str(bin_img))
                                out_txt.write_text("\n".join(text_lines) + "\n", encoding="utf-8")
                                break
                        except Exception:
                            pass
                        finally:
                            if _padded and _padded.exists():
                                _padded.unlink(missing_ok=True)

            if merge_lines == "yes" and seg_json.exists():
                try:
                    with open(seg_json, "r", encoding="utf-8") as _f:
                        seg_data = json.load(_f)
                    raw_lines = seg_data.get("lines") if isinstance(seg_data.get("lines"), list) else _find_lines(seg_data)
                    merged, merge_groups = _merge_split_lines(raw_lines)
                    if len(merged) < len(raw_lines):
                        seg_data["lines"] = merged
                        with open(seg_json, "w", encoding="utf-8") as _f:
                            json.dump(seg_data, _f, indent=2)
                        if out_txt.exists():
                            old_lines = out_txt.read_text("utf-8").splitlines()
                            merged_text = []
                            for grp in merge_groups:
                                merged_text.append(" ".join(old_lines[i] for i in grp if i < len(old_lines)))
                            out_txt.write_text("\n".join(merged_text) + "\n", encoding="utf-8")
                except Exception as e:
                    PROGRESS[session]["errors"].append(f"merge error {p.name}: {e}")

        except Exception as e:
            PROGRESS[session]["errors"].append(f"{p.name}: {e}")

        # Clip oversized last-line polygons
        if seg_json.exists():
            _im_clip = Image.open(bin_img); _clip_h = _im_clip.size[1]; _im_clip.close()
            _clip_seg_polygons(seg_json, _clip_h)

        # Crop line previews
        try:
            if seg_json.exists():
                PROGRESS[session]["state"] = f"cropping lines {p.name}"
                with open(seg_json, "r", encoding="utf-8") as f:
                    seg = json.load(f)
                if isinstance(seg.get("lines"), list):
                    lines = [l for l in seg["lines"] if isinstance(l, dict)]
                else:
                    lines = _find_lines(seg)
                src_img = Image.open(bin_img)
                for li, line in enumerate(lines, 1):
                    poly = _poly_from_line(line)
                    if not poly:
                        continue
                    x0, y0, x1, y1 = _bbox(poly)
                    crop = src_img.crop((max(0, x0), max(0, y0), min(src_img.width, x1), min(src_img.height, y1)))
                    crop_name = f"{base}_{li:04d}"
                    crop.save(gt_prev / f"{crop_name}.png")
                    gt_file = gt_prev / f"{crop_name}.gt.txt"
                    if not gt_file.exists():
                        text = line.get("text") or ""
                        gt_file.write_text(text.strip(), encoding="utf-8")
                src_img.close()
        except Exception as e:
            PROGRESS[session]["errors"].append(f"crop {p.name}: {e}")

        PROGRESS[session]["done_pages"] = idx

    PROGRESS[session]["state"] = "done"

# --- POST /reocr — Re-run OCR using existing (edited) polygons ---
@app.post("/reocr")
def reocr(session: str = Form(...), page: int = Form(...)):
    if "/" in session or ".." in session:
        return PlainTextResponse("Invalid session name.", status_code=400)
    sess = SESSIONS_BASE_DIR / session
    if not sess.exists():
        return PlainTextResponse("Session not found.", status_code=404)
    if session in PROGRESS and PROGRESS[session].get("state") not in ("done", None):
        return PlainTextResponse("A job is already running for this session.", status_code=409)
    model_path = (sess / "model.txt").read_text().strip() if (sess / "model.txt").exists() else DEFAULT_MODEL
    bin_dir = sess / "bin"
    pages_dir = sess / "pages"
    image_exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    page_imgs = sorted(
        [p for p in pages_dir.iterdir() if p.suffix.lower() in image_exts and p.stem.startswith("page-")],
        key=page_number,
    )
    if page < 1 or page > len(page_imgs):
        return PlainTextResponse("Page not found.", status_code=404)
    page_stem = page_imgs[page - 1].stem
    seg_json = pages_dir / f"{page_stem}.json"
    bin_img = bin_dir / f"{page_stem}.png"
    if not seg_json.exists():
        return PlainTextResponse("No segmentation JSON found for this page.", status_code=404)
    if not bin_img.exists():
        return PlainTextResponse("No binarized image found for this page.", status_code=404)
    PROGRESS[session] = {"total": 0, "done_pages": 0, "state": "reocr", "errors": []}
    threading.Thread(
        target=_reocr_job,
        args=(session, page_stem, model_path),
        daemon=True,
    ).start()
    return JSONResponse({"ok": True, "session": session})


def _reocr_job(session: str, page_stem: str, model_path: str):
    sess = SESSIONS_BASE_DIR / session
    bin_dir = sess / "bin"
    pages_dir = sess / "pages"
    out_dir = sess / "out"
    gt_prev = sess / "gt_preview"
    out_dir.mkdir(exist_ok=True)
    gt_prev.mkdir(exist_ok=True)

    seg_json = pages_dir / f"{page_stem}.json"
    bin_img = bin_dir / f"{page_stem}.png"
    out_txt = out_dir / f"{page_stem}.txt"

    try:
        with open(seg_json, "r", encoding="utf-8") as f:
            seg = json.load(f)
        if isinstance(seg.get("lines"), list):
            lines = [l for l in seg["lines"] if isinstance(l, dict)]
        else:
            lines = _find_lines(seg)

        # Sort lines by vertical position so text comes out in reading order
        def _line_y(line):
            bl = line.get("baseline")
            if isinstance(bl, list) and bl:
                return sum(pt[1] for pt in bl) / len(bl)
            poly = _poly_from_line(line)
            if poly:
                return sum(p[1] for p in poly) / len(poly)
            return 0
        lines.sort(key=_line_y)

        # Persist sorted order back to JSON
        seg["lines"] = lines
        with open(seg_json, "w", encoding="utf-8") as f:
            json.dump(seg, f, indent=2)

        PROGRESS[session]["total"] = len(lines)
        im = Image.open(bin_img)
        img_w, img_h = im.size

        # Run OCR using the Python API on full image with existing segmentation
        PROGRESS[session]["state"] = f"ocr {len(lines)} lines"
        ocr_results = kraken_ocr_lines(bin_img, model_path, seg, bidi_reordering=True)

        # Clean old gt_preview for this page
        for old in gt_prev.glob(f"{page_stem}_*"):
            old.unlink(missing_ok=True)

        text_lines = []
        for i, line in enumerate(lines, 1):
            ocr_text = ocr_results[i - 1] if i - 1 < len(ocr_results) else ""
            poly = _poly_from_line(line)
            if not poly:
                text_lines.append(ocr_text)
                continue

            x0, y0, x1, y1 = _bbox(poly, 3)
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(img_w, x1), min(img_h, y1)

            # Save crop to gt_preview
            crop_name = f"{page_stem}_{i:04d}"
            if x1 - x0 >= 3 and y1 - y0 >= 3:
                crop = im.crop((x0, y0, x1, y1))
                crop.save(gt_prev / f"{crop_name}.png")

            text_lines.append(ocr_text)
            (gt_prev / f"{crop_name}.gt.txt").write_text(ocr_text, encoding="utf-8")
            PROGRESS[session]["done_pages"] = i

        im.close()

        # Write assembled text
        out_txt.write_text("\n".join(text_lines) + "\n", encoding="utf-8")

        # Invalidate sidebar cache
        (sess / "sidebar_cache.json").unlink(missing_ok=True)
        # Remove cached viewer images for this page
        for f in sess.glob(f"{page_stem}_orig*"):
            f.unlink(missing_ok=True)
        for f in sess.glob(f"{page_stem}_bin*"):
            f.unlink(missing_ok=True)

        PROGRESS[session]["state"] = "done"

    except Exception as e:
        PROGRESS[session]["state"] = f"error: {e}"

# --- POST /set_script — Set/update the script tag on an existing session ---
@app.post("/set_script")
def set_script(
    session: str = Form(...),
    script: str = Form(...),
    return_to: str = Form("training"),
    page: str = Form("1"),
):
    if "/" in session or ".." in session:
        return PlainTextResponse("Invalid session name.", status_code=400)
    sess = SESSIONS_BASE_DIR / session
    if not sess.exists():
        return PlainTextResponse("Session not found.", status_code=404)
    clean = sanitize_script_name(script)
    if not clean:
        return PlainTextResponse(
            "Invalid script name. Allowed characters: letters, numbers, dashes, underscores.",
            status_code=400,
        )
    (sess / "script.txt").write_text(clean, encoding="utf-8")
    target = "/view" if return_to == "view" else "/training"
    return RedirectResponse(url=f"{target}?session={session}&page={page}", status_code=303)

# --- POST /delete_session — Remove a session directory entirely ---
@app.post("/delete_session")
def delete_session(session: str = Form(...)):
    # Reject anything that could escape SESSIONS_BASE_DIR
    if "/" in session or ".." in session or not session.strip():
        return PlainTextResponse("Invalid session name.", status_code=400)
    sess = SESSIONS_BASE_DIR / session
    try:
        resolved = sess.resolve()
        if SESSIONS_BASE_DIR.resolve() not in resolved.parents:
            return PlainTextResponse("Invalid session path.", status_code=400)
    except Exception:
        return PlainTextResponse("Invalid session path.", status_code=400)
    if not sess.exists():
        return RedirectResponse(url="/sessions", status_code=303)
    try:
        shutil.rmtree(sess)
    except Exception as e:
        return PlainTextResponse(f"Failed to delete session: {e}", status_code=500)
    PROGRESS.pop(session, None)
    return RedirectResponse(url="/sessions", status_code=303)

# --- GET /status — Poll job progress ---
@app.get("/status")
def status(session: str):
    return JSONResponse(
        PROGRESS.get(
            session,
            {"state": "unknown", "total": 0, "done_pages": 0, "errors": []},
        )
    )

# --- GET /image — Serve page/bin/preview image by name ---
@app.get("/image/{session}/{name}")
def serve_image(session: str, name: str):
    safe_name = os.path.basename(name)
    if not safe_name or safe_name != name:
        return PlainTextResponse("Invalid file name", status_code=400)

    sess_root = SESSIONS_BASE_DIR / session
    paths_to_check = [
        sess_root / safe_name,
        sess_root / "gt_preview" / safe_name,
        sess_root / "bin" / safe_name,
        sess_root / "pages" / safe_name,
        sess_root / "out" / safe_name,
    ]

    for path in paths_to_check:
        try:
            if not path.resolve().is_relative_to(sess_root.resolve()):
                continue
        except Exception:
            continue

        if path.exists():
            return FileResponse(
                str(path),
                headers={
                    "Cache-Control": "public, max-age=604800",
                    "Accept-Ranges": "bytes",
                },
            )

    return PlainTextResponse("Not found", status_code=404)

# --- GET /thumb — Serve disk-cached thumbnail (PNG or WEBP) ---
THUMBS_DIRNAME = "thumbs"

def _thumb_path(sess_root: Path, name: str, w: int, ext: str) -> Path:
    stem, _ = os.path.splitext(name)
    return sess_root / THUMBS_DIRNAME / f"{stem}_w{w}.{ext}"


@app.get("/thumb/{session}/{name}")
def serve_thumb(session: str, name: str, w: int = 240, fmt: str = "png"):
    safe_name = os.path.basename(name)
    if not safe_name or safe_name != name:
        return PlainTextResponse("Invalid file name", status_code=400)

    sess_root = SESSIONS_BASE_DIR / session

    # Try original location first, then gt_preview, mirroring /image
    sources = [
        sess_root / safe_name,
        sess_root / "gt_preview" / safe_name,
        sess_root / "bin" / safe_name,
        sess_root / "pages" / safe_name,
        sess_root / "out" / safe_name,
    ]
    src = next((p for p in sources if p.exists()), None)
    if src is None:
        return PlainTextResponse("Not found", status_code=404)

    ext = "webp" if fmt.lower() == "webp" else "png"
    out = _thumb_path(sess_root, safe_name, w, ext)
    out.parent.mkdir(parents=True, exist_ok=True)

    # If cached thumb is fresh, serve it
    try:
        if out.exists() and out.stat().st_mtime >= src.stat().st_mtime:
            return FileResponse(
                str(out),
                headers={"Cache-Control": "public, max-age=604800, immutable"},
            )
    except Exception:
        pass

    if Image is None:
        return PlainTextResponse("Pillow not installed for thumbnails", status_code=500)

    # Build/refresh thumb
    img = Image.open(src).convert("RGB")
    img.thumbnail((w, 99999), Image.LANCZOS)  # constrain by width
    if ext == "webp":
        img.save(out, "WEBP", quality=75, method=6)
    else:
        img.save(out, "PNG", optimize=True)

    return FileResponse(
        str(out), headers={"Cache-Control": "public, max-age=604800, immutable"}
    )

# --- POST /save — Save edited viewer text (supports autosave fetch + form redirect) ---
@app.post("/save")
async def save_text(
    request: Request,
    session: str = Form(...),
    txt_name: str = Form(...),
    text: str = Form(...),
    page: Optional[int] = Form(None),
):
    target = SESSIONS_BASE_DIR / session / txt_name

    # Path Traversal Protection (kept)
    try:
        if not target.resolve().is_relative_to(SESSIONS_BASE_DIR.resolve()):
            return PlainTextResponse("Invalid path", status_code=403)
    except Exception:
        return PlainTextResponse("Invalid path", status_code=403)

    if not target.exists():
        return PlainTextResponse("Target text file not found.", status_code=404)

    target.write_text(text, encoding="utf-8")

    # Decide the page to return to
    try:
        pnum = int(page) if page is not None else 1
    except Exception:
        pnum = 1
    if pnum < 1:
        pnum = 1

    # If called via fetch/beacon (autosave), return JSON; otherwise redirect
    wants_json = "application/json" in (request.headers.get("accept") or "")
    is_fetch = (request.headers.get("x-requested-with") or "").lower() == "fetch"
    is_keepalive = (request.headers.get("sec-fetch-mode") or "") == "no-cors"
    if wants_json or is_fetch or is_keepalive:
        return JSONResponse({"ok": True, "page": pnum})

    return RedirectResponse(url=f"/view?session={session}&page={pnum}", status_code=303)

# --- GET /view — Paginated OCR viewer ---
@app.get("/view", response_class=HTMLResponse)
def view(session: str, page: int = 1):
    sess = SESSIONS_BASE_DIR / session
    if not sess.exists():
        return HTMLResponse("<h3>Session not found.</h3>", status_code=404)

    pages_dir = sess / "pages"
    bin_dir = sess / "bin"
    out_dir = sess / "out"
    image_exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

    # Collect page outputs in natural order (relies on your existing `page_number` helper)
    page_files = sorted(out_dir.glob("page-*.txt"), key=page_number)
    if not page_files:
        # Segment-only sessions have no out/*.txt — paginate by binarized pages instead
        page_files = sorted(bin_dir.glob("page-*.png"), key=page_number)
    if not page_files:
        if session in PROGRESS and PROGRESS.get(session, {}).get("state") != "done":
            return RedirectResponse(url="/", status_code=303)
        return HTMLResponse(
            f"<h3>No page outputs found for session '{session}'.</h3>"
            f"<p>Did the job fail? <a href='/'>Go back</a></p>"
        )

    total_pages = len(page_files)

    # clamp page safely to 1..N
    try:
        page = int(page)
    except Exception:
        page = 1
    if page < 1:
        page = 1
    if page > total_pages:
        page = total_pages

    # ---- current page only ----
    t = page_files[page - 1]
    page_stem = t.stem  # e.g., "page-0007"

    # original image (copy once into session root for /image/<session>/<name>)
    orig = next((x for x in pages_dir.iterdir()
                 if x.stem == page_stem and x.suffix.lower() in image_exts), None)
    orig_name = None
    if orig:
        orig_name = f"{page_stem}_orig{orig.suffix.lower()}"
        if not (sess / orig_name).exists():
            try:
                shutil.copy(orig, sess / orig_name)
            except Exception:
                orig_name = None

    # binarized image (copy once)
    bin_name = None
    bin_src = bin_dir / f"{page_stem}.png"
    if bin_src.exists():
        bin_name = f"{page_stem}_bin.png"
        if not (sess / bin_name).exists():
            try:
                shutil.copy(bin_src, sess / bin_name)
            except Exception:
                bin_name = None

    # Load segmentation polygons for viewer overlay (optional — degrades gracefully)
    seg_lines = []
    seg_json_path = pages_dir / f"{page_stem}.json"
    if seg_json_path.exists():
        try:
            with open(seg_json_path, "r", encoding="utf-8") as f:
                seg = json.load(f)
            raw_lines = [l for l in seg["lines"] if isinstance(l, dict)] \
                if isinstance(seg.get("lines"), list) else _find_lines(seg)
            for i, line in enumerate(raw_lines, 1):
                poly = _poly_from_line(line)
                if poly:
                    seg_lines.append({"idx": i, "points_str": " ".join(f"{x},{y}" for x,y in poly)})
        except Exception:
            pass

    # Load per-page metadata
    page_meta = {"folio": "", "side": "", "col": "", "part": ""}
    page_meta_path = pages_dir / f"{page_stem}.meta.json"
    if page_meta_path.exists():
        try:
            with open(page_meta_path, "r", encoding="utf-8") as pm:
                page_meta.update(json.load(pm))
        except Exception:
            pass

    page_txt = out_dir / f"{page_stem}.txt"
    page_text = page_txt.read_text("utf-8") if page_txt.exists() else ""
    current_page = {
        "title": page,  # human-friendly number
        "page_stem": page_stem,
        "orig_image_name": orig_name or "not_found.png",
        "bin_image_name": bin_name or "not_found.png",
        "txt_name": f"out/{page_stem}.txt",
        "text": page_text,
        "folio": page_meta.get("folio", ""),
        "side": page_meta.get("side", ""),
        "part": page_meta.get("part", ""),
        "notes": page_meta.get("notes", ""),
    }

    # ---- lightweight sidebar list for thumbs (no heavy text reads) ----
    sidebar_pages = []
    for i, pt in enumerate(page_files, start=1):
        stem = pt.stem
        thumb = next((x for x in pages_dir.iterdir()
                      if x.stem == stem and x.suffix.lower() in image_exts), None)
        thumb_name = None
        if thumb:
            thumb_name = f"{stem}_orig{thumb.suffix.lower()}"
            if not (sess / thumb_name).exists():
                try:
                    shutil.copy(thumb, sess / thumb_name)
                except Exception:
                    thumb_name = None

        sidebar_pages.append({
            "title": i,  # used as page number in links
            "orig_image_name": thumb_name or "not_found.png",
            "text": "",  # keep light
        })

    _preprocess_labels = {"binarize": "Binarized", "grayscale": "Grayscale", "original": "Preprocessed (original)"}
    _preprocess_key = (sess / "preprocess.txt").read_text().strip() if (sess / "preprocess.txt").exists() else "binarize"
    preprocess_label = _preprocess_labels.get(_preprocess_key, "Binarized")

    model_name = ""
    if (sess / "model.txt").exists():
        model_name = Path((sess / "model.txt").read_text().strip()).stem

    script_file = sess / "script.txt"
    current_script = script_file.read_text("utf-8").strip() if script_file.exists() else ""

    return VIEW_HTML.render(
        session=session,
        page=page,
        total_pages=total_pages,
        current_page=current_page,  # single page for the main area
        pages=sidebar_pages,        # used by the sidebar only
        seg_lines=seg_lines,        # polygon overlay for binarized image
        preprocess_label=preprocess_label,
        model_name=model_name,
        scripts=list_script_dirs(),
        current_script=current_script,
    )

# --- GET /download — Download all pages merged into a single text file ---
@app.get("/download")
def download(session: str):
    sess=SESSIONS_BASE_DIR/session; out_dir=sess/"out"; merged=sess/"merged.txt"
    if not out_dir.exists(): return PlainTextResponse("No outputs found.",status_code=404)
    with open(merged, "w", encoding="utf-8") as m:
        for t in sorted(out_dir.glob("page-*.txt"),key=page_number): m.write(t.read_text("utf-8")+"\n\n")
    return FileResponse(str(merged),filename="ocr_merged.txt",media_type="text/plain")

# --- GET /export_gt — Export line crops + GT text as a zip ---
@app.get("/export_gt")
def export_gt(session: str, prefill: Optional[str]=None):
    sess=SESSIONS_BASE_DIR/session
    if not sess.exists(): return PlainTextResponse("Session not found.",status_code=404)
    model_path=(sess/"model.txt").read_text().strip() if (sess/"model.txt").exists() else DEFAULT_MODEL
    pages_dir=sess/"pages"; bin_dir=sess/"bin"; gt_dir=sess/"ground_truth"; gt_dir.mkdir(exist_ok=True)
    for bin_img in sorted(bin_dir.glob("page-*.png"),key=page_number):
        seg_json=pages_dir/f"{bin_img.stem}.json"
        if not seg_json.exists():
            try:
                seg_data = kraken_segment(bin_img)
                with open(seg_json, "w", encoding="utf-8") as _f:
                    json.dump(seg_data, _f, indent=2)
            except Exception as e: print(f"Failed to segment {bin_img.name}: {e}"); continue

        if extract_lines_to_gt(bin_img,seg_json,gt_dir)>0 and prefill=="yes":
            # OCR all lines at once using the segmentation
            with open(seg_json, "r", encoding="utf-8") as _f:
                seg_data = json.load(_f)
            try:
                ocr_results = kraken_ocr_lines(bin_img, model_path, seg_data, bidi_reordering=True)
            except Exception as e:
                print(f"Failed to OCR {bin_img.name}: {e}")
                ocr_results = []
            gt_pngs = sorted(gt_dir.glob(f"{bin_img.stem}_*.png"))
            for li, line_png in enumerate(gt_pngs):
                gt = line_png.with_suffix(".gt.txt")
                if gt.exists() and gt.stat().st_size > 0: continue
                ocr_text = ocr_results[li] if li < len(ocr_results) else ""
                gt.write_text(ocr_text, "utf-8")

    zip_path=sess/"ground_truth.zip"
    with zipfile.ZipFile(zip_path,"w",zipfile.ZIP_DEFLATED) as z:
        for p in sorted(gt_dir.glob("*")): z.write(p,arcname=f"ground_truth/{p.name}")
    return FileResponse(str(zip_path),filename="ground_truth.zip",media_type="application/zip")

# --- GET /training — Training annotation view (paginated, polygon editing) ---
@app.get("/training", response_class=HTMLResponse)
def training(session: str, prefill: Optional[str] = None, page: int = 1):
    if Image is None: return HTMLResponse("<h3>Error: Pillow (PIL) library not found.</h3>", 500)
    sess = SESSIONS_BASE_DIR / session
    if not sess.exists(): return HTMLResponse("<h3>Session not found.</h3>", 404)
    pages_dir = sess / "pages"
    bin_dir = sess / "bin"
    out_dir = sess / "out"
    gt_prev = sess / "gt_preview"
    gt_prev.mkdir(exist_ok=True)
    model_path = (sess / "model.txt").read_text().strip() if (sess / "model.txt").exists() else DEFAULT_MODEL
    image_exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

    # --- Part 1: Build sidebar page list (cached) ---
    all_pages_for_sidebar = _build_sidebar_cache(sess, bin_dir, pages_dir, image_exts)

    # --- Part 2: Load and render the requested page ---
    def _resolve_page_stem(n: int) -> str:
        """
        Return the actual 'page-XX' stem for page number n, regardless of zero-padding.
        Prefer out/ (txt) then bin/ (png) then pages/ (json). Fallback to 'page-n'.
        """
        def _match_from(glob_iter):
            for p in sorted(glob_iter, key=page_number):
                try:
                    if page_number(p) == n:
                        return p.stem
                except Exception:
                    continue
            return None

        # Try out/ text files
        stem = _match_from(out_dir.glob("page-*.txt"))
        if stem:
            return stem
        # Try bin/ images
        stem = _match_from(bin_dir.glob("page-*.png"))
        if stem:
            return stem
        # Try pages/ json
        stem = _match_from(pages_dir.glob("page-*.json"))
        if stem:
            return stem

        # Fallback (works for non-padded layouts)
        return f"page-{n}"

    page_stem_to_load = _resolve_page_stem(page)

    bin_img = bin_dir / f"{page_stem_to_load}.png"
    seg_json_path = pages_dir / f"{page_stem_to_load}.json"
    viewer_txt_path = out_dir / f"{page_stem_to_load}.txt"

    if not bin_img.exists():
        return HTMLResponse(f"<h3>Page {page} not found for this session.</h3>", 404)

    if not seg_json_path.exists():
        return HTMLResponse(f"<h3>Page {page} not found or processed.</h3>", 404)

    try:
        with open(seg_json_path, "r", encoding="utf-8") as f: seg = json.load(f)
        # Prefer authoritative top-level lines
        if isinstance(seg.get("lines"), list):
            lines = [l for l in seg["lines"] if isinstance(l, dict)]
        else:
            # Fallback for older sessions
            lines = _find_lines(seg)
    except Exception as e: return HTMLResponse(f"<h3>Failed to read segmentation for page {page}: {e}</h3>", 500)

    try:
        im = Image.open(bin_img)
        W, H = im.size
        im.close()
    except Exception as e:
        return HTMLResponse(f"<h3>Failed to open binarized image for page {page}: {e}</h3>", 500)

    line_items = []
    for i, line in enumerate(lines, 1):
        poly = _poly_from_line(line)
        if not poly: continue

        thumb_name = f"{page_stem_to_load}_{i:04d}.png"
        gt_name = f"{page_stem_to_load}_{i:04d}.gt.txt"
        gt_path = gt_prev / gt_name

        current_text = ""
        thumb_missing = False
        if gt_path.exists():
            current_text = gt_path.read_text("utf-8")
        elif (gt_prev / thumb_name).exists():
            gt_path.write_text("", encoding="utf-8")
        else:
            # Defer cropping — will be done lazily via /api/crop_line
            thumb_missing = True
            gt_path.write_text("", encoding="utf-8")

        line_items.append({
            "idx": i,
            "points_str": " ".join(f"{x},{y}" for x, y in poly),
            "thumb_name": thumb_name,
            "thumb_missing": thumb_missing,
            "gt_rel": f"gt_preview/{gt_name}",
            "text": current_text,
        })

    bin_name = f"{page_stem_to_load}_bin.png"
    if not (sess / bin_name).exists() and bin_img.exists():
        try: shutil.copy(bin_img, sess / bin_name)
        except Exception: pass

    page_data_for_one = {
        "title": page_stem_to_load.replace("page-",""), 
        "bin_image_name": bin_name, 
        "img_w": W, 
        "img_h": H, 
        "lines": line_items
    }

    script_file = sess / "script.txt"
    current_script = script_file.read_text("utf-8").strip() if script_file.exists() else ""
    return TRAIN_HTML.render(
        session=session,
        all_pages=all_pages_for_sidebar,  # The list for the sidebar
        page=page_data_for_one,           # The data for the main content
        current_page=page_data_for_one["title"], # To highlight the active page
        current_page_num=page,
        total_pages=len(all_pages_for_sidebar),
        render_ts=int(time.time()),
        scripts=list_script_dirs(),
        current_script=current_script,
    )

# --- GET /api/crop_line — Lazy on-demand line crop for older sessions ---
@app.get("/api/crop_line")
def crop_line(session: str, page_stem: str, line_idx: int):
    if Image is None:
        return PlainTextResponse("Pillow not available", status_code=500)
    sess = SESSIONS_BASE_DIR / session
    # Normalise page_stem: the template passes just the number (e.g. "01"), add prefix
    page_key = page_stem if page_stem.startswith("page-") else f"page-{page_stem}"
    gt_prev = sess / "gt_preview"
    thumb_name = f"{page_key}_{line_idx:04d}.png"
    thumb_path = gt_prev / thumb_name
    # If already cropped (race condition guard), just serve it
    if thumb_path.exists():
        return FileResponse(str(thumb_path), media_type="image/png",
                            headers={"Cache-Control": "public, max-age=604800"})
    # Crop on demand
    bin_img = sess / "bin" / f"{page_key}.png"
    seg_json = sess / "pages" / f"{page_key}.json"
    if not bin_img.exists() or not seg_json.exists():
        return PlainTextResponse("Source files not found", status_code=404)
    try:
        with open(seg_json, "r", encoding="utf-8") as f:
            seg = json.load(f)
        if isinstance(seg.get("lines"), list):
            lines = [l for l in seg["lines"] if isinstance(l, dict)]
        else:
            lines = _find_lines(seg)
        if line_idx < 1 or line_idx > len(lines):
            return PlainTextResponse("Line index out of range", status_code=404)
        poly = _poly_from_line(lines[line_idx - 1])
        if not poly:
            return PlainTextResponse("No polygon for line", status_code=404)
        im = Image.open(bin_img)
        W, H = im.size
        x0, y0, x1, y1 = _bbox(poly, 3)
        x0 = max(0, min(x0, W - 1)); y0 = max(0, min(y0, H - 1))
        x1 = max(x0 + 1, min(x1, W)); y1 = max(y0 + 1, min(y1, H))
        if (x1 - x0) >= 3 and (y1 - y0) >= 3:
            gt_prev.mkdir(exist_ok=True)
            im.crop((x0, y0, x1, y1)).save(thumb_path)
        im.close()
        if thumb_path.exists():
            return FileResponse(str(thumb_path), media_type="image/png",
                                headers={"Cache-Control": "public, max-age=604800"})
        return PlainTextResponse("Crop too small", status_code=404)
    except Exception as e:
        return PlainTextResponse(f"Crop failed: {e}", status_code=500)

# --- POST /save_line — Save individual GT line text ---
@app.post("/save_line")
def save_line_text(session: str = Form(...), gt_name: str = Form(...), text: str = Form(...)):
    # Construct the full path from the session and gt_name
    target = SESSIONS_BASE_DIR / session / gt_name
    try:
        if not target.resolve().is_relative_to(SESSIONS_BASE_DIR.resolve()):
            return JSONResponse({"status": "error", "message": "Invalid path"}, status_code=403)
    except Exception:
        return JSONResponse({"status": "error", "message": "Invalid path check failed"}, status_code=403)

    try:
        target.write_text(text, encoding="utf-8")
        return JSONResponse({"status": "ok", "message": "Saved!"})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

# --- POST /export_final_gt — Copy GT crops to training/validation dir ---
@app.post("/export_final_gt")
def export_final_gt(session: str = Form(...)):
    sess = SESSIONS_BASE_DIR / session
    gt_prev = sess / "gt_preview"
    if not gt_prev.exists():
        return PlainTextResponse("No ground truth data found to export.", status_code=404)

    script_file = sess / "script.txt"
    raw = script_file.read_text("utf-8").strip() if script_file.exists() else ""
    script = sanitize_script_name(raw)
    if not script:
        return PlainTextResponse(
            "This session has no valid script tag. Recreate the session or fix "
            "sessions/<session>/script.txt (allowed: letters, numbers, dashes, underscores).",
            status_code=400,
        )
    target_dir = FINAL_GT_DIR / script
    target_dir.mkdir(parents=True, exist_ok=True)

    count_files = 0
    for p in gt_prev.glob("*"):
        unique_name = f"{session}_{p.name}"
        try:
            shutil.copy(p, target_dir / unique_name)
            count_files += 1
        except Exception as e:
            print(f"Could not copy {p.name} to {target_dir}: {e}")

    # each line corresponds to 2 files: .png and .gt.txt
    return PlainTextResponse(f"Exported {count_files // 2} lines to {script}/ (PNG + GT).")

# --- GET /export_viewer_csv — Export viewer text to CSV in ~/Downloads ---
@app.get("/export_viewer_csv")
def export_viewer_csv(session: str):
    sess = SESSIONS_BASE_DIR / session
    out_dir = sess / "out"
    if not out_dir.exists():
        return PlainTextResponse("No outputs found.", status_code=404)

    csv_path = DOWNLOADS_DIR / f"{session}_viewer_lines.csv"
    headers = [
        "title", "page", "line",
        "editor_id", "obj_id", "part", "side", "col", "folio", "book", "chapter", "verse",
        "translation", "unicode", "transliteration", "notes", "no_vowels",
        "text_order_id", "lang_id", "timestamp", "update", "line_1", "script_id"
    ]

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Extract title from session name: strip "hebrocr_" prefix and "_<uuid>" suffix
    title = session
    if title.startswith("hebrocr_"):
        title = title[len("hebrocr_"):]
    if len(title) > 9 and title[-9] == "_":
        title = title[:-9]

    # Read obj_id from file (blank if not set)
    obj_id_path = sess / "obj_id.txt"
    obj_id_val = ""
    if obj_id_path.exists():
        obj_id_val = obj_id_path.read_text("utf-8").strip()

    # Read script tag (script_id column; later this will be resolved to an integer)
    script_path = sess / "script.txt"
    script_val = ""
    if script_path.exists():
        script_val = script_path.read_text("utf-8").strip()

    text_order_counter = 0

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()

        for t in sorted(out_dir.glob("page-*.txt"), key=page_number):
            page_stem = t.stem  # e.g., "page-1"
            try:
                page_num = int(page_stem.split("-")[-1])
            except Exception:
                page_num = ""

            # Load per-page metadata if available
            page_meta_path = sess / "pages" / f"{page_stem}.meta.json"
            page_meta = {}
            if page_meta_path.exists():
                try:
                    with open(page_meta_path, "r", encoding="utf-8") as pm:
                        page_meta = json.load(pm)
                except Exception:
                    pass

            # Load per-line metadata
            lines = t.read_text("utf-8").splitlines()

            for idx, text_line in enumerate(lines, start=1):
                text_order_counter += 1

                # Load per-line metadata if available
                line_meta_path = sess / "gt_preview" / f"{page_stem}_{idx:04d}.meta.json"
                line_meta = {}
                if line_meta_path.exists():
                    try:
                        with open(line_meta_path, "r", encoding="utf-8") as lm:
                            line_meta = json.load(lm)
                    except Exception:
                        pass

                row = {
                    "title": title,
                    "page": page_num,
                    "line": idx,
                    "editor_id": "",
                    "obj_id": obj_id_val,
                    "part": page_meta.get("part", ""),
                    "side": page_meta.get("side", ""),
                    "col": line_meta.get("col", ""),
                    "folio": page_meta.get("folio", ""),
                    "book": "",
                    "chapter": line_meta.get("chapter", ""),
                    "verse": line_meta.get("verse", ""),
                    "translation": line_meta.get("translation", ""),
                    "unicode": text_line,
                    "transliteration": line_meta.get("transliteration", ""),
                    "notes": line_meta.get("notes", "") or page_meta.get("notes", ""),
                    "no_vowels": "",
                    "text_order_id": text_order_counter,
                    "lang_id": "",
                    "timestamp": now,
                    "update": now,
                    "line_1": "",
                    "script_id": script_val,
                }
                writer.writerow(row)

    return PlainTextResponse(f"CSV saved to: {csv_path}")

# --- POST /api/save_page_meta — Save per-page metadata (folio, side, part) ---
@app.post("/api/save_page_meta")
def save_page_meta(
    session: str = Form(...),
    page_stem: str = Form(...),
    folio: str = Form(""),
    side: str = Form(""),
    part: str = Form(""),
    notes: str = Form(""),
):
    sess = SESSIONS_BASE_DIR / session
    if not sess.exists():
        return JSONResponse({"error": "Session not found"}, status_code=404)
    page_key = page_stem if page_stem.startswith("page-") else f"page-{page_stem}"
    meta_path = sess / "pages" / f"{page_key}.meta.json"
    meta = {"folio": folio.strip(), "side": side.strip(), "part": part.strip(), "notes": notes}
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f)
        return JSONResponse({"status": "ok"})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# --- POST /api/save_line_meta — Save per-line metadata ---
@app.post("/api/save_line_meta")
def save_line_meta(
    session: str = Form(...),
    page_stem: str = Form(...),
    line_idx: int = Form(...),
    col: str = Form(""),
    chapter: str = Form(""),
    verse: str = Form(""),
    translation: str = Form(""),
    transliteration: str = Form(""),
    notes: str = Form(""),
):
    sess = SESSIONS_BASE_DIR / session
    if not sess.exists():
        return JSONResponse({"error": "Session not found"}, status_code=404)
    page_key = page_stem if page_stem.startswith("page-") else f"page-{page_stem}"
    meta_path = sess / "gt_preview" / f"{page_key}_{line_idx:04d}.meta.json"
    meta = {
        "col": col.strip(),
        "chapter": chapter.strip(), "verse": verse.strip(),
        "translation": translation.strip(), "transliteration": transliteration.strip(),
        "notes": notes.strip(),
    }
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f)
        return JSONResponse({"status": "ok"})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# --- POST /api/push_to_gt — Push viewer text lines into GT files ---
@app.post("/api/push_to_gt")
def api_push_to_gt(session: str = Form(...), page_stem: str = Form(...)):
    session_dir = SESSIONS_BASE_DIR / session
    if not session_dir.exists():
        return JSONResponse({"error": "Session not found"}, status_code=404)
    txt_path = session_dir / "out" / f"{page_stem}.txt"
    if not txt_path.exists():
        return JSONResponse({"error": "Page text not found"}, status_code=404)
    viewer_lines = txt_path.read_text(encoding="utf-8").split("\n")
    gt_dir = session_dir / "gt_preview"
    gt_files = sorted(gt_dir.glob(f"{page_stem}_*.gt.txt"))
    updated = 0
    for i, gt_file in enumerate(gt_files):
        if i < len(viewer_lines):
            gt_file.write_text(viewer_lines[i], encoding="utf-8")
            updated += 1
    return JSONResponse({"updated": updated, "total_gt": len(gt_files), "total_lines": len(viewer_lines)})

# --- POST /update_segmentation — Save edited polygons and regenerate line crops ---
class PolygonUpdate(BaseModel):
    session: str
    page_id: str  # e.g., "1" (we’ll add "page-" where needed)
    polygons: List[List[Tuple[float, float]]]
    texts: Optional[List[str]] = None

@app.post("/update_segmentation")
def update_segmentation(update: PolygonUpdate):
    """
    Robust saver that supports both "page-<n>" and "<n>" filenames, and clears previews accordingly.
    """
    print(f"update_segmentation: session={update.session} page={update.page_id} ---")
    sess        = SESSIONS_BASE_DIR / update.session
    pages_dir   = sess / "pages"
    bin_dir     = sess / "bin"
    gt_prev     = sess / "gt_preview"

    raw_id      = str(update.page_id)
    page_key    = raw_id if raw_id.startswith("page-") else f"page-{raw_id}"
    # Try both filename styles
    seg_candidates = [pages_dir / f"{page_key}.json", pages_dir / f"{raw_id}.json"]
    bin_candidates = [bin_dir / f"{page_key}.png",  bin_dir / f"{raw_id}.png"]

    seg_json = next((p for p in seg_candidates if p.exists()), seg_candidates[0])
    bin_img  = next((p for p in bin_candidates if p.exists()), bin_candidates[0])

    if not seg_json.exists():
        return PlainTextResponse("Segmentation file not found.", status_code=404)
    if not bin_img.exists():
        return PlainTextResponse("Binarized image not found for cropping.", status_code=404)
    if Image is None:
        return PlainTextResponse("Pillow (PIL) library is missing.", status_code=500)

    try:
        # 1) Load existing segmentation JSON (or empty)
        try:
            with open(seg_json, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"  WARN: Could not read JSON, starting fresh: {e}")
            data = {}

        # 2) Build authoritative new lines in order
        new_lines = []
        for i, poly in enumerate(update.polygons or [], start=1):
            new_lines.append({
                "id": f"line_{i:04d}",
                "boundary": [[float(x), float(y)] for (x, y) in poly],
                "baseline": _baseline_from_poly(poly),
            })
        print(f"  Prepared {len(new_lines)} new lines.")

        # 3) Clear ALL previous 'lines' arrays, then set a single top-level one
        def _remove_all_lines_keys(node):
            if isinstance(node, dict):
                if 'lines' in node and isinstance(node['lines'], list):
                    node['lines'] = []
                for v in node.values():
                    _remove_all_lines_keys(v)
            elif isinstance(node, list):
                for v in node:
                    _remove_all_lines_keys(v)
        _remove_all_lines_keys(data)
        data["lines"] = new_lines

        # 4) Write updated segmentation JSON
        with open(seg_json, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"  Wrote updated segmentation JSON to {seg_json.name}.")

        # Invalidate sidebar cache since line counts may have changed
        sidebar_cache = sess / "sidebar_cache.json"
        if sidebar_cache.exists():
            try: sidebar_cache.unlink()
            except Exception: pass

        # 5) Clear old gt_preview files for this page (support both prefixes)
        gt_prev.mkdir(parents=True, exist_ok=True)
        removed = 0
        for key in {page_key, raw_id}:
            for p in gt_prev.glob(f"{key}_*"):
                try:
                    p.unlink()
                    removed += 1
                except Exception as e:
                    print(f"  WARN: Could not delete {p.name}: {e}")
        print(f"  Cleared {removed} old preview files.")

        # 6) Re-extract crops and write GT texts
        im = Image.open(bin_img)
        W, H = im.size
        written = 0
        num_texts = len(update.texts) if update.texts else 0

        for i, line in enumerate(new_lines, start=1):
            poly = line.get("boundary") or []
            if not poly: continue
            poly_t = [(float(x), float(y)) for x, y in poly]

            x0, y0, x1, y1 = _bbox(poly_t, pad=3)
            x0 = max(0, min(x0, W - 1))
            y0 = max(0, min(y0, H - 1))
            x1 = max(x0 + 1, min(x1, W))
            y1 = max(y0 + 1, min(y1, H))
            if (x1 - x0) < 3 or (y1 - y0) < 3: continue

            thumb = gt_prev / f"{page_key}_{i:04d}.png"
            gt    = thumb.with_suffix(".gt.txt")

            try:
                im.crop((x0, y0, x1, y1)).save(thumb)
                text_value = ""
                if update.texts and (i - 1) < num_texts:
                    text_value = update.texts[i - 1]
                gt.write_text(text_value, encoding="utf-8")
                written += 1
            except Exception as e:
                print(f"  WARN: Could not write crop/gt for line {i}: {e}")

        im.close()
        print(f"  Regenerated {written} crops/GTs for {update.page_id}.")

        print(f"--- update_segmentation done for {update.page_id} ---")
        return PlainTextResponse("Segmentation and text updated.", status_code=200)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return PlainTextResponse(f"An error occurred: {e}", status_code=500)

# ---- Mount page routers (at end to avoid circular import) ----
from models_page import router as models_router
from sessions_page import router as sessions_router
app.include_router(models_router)
app.include_router(sessions_router)