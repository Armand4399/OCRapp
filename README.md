# Manuscript OCR

A local desktop application for OCR (Optical Character Recognition) of historical manuscripts, built on the [Kraken](https://kraken.re) engine. Supports any script or language that Kraken can be trained on — Hebrew, Arabic, Syriac, Greek, Latin, and more. Upload manuscript images, run automatic text recognition, correct the output, and train improved models — all from your browser.

![Python](https://img.shields.io/badge/python-3.12-blue)
![Kraken](https://img.shields.io/badge/kraken-6.0-green)
![Platform](https://img.shields.io/badge/platform-macOS-lightgrey)
![License](https://img.shields.io/badge/license-MIT-orange)

## Features

- **Automatic text recognition** — segment and OCR manuscript pages in any script using Kraken models with MPS (Apple GPU) acceleration
- **Interactive viewer** — side-by-side original and processed images with editable OCR text, polygon overlays, and autosave
- **Layout modes** — Split (images top, text bottom) or Reading (images left, text right)
- **Per-line annotation** — notes, language, book, chapter, and verse metadata per line
- **Folio cascade** — auto-fill folio/side numbering across all pages (1r, 1v, 2r, 2v...)
- **Training workflow** — edit line polygons, export ground truth, fine-tune models, and test accuracy
- **Model management** — view model stats, accuracy curves, character sets, and run quick CER tests
- **Search** — full-text search across all sessions with source image preview
- **CSV export** — structured export with all metadata columns
- **PDF support** — multi-page PDFs automatically split into individual pages
- **Image filters** — invert, contrast, brightness adjustments for reading faded manuscripts
- **Customizable polygon colors** — high-contrast presets for different manuscript backgrounds

## Quick Start

### Option 1: macOS App (recommended for most users)

1. Download `ManuscriptOCR.dmg` from [Releases](https://github.com/Armand4399/OCRapp/releases)
2. Open the DMG and drag **Manuscript OCR** to your Applications folder
3. Double-click to launch — your browser opens automatically

> **macOS security note:** Since the app is not signed with an Apple Developer ID, macOS may show "cannot be opened because the developer cannot be verified." To bypass this:
> 1. **Right-click** (or Control-click) the app in Applications
> 2. Click **Open** from the context menu
> 3. Click **Open** again in the dialog
>
> You only need to do this once — after that it opens normally.

### Option 2: Run from source

Requires Python 3.12, [Homebrew](https://brew.sh), and `poppler` (for PDF support).

```bash
# Clone the repo
git clone https://github.com/Armand4399/OCRapp.git
cd OCRapp

# Create virtual environment and install dependencies
python3.12 -m venv kraken-env
source kraken-env/bin/activate
pip install -r requirements.txt

# Install poppler for PDF support
brew install poppler

# Place your .mlmodel files in the models/ directory

# Launch
uvicorn app:app
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

## Models

The app ships with bundled models to get started:

- **BiblIA** — Hebrew manuscript recognition (grayscale)
- **blla** — baseline segmentation (language-independent)

These are starting points — you can train models for any script within the app, or download pre-trained models from [Zenodo](https://zenodo.org) and the [Kraken model repository](https://kraken.re/main/models.html). Place `.mlmodel` files in:
- `models/` (app directory) — bundled with the app
- `~/Library/Application Support/ManuscriptOCR/models/` — user-added models

## Workflow

1. **Upload** an image or PDF on the home page
2. **Review** the OCR output in the viewer — edit text, adjust settings
3. **Annotate** with folio numbers, notes, and metadata
4. **Train** — open the training page to edit line polygons and export ground truth
5. **Fine-tune** — use the models page to retrain on your corrections
6. **Export** — download merged text, CSV with metadata, or ground truth zip

## Data Storage

User data is stored in `~/Library/Application Support/ManuscriptOCR/`:

```
ManuscriptOCR/
├── sessions/      # OCR session data (images, text, segmentation)
├── training/      # Ground truth for model training
│   ├── train/     # Training data by script
│   └── val/       # Validation data by script
└── models/        # User-added .mlmodel files
```

## Architecture

The app is a [FastAPI](https://fastapi.tiangolo.com/) server with inline Jinja2 templates, calling the Kraken Python API directly (no CLI subprocess for OCR). It runs locally and opens in your default browser.

| File | Purpose |
|------|---------|
| `app.py` | Main application — routes, templates, OCR pipeline |
| `models_page.py` | Model management, training, quick test |
| `sessions_page.py` | Session browser |
| `search_page.py` | Full-text search across sessions |
| `help_page.py` | Usage instructions |
| `launcher.py` | macOS .app launcher (starts server + opens browser) |

## Requirements

- macOS 12+ (Apple Silicon or Intel)
- Python 3.12 (if running from source)
- ~2GB disk space (PyTorch + dependencies)

## Acknowledgments

Built on [Kraken](https://kraken.re) by Benjamin Kiessling. Kraken is an open-source OCR engine for historical and non-Latin scripts.

## License

MIT
