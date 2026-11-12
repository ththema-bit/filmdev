# FilmDev: Analog Scan Processor

Desktop tool for processing analog film negative/positive scans from RAW files (Sony ARW, Canon CR2/CR3, Nikon NEF, DNG, and more).

## What It Does

1. **Load RAW files** from a folder (ARW, CR2, CR3, NEF, DNG, ORF, RAF, RW2, etc.)
2. **Choose mode**: Negative or Positive
3. **Click the film mask** (the border of the film strip) for each image → program samples WB reference
4. **Auto White Balance** correction based on film mask colour
5. **Negative inversion** (linear or logarithmic)
6. **Auto Black/White Point** (percentile or iterative, no clipped pixels)
7. **Export** as 16-bit TIFF for further editing in Lightroom / Capture One

## Installation

### Requirements

- Python 3.10+
- Windows / macOS / Linux

### Install dependencies

```bash
pip install -r requirements.txt
```

> **Note on rawpy / libraw on Windows:**  
> `rawpy` requires the `libraw` DLL. On Windows, install via conda or use the pre-built wheel:
> ```
> pip install rawpy
> ```
> If it fails, try: `conda install -c conda-forge rawpy`

### Run

```bash
cd filmdev
python filmdev.py
```

With verbose logging:
```bash
python filmdev.py --debug
```

## Workflow

```
Open Folder → [Neg/Pos Mode] → For each image:
  1. Click "🎯 Click Film Mask"
  2. Click on the film border area in the preview
  3. Preview updates with WB correction applied
  4. Press → (or "Next ▶") to advance
  5. Use "↑ Copy WB from Prev" for same-roll images with same mask colour
  6. "Skip" to advance without setting WB (uses previous WB)
→ Click "✅ Process All" → 16-bit TIFFs exported to <folder>/processed/
```

## Keyboard Shortcuts

| Key      | Action                  |
|----------|-------------------------|
| `→` / `←` | Next / Previous image  |
| `Enter`  | Next image              |
| `R`      | Re-activate mask click  |
| `S`      | Skip current image      |

## Settings

Access via **⚙ Settings** button:

| Setting | Options | Default |
|---------|---------|---------|
| BP/WP Method | Percentile / Iterative | Percentile |
| Black Point % | 0.0 to 10.0 | 0.5 |
| White Point % | 90.0 to 100.0 | 99.5 |
| Inversion | Linear / Log | Linear |
| Export TIFF | On/Off | On |
| Export JPEG | On/Off | Off |
| JPEG Quality | 50 to 100 | 92 |
| Sample Radius | 5×5 / 15×15 / 25×25 px | 15×15 (7px) |
| Auto-skip WB | On/Off | Off |

## Architecture

```
filmdev/
├── filmdev.py              ← Entry point
├── core/
│   ├── raw_loader.py       ← RAW → numpy float32, thumbnail extraction
│   ├── wb_engine.py        ← WB sample, compute multipliers, apply
│   ├── levels_engine.py    ← Auto BP/WP (percentile + iterative)
│   ├── invert_engine.py    ← Linear + log inversion
│   └── exporter.py         ← 16-bit TIFF + JPEG export
├── ui/
│   ├── main_window.py      ← Tkinter main window
│   ├── image_viewer.py     ← Canvas viewer with click handler
│   └── settings_dialog.py  ← Settings popup
└── requirements.txt
```

## Processing Pipeline

```
RAW file
  │
  ├─→ Thumbnail (embedded JPEG) ──→ Quick preview display
  │
  └─→ Full RAW decode (16-bit) ──→ WB correction
                                        │
                                   [Neg mode] → Inversion (linear / log)
                                        │
                                   Auto Levels (percentile / iterative)
                                        │
                                   Export 16-bit TIFF
                                   (+ optional JPEG preview)
```

## Notes

- `ROADMAP.md` (German) describes planned film stock based calibration; `data/scan_log.json` collects per-scan records for that.

- **Preview uses embedded JPEG thumbnail** for fast navigation: actual RAW is loaded only during batch export
- **WB is sampled from the preview** (thumbnail) but applied correctly to the full RAW during export via stored multipliers
- **Memory**: each processed array is freed after export (`del arr; gc.collect()`)
- **Thread safety**: RAW loading and batch export run in background threads; UI updates via queue polling

## Creating a Windows EXE

```bash
pip install pyinstaller
pyinstaller --onefile --windowed --name FilmDev filmdev.py
# Output: dist/FilmDev.exe
```

> The `--windowed` flag suppresses the console window on Windows.

## Tests

```bash
python3 -m pytest tests/test_wb_invert_pipeline.py -v
```

## License

MIT
