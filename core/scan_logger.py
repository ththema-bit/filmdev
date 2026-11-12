"""
scan_logger.py — Per-scan log for ML training data collection (Stufe 3).

Each processed scan appends one JSON record to data/scan_log.json.
After ~50 scans per filmstock this dataset is dense enough for
k-NN or clustering-based auto-detection (filmstock_detector.py, Stufe 3).

Record schema:
{
    "date":        "2026-07-16",
    "file":        "DSC00123.ARW",          ← basename only (no path)
    "filmstock":   "kodak_gold_200",
    "mask_rgb":    [0.812, 0.513, 0.187],   ← float, 3-channel mean
    "mask_stddev": [0.018, 0.012, 0.009],   ← optional: stdev of sample patch
    "wb_mults":    [0.632, 1.0, 2.746],     ← WB multipliers applied
    "camera":      "Sony A7RII",            ← optional, from EXIF or user
    "notes":       ""                       ← optional free text
}
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Resolve path relative to this file's location → …/filmdev/data/scan_log.json
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_MODULE_DIR, "..", "data")
SCAN_LOG_PATH = os.path.join(_DATA_DIR, "scan_log.json")


def _ensure_data_dir() -> None:
    """Create data/ directory if it doesn't exist."""
    os.makedirs(_DATA_DIR, exist_ok=True)
    gitkeep = os.path.join(_DATA_DIR, ".gitkeep")
    if not os.path.exists(gitkeep):
        try:
            open(gitkeep, "w").close()
        except OSError:
            pass


def log_scan(
    filepath: str,
    filmstock: str,
    mask_rgb: np.ndarray,
    wb_mults: np.ndarray,
    camera: Optional[str] = None,
    notes: Optional[str] = None,
) -> None:
    """Append one scan record to scan_log.json.

    Args:
        filepath:   Full path to the source RAW file (only basename is stored).
        filmstock:  Profile key, e.g. "kodak_gold_200" or "auto".
        mask_rgb:   1D array [R, G, B] — mean colour of the sampled mask patch.
        wb_mults:   1D array [R, G, B] — WB multipliers applied during processing.
        camera:     Optional camera/lens string (e.g. from EXIF).
        notes:      Optional free-text note.
    """
    _ensure_data_dir()

    # --- Normalise inputs ---
    try:
        mask_list = [round(float(v), 6) for v in mask_rgb[:3]]
    except (TypeError, IndexError, ValueError):
        mask_list = [0.0, 0.0, 0.0]

    try:
        wb_list = [round(float(v), 6) for v in wb_mults[:3]]
    except (TypeError, IndexError, ValueError):
        wb_list = [1.0, 1.0, 1.0]

    record = {
        "date": str(date.today()),
        "file": os.path.basename(filepath),
        "filmstock": str(filmstock),
        "mask_rgb": mask_list,
        "wb_mults": wb_list,
        "camera": camera or "",
        "notes": notes or "",
    }

    # --- Load existing log ---
    existing: list = []
    if os.path.isfile(SCAN_LOG_PATH):
        try:
            with open(SCAN_LOG_PATH, "r", encoding="utf-8") as fh:
                existing = json.load(fh)
            if not isinstance(existing, list):
                logger.warning("scan_log.json has unexpected format — resetting")
                existing = []
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read scan_log.json (%s) — starting fresh", exc)
            existing = []

    # --- Append + write ---
    existing.append(record)
    try:
        with open(SCAN_LOG_PATH, "w", encoding="utf-8") as fh:
            json.dump(existing, fh, indent=2, ensure_ascii=False)
        logger.debug("scan_log: appended record for %s (total=%d)", record["file"], len(existing))
    except OSError as exc:
        logger.error("scan_log: could not write %s: %s", SCAN_LOG_PATH, exc)


def load_scan_log() -> list:
    """Load and return all scan records from scan_log.json.

    Returns:
        List of record dicts.  Empty list if the file doesn't exist or is corrupt.
    """
    if not os.path.isfile(SCAN_LOG_PATH):
        return []
    try:
        with open(SCAN_LOG_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, list):
            logger.warning("scan_log.json is not a list — returning empty")
            return []
        return data
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Could not read scan_log.json: %s", exc)
        return []


def scan_log_summary() -> dict:
    """Return a quick summary of the scan log for diagnostics.

    Returns:
        {"total": int, "by_filmstock": {filmstock: count}, "log_path": str}
    """
    records = load_scan_log()
    by_stock: dict = {}
    for rec in records:
        stock = rec.get("filmstock", "unknown")
        by_stock[stock] = by_stock.get(stock, 0) + 1
    return {
        "total": len(records),
        "by_filmstock": by_stock,
        "log_path": SCAN_LOG_PATH,
    }
