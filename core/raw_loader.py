"""
raw_loader.py — RAW file loading using rawpy (libraw wrapper).

Supports: ARW, CR2, CR3, NEF, DNG, ORF, RAF, RW2 and other libraw-supported formats.
"""

import os
import gc
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Supported RAW extensions
SUPPORTED_EXTENSIONS = {
    ".arw", ".cr2", ".cr3", ".nef", ".dng",
    ".orf", ".raf", ".rw2", ".pef", ".srw",
    ".x3f", ".3fr", ".mef", ".nrw", ".rwl",
}


class RawLoader:
    """Loads RAW files into float32 numpy arrays and extracts thumbnails."""

    def __init__(self):
        self._rawpy_available = self._check_rawpy()

    def _check_rawpy(self) -> bool:
        try:
            import rawpy  # noqa: F401
            return True
        except ImportError:
            logger.warning("rawpy not available — RAW loading disabled, only thumbnails via Pillow fallback")
            return False

    @staticmethod
    def list_raw_files(folder: str) -> list[str]:
        """Return sorted list of supported RAW files in the given folder."""
        folder_path = Path(folder)
        files = []
        for f in sorted(folder_path.iterdir()):
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(str(f))
        return files

    def load_full(self, filepath: str) -> np.ndarray:
        """
        Load a RAW file and return a float32 numpy array in range [0, 1].
        Shape: (H, W, 3) — RGB order.

        Raises:
            RuntimeError: if rawpy is not available or loading fails.
        """
        if not self._rawpy_available:
            raise RuntimeError("rawpy is not installed. Cannot load RAW files.")

        import rawpy

        try:
            with rawpy.imread(filepath) as raw:
                rgb = raw.postprocess(
                    output_bps=16,
                    no_auto_bright=True,
                    use_camera_wb=False,
                    output_color=rawpy.ColorSpace.sRGB,
                    demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
                )
            # rgb is uint16, range [0, 65535]
            arr = rgb.astype(np.float32) / 65535.0
            logger.debug("Loaded %s: shape=%s", filepath, arr.shape)
            return arr
        except Exception as e:
            logger.error("Failed to load RAW %s: %s", filepath, e)
            raise RuntimeError(f"Could not load RAW file: {e}") from e

    def load_thumbnail(self, filepath: str) -> Optional[np.ndarray]:
        """
        Extract the embedded JPEG thumbnail from a RAW file for fast preview.
        Returns float32 array [0,1] or None if not available.
        """
        if self._rawpy_available:
            try:
                import rawpy
                with rawpy.imread(filepath) as raw:
                    thumb = raw.extract_thumb()
                if thumb.format == rawpy.ThumbFormat.JPEG:
                    import io
                    from PIL import Image
                    img = Image.open(io.BytesIO(thumb.data))
                    img = img.convert("RGB")
                    arr = np.array(img, dtype=np.float32) / 255.0
                    logger.debug("Extracted JPEG thumb from %s: shape=%s", filepath, arr.shape)
                    return arr
                elif thumb.format == rawpy.ThumbFormat.BITMAP:
                    arr = np.array(thumb.data, dtype=np.float32) / 255.0
                    return arr
            except Exception as e:
                logger.warning("Could not extract thumb from %s: %s", filepath, e)

        # Fallback: try Pillow directly (works for DNG/some formats)
        try:
            from PIL import Image
            img = Image.open(filepath).convert("RGB")
            # Resize to reasonable preview size
            img.thumbnail((1920, 1920), Image.LANCZOS)
            arr = np.array(img, dtype=np.float32) / 255.0
            logger.debug("Loaded via Pillow fallback: %s", filepath)
            return arr
        except Exception as e:
            logger.warning("Pillow fallback failed for %s: %s", filepath, e)

        return None

    def load_half_res(self, filepath: str) -> Optional[np.ndarray]:
        """
        Load RAW at half resolution for faster preview while maintaining quality.
        Returns float32 [0,1] or None on failure.
        """
        if not self._rawpy_available:
            return self.load_thumbnail(filepath)

        import rawpy
        try:
            with rawpy.imread(filepath) as raw:
                rgb = raw.postprocess(
                    output_bps=16,
                    no_auto_bright=True,
                    use_camera_wb=False,
                    output_color=rawpy.ColorSpace.sRGB,
                    half_size=True,
                )
            arr = rgb.astype(np.float32) / 65535.0
            return arr
        except Exception as e:
            logger.warning("Half-res load failed for %s: %s", filepath, e)
            return self.load_thumbnail(filepath)

    @staticmethod
    def free_array(arr: Optional[np.ndarray]) -> None:
        """Explicitly release a numpy array and trigger GC."""
        if arr is not None:
            del arr
        gc.collect()

    @staticmethod
    def get_file_info(filepath: str) -> dict:
        """Return basic info about a RAW file without full decode."""
        info = {
            "path": filepath,
            "filename": os.path.basename(filepath),
            "size_mb": os.path.getsize(filepath) / (1024 * 1024),
        }
        try:
            import rawpy
            with rawpy.imread(filepath) as raw:
                sizes = raw.sizes
                info["width"] = sizes.width
                info["height"] = sizes.height
                info["raw_width"] = sizes.raw_width
                info["raw_height"] = sizes.raw_height
        except Exception:
            info["width"] = None
            info["height"] = None
        return info
