"""
exporter.py — Export processed images to 16-bit TIFF and/or JPEG.

Output directory: <source_folder>/processed/
Filename convention: <original_stem>_processed.tiff / _preview.jpg
"""

import gc
import logging
import os
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class Exporter:
    """Write processed float32 arrays to disk."""

    def __init__(
        self,
        output_subdir: str = "processed",
        export_tiff: bool = True,
        export_jpeg: bool = False,
        jpeg_quality: int = 92,
    ):
        self.output_subdir = output_subdir
        self.export_tiff = export_tiff
        self.export_jpeg = export_jpeg
        self.jpeg_quality = jpeg_quality

    # ------------------------------------------------------------------
    # Single image export
    # ------------------------------------------------------------------

    def export(
        self,
        image: np.ndarray,
        source_path: str,
        output_dir: Optional[str] = None,
    ) -> List[str]:
        """
        Export a processed image.

        Args:
            image:       float32 (H, W, 3), range [0, 1].
            source_path: Original source file path (used to derive output name).
            output_dir:  Override output directory.  If None, uses
                         <source_folder>/processed/.

        Returns:
            List of written file paths.
        """
        source = Path(source_path)
        if output_dir is None:
            out_dir = source.parent / self.output_subdir
        else:
            out_dir = Path(output_dir)

        out_dir.mkdir(parents=True, exist_ok=True)
        stem = source.stem
        written = []

        if self.export_tiff:
            tiff_path = out_dir / f"{stem}_processed.tiff"
            self._write_tiff(image, str(tiff_path))
            written.append(str(tiff_path))

        if self.export_jpeg:
            jpeg_path = out_dir / f"{stem}_preview.jpg"
            self._write_jpeg(image, str(jpeg_path))
            written.append(str(jpeg_path))

        return written

    # ------------------------------------------------------------------
    # Batch export
    # ------------------------------------------------------------------

    def export_batch(
        self,
        jobs: List[dict],
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> dict:
        """
        Process a batch of export jobs.

        Each job dict must have:
            "image":       float32 np.ndarray
            "source_path": str

        Optionally:
            "output_dir":  str override

        Args:
            jobs:              List of job dicts.
            progress_callback: Called as callback(current, total, filename).

        Returns:
            dict with keys:
                "success":  list of written paths
                "failed":   list of (source_path, error_str) tuples
        """
        success = []
        failed = []
        total = len(jobs)

        for i, job in enumerate(jobs):
            src = job.get("source_path", "<unknown>")
            fname = os.path.basename(src)
            if progress_callback:
                progress_callback(i + 1, total, fname)

            try:
                written = self.export(
                    image=job["image"],
                    source_path=src,
                    output_dir=job.get("output_dir"),
                )
                success.extend(written)
                logger.info("Exported %s → %s", fname, written)
            except Exception as e:
                logger.error("Export failed for %s: %s", fname, e)
                failed.append((src, str(e)))
            finally:
                # Free memory after each export
                arr = job.get("image")
                if arr is not None:
                    del arr
                    job["image"] = None
                gc.collect()

        return {"success": success, "failed": failed}

    # ------------------------------------------------------------------
    # Format writers
    # ------------------------------------------------------------------

    def _write_tiff(self, image: np.ndarray, path: str) -> None:
        """Write a float32 [0,1] image as 16-bit TIFF."""
        # Convert to uint16
        img16 = (np.clip(image, 0.0, 1.0) * 65535.0).astype(np.uint16)

        try:
            import tifffile
            tifffile.imwrite(
                path,
                img16,
                photometric="rgb",
                compression=None,  # uncompressed for maximum compatibility
            )
            logger.debug("Wrote 16-bit TIFF: %s", path)
            return
        except ImportError:
            logger.warning("tifffile not available — falling back to imageio")

        try:
            import imageio
            imageio.imwrite(path, img16)
            logger.debug("Wrote TIFF via imageio: %s", path)
            return
        except ImportError:
            logger.warning("imageio not available — falling back to Pillow")

        # Pillow fallback (loses 16-bit on some builds but still writes)
        from PIL import Image
        pil_img = Image.fromarray(img16, mode="RGB")
        pil_img.save(path, format="TIFF")
        logger.debug("Wrote TIFF via Pillow (may be 8-bit): %s", path)

    def _write_jpeg(self, image: np.ndarray, path: str) -> None:
        """Write a float32 [0,1] image as 8-bit JPEG."""
        img8 = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
        from PIL import Image
        pil_img = Image.fromarray(img8, mode="RGB")
        pil_img.save(path, format="JPEG", quality=self.jpeg_quality)
        logger.debug("Wrote JPEG: %s", path)

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    def set_options(
        self,
        export_tiff: Optional[bool] = None,
        export_jpeg: Optional[bool] = None,
        jpeg_quality: Optional[int] = None,
    ) -> None:
        if export_tiff is not None:
            self.export_tiff = export_tiff
        if export_jpeg is not None:
            self.export_jpeg = export_jpeg
        if jpeg_quality is not None:
            self.jpeg_quality = max(1, min(100, jpeg_quality))
