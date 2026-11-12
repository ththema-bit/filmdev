"""
main_window.py — FilmDev main application window.

Layout:
  ┌──────────────────────────────────────────────────────────────────┐
  │  Toolbar: [Open Folder]  [Neg ●/Pos ○]  [⚙ Settings]            │
  ├─────────────────┬────────────────────────────────────────────────┤
  │                 │                                                 │
  │  File List      │   Image Viewer                                  │
  │  (Listbox)      │   (ImageViewer widget)                          │
  │                 │                                                 │
  ├─────────────────┴────────────────────────────────────────────────┤
  │  [◀ Prev]  Image N / M  [Next ▶]  [↑ Copy WB]  [Skip]  [✅ All] │
  ├──────────────────────────────────────────────────────────────────┤
  │  Progress bar + log text area                                     │
  └──────────────────────────────────────────────────────────────────┘
"""

import gc
import logging
import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List, Optional

import numpy as np

from core.exporter import Exporter
from core.filmstock_profiles import (
    FILMSTOCK_PROFILES,
    apply_profile,
    get_profile_labels,
    get_profile_keys_by_label,
)
from core.invert_engine import InvertEngine
from core.levels_engine import LevelsEngine
from core.raw_loader import RawLoader
from core.scan_logger import log_scan
from core.wb_engine import WBEngine
from ui.image_viewer import ImageViewer
from ui.settings_dialog import DEFAULT_SETTINGS, SettingsDialog

logger = logging.getLogger(__name__)


class MainWindow(tk.Tk):
    """Main application window for FilmDev."""

    def __init__(self):
        super().__init__()
        from core.gpu_backend import is_gpu_available, device_name
        gpu_str = f"⚡ {device_name()}" if is_gpu_available() else "💻 CPU"
        self.title(f"FilmDev — Analog Scan Processor  [{gpu_str}]")
        self.geometry("1100x750")
        self.minsize(800, 550)

        # ---- Core engines ----------------------------------------
        self.raw_loader = RawLoader()
        self.wb_engine = WBEngine()
        self.levels_engine = LevelsEngine()
        self.invert_engine = InvertEngine()
        self.exporter = Exporter()

        # ---- Application state -----------------------------------
        self.settings: Dict = dict(DEFAULT_SETTINGS)
        self.file_list: List[str] = []
        self.current_index: int = -1

        # Per-image WB state {filepath: {sample, multipliers}}
        self.wb_state: Dict[str, Dict] = {}

        # Raw array cache (current image only)
        self._raw_array: Optional[np.ndarray] = None
        self._preview_array: Optional[np.ndarray] = None  # WB/inverted preview

        # Mode: "neg" or "pos"
        self._mode = tk.StringVar(value="neg")

        # Filmstock selector
        self._filmstock_var = tk.StringVar(value="auto")

        # Label→key reverse map (for combo display)
        self._filmstock_label_to_key: dict = {}

        # Threading
        self._load_queue: queue.Queue = queue.Queue()
        self._batch_queue: queue.Queue = queue.Queue()
        self._processing = False

        self._build_ui()
        self._bind_keys()
        self._after_loop()

    # ==================================================================
    # UI Construction
    # ==================================================================

    def _build_ui(self) -> None:
        self._build_toolbar()
        self._build_main_area()
        self._build_nav_bar()
        self._build_log_area()

    def _build_toolbar(self) -> None:
        toolbar = ttk.Frame(self, padding=(6, 4))
        toolbar.pack(fill=tk.X, side=tk.TOP)

        ttk.Button(toolbar, text="📁 Open Folder", command=self._open_folder).pack(side=tk.LEFT, padx=(0, 8))

        # Mode toggle
        mode_frame = ttk.LabelFrame(toolbar, text="Film Mode", padding=(6, 2))
        mode_frame.pack(side=tk.LEFT, padx=8)
        ttk.Radiobutton(mode_frame, text="Negative", variable=self._mode,
                        value="neg", command=self._on_mode_change).pack(side=tk.LEFT)
        ttk.Radiobutton(mode_frame, text="Positive", variable=self._mode,
                        value="pos", command=self._on_mode_change).pack(side=tk.LEFT, padx=(8, 0))

        # WB mode button
        self._wb_btn = ttk.Button(toolbar, text="🎯 Click Film Mask", command=self._activate_wb_click)
        self._wb_btn.pack(side=tk.LEFT, padx=8)

        # ---- Filmstock selector ----------------------------------
        stock_frame = ttk.LabelFrame(toolbar, text="Film Stock", padding=(4, 2))
        stock_frame.pack(side=tk.LEFT, padx=8)

        labels = get_profile_labels()          # {key: label}
        label_list = list(labels.values())     # ordered display names
        self._filmstock_label_to_key = get_profile_keys_by_label()  # {label: key}

        self._filmstock_combo = ttk.Combobox(
            stock_frame,
            textvariable=self._filmstock_var,
            values=label_list,
            state="readonly",
            width=22,
        )
        # Store display label in var initially
        self._filmstock_var.set(FILMSTOCK_PROFILES["auto"]["label"])
        self._filmstock_combo.pack(side=tk.LEFT)
        self._filmstock_combo.bind("<<ComboboxSelected>>", self._on_filmstock_change)

        ttk.Button(toolbar, text="⚙ Settings", command=self._open_settings).pack(side=tk.RIGHT)

        # Folder path label
        self._folder_label = ttk.Label(toolbar, text="No folder selected", foreground="gray")
        self._folder_label.pack(side=tk.LEFT, padx=12)

    def _build_main_area(self) -> None:
        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # ---- File list (left panel) --------------------------------
        list_frame = ttk.LabelFrame(paned, text="Files", width=220)
        list_frame.pack_propagate(False)
        paned.add(list_frame, weight=1)

        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._file_listbox = tk.Listbox(
            list_frame,
            yscrollcommand=scrollbar.set,
            selectmode=tk.SINGLE,
            activestyle="dotbox",
            font=("Consolas", 9),
            bg="#1e1e1e",
            fg="#e0e0e0",
            selectbackground="#2d5a8e",
            selectforeground="white",
        )
        self._file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.configure(command=self._file_listbox.yview)
        self._file_listbox.bind("<<ListboxSelect>>", self._on_listbox_select)

        # ---- Image viewer (right panel) ----------------------------
        viewer_frame = ttk.LabelFrame(paned, text="Preview")
        paned.add(viewer_frame, weight=5)

        self.viewer = ImageViewer(
            viewer_frame,
            on_click=self._on_viewer_click,
        )
        self.viewer.pack(fill=tk.BOTH, expand=True)

    def _build_nav_bar(self) -> None:
        nav = ttk.Frame(self, padding=(6, 4))
        nav.pack(fill=tk.X, side=tk.BOTTOM)

        ttk.Button(nav, text="◀ Prev", command=self._prev_image).pack(side=tk.LEFT, padx=2)

        self._nav_label = ttk.Label(nav, text="No images loaded", width=20, anchor="center")
        self._nav_label.pack(side=tk.LEFT, padx=6)

        ttk.Button(nav, text="Next ▶", command=self._next_image).pack(side=tk.LEFT, padx=2)

        ttk.Separator(nav, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        self._copy_wb_btn = ttk.Button(nav, text="↑ Copy WB from Prev", command=self._copy_wb_from_prev,
                                       state=tk.DISABLED)
        self._copy_wb_btn.pack(side=tk.LEFT, padx=2)

        self._skip_btn = ttk.Button(nav, text="Skip", command=self._skip_image, state=tk.DISABLED)
        self._skip_btn.pack(side=tk.LEFT, padx=2)

        ttk.Separator(nav, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        self._process_btn = ttk.Button(
            nav, text="✅ Process All", command=self._start_batch, state=tk.DISABLED
        )
        self._process_btn.pack(side=tk.RIGHT, padx=4)

    def _build_log_area(self) -> None:
        log_frame = ttk.LabelFrame(self, text="Processing Log", padding=(4, 2))
        log_frame.pack(fill=tk.X, padx=4, pady=(0, 4))

        self._progress = ttk.Progressbar(log_frame, mode="determinate", length=300)
        self._progress.pack(side=tk.LEFT, padx=(0, 8), pady=2)
        self._progress_label = ttk.Label(log_frame, text="Idle")
        self._progress_label.pack(side=tk.LEFT)

        self._log_text = tk.Text(log_frame, height=4, state=tk.DISABLED,
                                 bg="#1a1a1a", fg="#aaaaaa", font=("Consolas", 8))
        self._log_text.pack(fill=tk.X, pady=(4, 0))

        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self._log_text.yview)
        self._log_text.configure(yscrollcommand=scrollbar.set)

    # ==================================================================
    # Key Bindings
    # ==================================================================

    def _bind_keys(self) -> None:
        self.bind("<Right>", lambda e: self._next_image())
        self.bind("<Left>", lambda e: self._prev_image())
        self.bind("<Return>", lambda e: self._next_image())
        self.bind("<r>", lambda e: self._activate_wb_click())
        self.bind("<R>", lambda e: self._activate_wb_click())
        self.bind("<s>", lambda e: self._skip_image())
        self.bind("<S>", lambda e: self._skip_image())

    # ==================================================================
    # Folder + File List
    # ==================================================================

    def _open_folder(self) -> None:
        folder = filedialog.askdirectory(title="Select folder with RAW files")
        if not folder:
            return

        files = RawLoader.list_raw_files(folder)
        if not files:
            messagebox.showwarning("No RAW files", f"No supported RAW files found in:\n{folder}")
            return

        self.file_list = files
        self.wb_state = {}
        self.current_index = -1

        self._folder_label.configure(text=os.path.basename(folder), foreground="white")

        self._file_listbox.delete(0, tk.END)
        for f in files:
            name = os.path.basename(f)
            self._file_listbox.insert(tk.END, name)

        self._log(f"Loaded {len(files)} RAW files from: {folder}")
        self._update_nav_label()

        # Navigate to first image
        self._go_to_index(0)

    def _on_listbox_select(self, event) -> None:
        sel = self._file_listbox.curselection()
        if sel:
            idx = sel[0]
            if idx != self.current_index:
                self._go_to_index(idx)

    # ==================================================================
    # Navigation
    # ==================================================================

    def _prev_image(self) -> None:
        if self.current_index > 0:
            self._go_to_index(self.current_index - 1)

    def _next_image(self) -> None:
        if self.current_index < len(self.file_list) - 1:
            self._go_to_index(self.current_index + 1)

    def _skip_image(self) -> None:
        """Skip current image (carry forward previous WB if available)."""
        if self.current_index < 0 or not self.file_list:
            return
        filepath = self.file_list[self.current_index]
        # Mark as skipped (no WB click required)
        if filepath not in self.wb_state:
            # Try to copy from previous
            self._copy_wb_from_prev(silent=True)
        self._log(f"Skipped: {os.path.basename(filepath)}")
        self._next_image()

    def _go_to_index(self, index: int) -> None:
        if index < 0 or index >= len(self.file_list):
            return

        self.current_index = index
        self._file_listbox.selection_clear(0, tk.END)
        self._file_listbox.selection_set(index)
        self._file_listbox.see(index)
        self._update_nav_label()
        self._update_button_states()

        # Load image in background thread
        filepath = self.file_list[index]
        self._load_image_async(filepath)

    def _update_nav_label(self) -> None:
        if not self.file_list:
            self._nav_label.configure(text="No images loaded")
        else:
            self._nav_label.configure(
                text=f"Image {self.current_index + 1} / {len(self.file_list)}"
            )

    def _update_button_states(self) -> None:
        has_files = bool(self.file_list)
        has_prev_wb = self._has_prev_wb()

        self._skip_btn.configure(state=tk.NORMAL if has_files else tk.DISABLED)
        self._copy_wb_btn.configure(state=tk.NORMAL if has_prev_wb else tk.DISABLED)

        # Enable batch when all images have WB (or at least some)
        wb_count = sum(1 for f in self.file_list if f in self.wb_state)
        if wb_count > 0:
            self._process_btn.configure(state=tk.NORMAL)
        else:
            self._process_btn.configure(state=tk.DISABLED)

    # ==================================================================
    # Image Loading (background thread)
    # ==================================================================

    def _load_image_async(self, filepath: str) -> None:
        """Load thumbnail in a background thread, then display."""
        self._set_progress_label("Loading preview…")

        def _worker():
            try:
                arr = self.raw_loader.load_thumbnail(filepath)
                if arr is None:
                    arr = np.zeros((100, 100, 3), dtype=np.float32)
                self._load_queue.put(("ok", filepath, arr))
            except Exception as e:
                logger.error("Load failed: %s", e)
                self._load_queue.put(("error", filepath, str(e)))

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

    def _process_load_queue(self) -> None:
        """Drain the load queue on the main thread."""
        try:
            while True:
                item = self._load_queue.get_nowait()
                status = item[0]
                filepath = item[1]

                if status == "error":
                    self._log(f"⚠ Load error: {item[2]}")
                    self._set_progress_label("Load failed")
                    continue

                arr = item[2]

                # Only display if still the current file
                if self.current_index >= 0 and self.file_list[self.current_index] == filepath:
                    self._raw_array = arr

                    # Apply existing WB if available
                    preview = self._apply_pipeline(arr, filepath)
                    self._preview_array = preview
                    self.viewer.display_array(preview)

                    # Show existing WB marker
                    wb = self.wb_state.get(filepath)
                    if wb:
                        self.viewer.update_wb_info(wb.get("sample"), wb.get("multipliers"))
                    else:
                        self.viewer.update_wb_info(None)
                        # Auto-activate click mode if settings say so or first image
                        if not self.settings.get("auto_skip_wb_set", False):
                            self.viewer.set_click_mode(True)

                    self._set_progress_label("Ready")
                    self._update_file_list_indicator(filepath)

        except queue.Empty:
            pass

    def _apply_pipeline(self, arr: np.ndarray, filepath: str, for_export: bool = False) -> np.ndarray:
        """Apply pipeline for preview or export.

        Correct order for colour negatives (Aaron Buchler / NLP standard):
          1. WB first   -- on the NEGATIVE (neutralises the orange mask)
          2. Invert     -- mask already neutral, inversion gives clean colours
          3. (Levels)   -- applied in batch export via levels_engine

        Why WB before inversion:
          The orange film mask is a uniform tint on the negative.  Balancing it
          to neutral grey BEFORE 1/x inversion means the formerly-orange areas
          become neutral after inversion instead of blue/cyan.  Inverting first
          turns orange into blue/cyan; a subsequent WB over-corrects other tones.

        User UX:
          User clicks on the orange mask IN THE NEGATIVE preview (pre-inversion).
          WB sample is taken from the negative with green-ref method (mult_G=1.0).
          Preview shows WB-neutralised negative; inverted result has no colour cast.
        """
        result = arr.copy()

        # Step 1: WB on the NEGATIVE -- MUST come before inversion
        wb = self.wb_state.get(filepath)
        if wb and "multipliers" in wb:
            result = self.wb_engine.apply(result, wb["multipliers"])

        # Step 2: Invert (neg mode) -- WB has already neutralised the orange mask
        if self._mode.get() == "neg":
            method = self.settings.get("inversion_method", "transmittance")
            result = self.invert_engine.invert(result, method=method)

        # Step 3: Filmstock profile (preview only; export applies it too)
        filmstock_key = self._get_filmstock_key()
        if filmstock_key != "auto":
            result = apply_profile(result, filmstock_key)

        return result

    def _update_file_list_indicator(self, filepath: str) -> None:
        """Mark processed files with indicator in listbox."""
        try:
            idx = self.file_list.index(filepath)
            name = os.path.basename(filepath)
            has_wb = filepath in self.wb_state
            marker = "✓ " if has_wb else "  "
            self._file_listbox.delete(idx)
            self._file_listbox.insert(idx, marker + name)
            self._file_listbox.selection_set(idx)
        except (ValueError, tk.TclError):
            pass

    # ==================================================================
    # WB Click Handling
    # ==================================================================

    def _activate_wb_click(self) -> None:
        """Activate click-to-sample mode."""
        self.viewer.set_click_mode(True)

    def _on_viewer_click(self, orig_x: int, orig_y: int, image_array: np.ndarray) -> None:
        """
        Called when user clicks on the film mask in the viewer.
        Computes WB and updates the preview.
        """
        filepath = self.file_list[self.current_index] if self.current_index >= 0 else None
        if filepath is None:
            return

        radius = self.settings.get("sample_radius", 7)

        # Sample from the RAW (pre-WB) array for correct multiplier computation
        source = self._raw_array if self._raw_array is not None else image_array

        sample = self.wb_engine.sample_point(source, orig_x, orig_y, radius=radius)
        mults = self.wb_engine.compute_multipliers(sample)

        # Store WB state
        self.wb_state[filepath] = {
            "sample": sample,
            "multipliers": mults,
            "click_x": orig_x,
            "click_y": orig_y,
        }

        # Update UI
        self.viewer.update_wb_info(sample, mults)
        self.viewer.set_click_mode(False)

        # Re-render preview with WB applied
        if self._raw_array is not None:
            preview = self._apply_pipeline(self._raw_array, filepath)
            self._preview_array = preview
            self.viewer.display_array(preview)
            self.viewer.show_marker(orig_x, orig_y)

        self._log(f"WB sampled at ({orig_x},{orig_y}) for {os.path.basename(filepath)}")
        self._update_file_list_indicator(filepath)
        self._update_button_states()

    def _copy_wb_from_prev(self, silent: bool = False) -> None:
        """Copy WB multipliers from the previous image that has WB set."""
        if self.current_index <= 0:
            return

        prev_wb = None
        for i in range(self.current_index - 1, -1, -1):
            fp = self.file_list[i]
            if fp in self.wb_state:
                prev_wb = self.wb_state[fp]
                break

        if prev_wb is None:
            if not silent:
                messagebox.showinfo("No previous WB", "No previous WB sample found.")
            return

        filepath = self.file_list[self.current_index]
        self.wb_state[filepath] = dict(prev_wb)

        self.viewer.update_wb_info(prev_wb.get("sample"), prev_wb.get("multipliers"))

        if self._raw_array is not None:
            preview = self._apply_pipeline(self._raw_array, filepath)
            self._preview_array = preview
            self.viewer.display_array(preview)

        if not silent:
            self._log(f"Copied WB from previous to {os.path.basename(filepath)}")

        self._update_button_states()
        self._update_file_list_indicator(filepath)

    def _has_prev_wb(self) -> bool:
        if self.current_index <= 0:
            return False
        for i in range(self.current_index - 1, -1, -1):
            if self.file_list[i] in self.wb_state:
                return True
        return False

    # ==================================================================
    # Filmstock helpers
    # ==================================================================

    def _get_filmstock_key(self) -> str:
        """Return the profile key for the currently selected filmstock label."""
        label = self._filmstock_var.get()
        return self._filmstock_label_to_key.get(label, "auto")

    def _on_filmstock_change(self, event=None) -> None:
        """Called when the filmstock dropdown selection changes."""
        key = self._get_filmstock_key()
        logger.debug("Filmstock changed to: %s", key)
        # Re-render preview
        if self._raw_array is not None and self.current_index >= 0:
            filepath = self.file_list[self.current_index]
            preview = self._apply_pipeline(self._raw_array, filepath)
            self._preview_array = preview
            self.viewer.display_array(preview)

    # ==================================================================
    # Mode Change
    # ==================================================================

    def _on_mode_change(self) -> None:
        """Re-render current preview when mode changes."""
        if self._raw_array is not None and self.current_index >= 0:
            filepath = self.file_list[self.current_index]
            preview = self._apply_pipeline(self._raw_array, filepath)
            self._preview_array = preview
            self.viewer.display_array(preview)

    # ==================================================================
    # Batch Processing
    # ==================================================================

    def _start_batch(self) -> None:
        """Start batch processing in a background thread."""
        if self._processing:
            messagebox.showwarning("Processing", "A batch job is already running.")
            return

        # Gather jobs (files with WB set)
        jobs = []
        skipped = []
        for fp in self.file_list:
            wb = self.wb_state.get(fp)
            if wb:
                jobs.append({"source_path": fp, "wb": wb})
            else:
                skipped.append(fp)

        if not jobs:
            messagebox.showwarning("No WB Data", "Please set WB for at least one image before processing.")
            return

        if skipped:
            msg = (
                f"{len(skipped)} image(s) have no WB set and will be skipped.\n\n"
                f"Process {len(jobs)} image(s)?"
            )
            if not messagebox.askyesno("Confirm Batch", msg):
                return

        self._processing = True
        self._process_btn.configure(state=tk.DISABLED, text="Processing…")
        self._progress["maximum"] = len(jobs)
        self._progress["value"] = 0

        # Update exporter settings
        self.exporter.set_options(
            export_tiff=self.settings.get("export_tiff", True),
            export_jpeg=self.settings.get("export_jpeg", False),
            jpeg_quality=self.settings.get("jpeg_quality", 92),
        )

        def _batch_worker():
            success = []
            failed = []
            mode = self._mode.get()
            inv_method = self.settings.get("inversion_method", "transmittance")
            levels_method = self.settings.get("levels_method", "percentile")
            bp_pct = self.settings.get("bp_percentile", 0.5)
            wp_pct = self.settings.get("wp_percentile", 99.5)
            per_ch = self.settings.get("per_channel_levels", False)

            for i, job in enumerate(jobs):
                fp = job["source_path"]
                fname = os.path.basename(fp)
                self._batch_queue.put(("progress", i + 1, len(jobs), f"Processing {fname}…"))

                try:
                    # Load full RAW
                    arr = self.raw_loader.load_full(fp)

                    # Step 1: WB on the NEGATIVE -- MUST come before inversion
                    # (Aaron Buchler / NLP standard: WB first on neg, then invert)
                    mults = job["wb"]["multipliers"]
                    arr = self.wb_engine.apply(arr, mults)

                    # Step 2: Invert (neg mode) -- mask already neutralised by WB
                    if mode == "neg":
                        arr = self.invert_engine.invert(arr, method=inv_method)

                    # Auto levels
                    if levels_method == "iterative":
                        arr = self.levels_engine.auto_levels_iterative(arr)
                    else:
                        arr = self.levels_engine.auto_levels_percentile(
                            arr,
                            bp_percentile=bp_pct,
                            wp_percentile=wp_pct,
                            per_channel=per_ch,
                        )

                    # Step 3: Filmstock profile (batch)
                    filmstock_key = self._get_filmstock_key()
                    if filmstock_key != "auto":
                        arr = apply_profile(arr, filmstock_key)

                    # Export
                    written = self.exporter.export(arr, fp)
                    success.extend(written)
                    self._batch_queue.put(("log", f"✓ {fname} → {', '.join(os.path.basename(w) for w in written)}"))

                    # Scan log (ML training data)
                    try:
                        mask_rgb = job["wb"].get("sample", [0.0, 0.0, 0.0])
                        wb_mults_log = mults if isinstance(mults, (list, np.ndarray)) else [1.0, 1.0, 1.0]
                        log_scan(
                            filepath=fp,
                            filmstock=filmstock_key,
                            mask_rgb=mask_rgb,
                            wb_mults=wb_mults_log,
                        )
                    except Exception as log_exc:
                        logger.debug("scan_log non-fatal error: %s", log_exc)

                except Exception as e:
                    logger.error("Batch export failed for %s: %s", fname, e)
                    failed.append((fp, str(e)))
                    self._batch_queue.put(("log", f"✗ {fname}: {e}"))
                finally:
                    try:
                        del arr
                    except UnboundLocalError:
                        pass
                    gc.collect()

            self._batch_queue.put(("done", success, failed))

        t = threading.Thread(target=_batch_worker, daemon=True)
        t.start()

    def _process_batch_queue(self) -> None:
        """Drain batch results queue on the main thread."""
        try:
            while True:
                item = self._batch_queue.get_nowait()
                kind = item[0]

                if kind == "progress":
                    _, current, total, msg = item
                    self._progress["value"] = current
                    self._set_progress_label(msg)

                elif kind == "log":
                    self._log(item[1])

                elif kind == "done":
                    _, success, failed = item
                    self._processing = False
                    self._process_btn.configure(state=tk.NORMAL, text="✅ Process All")
                    self._set_progress_label(f"Done: {len(success)} files exported")
                    self._log(
                        f"Batch complete: {len(success)} exported, {len(failed)} failed."
                    )
                    output_dir = None
                    if success:
                        output_dir = os.path.dirname(success[0])
                    msg = (
                        f"✅ Batch complete!\n\n"
                        f"Exported: {len(success)} files\n"
                        f"Failed:   {len(failed)} files\n"
                    )
                    if output_dir:
                        msg += f"\nOutput: {output_dir}"
                    messagebox.showinfo("Batch Done", msg)
                    # Open output folder in Explorer automatically
                    if output_dir and os.path.isdir(output_dir):
                        import subprocess, platform
                        try:
                            if platform.system() == "Windows":
                                subprocess.Popen(["explorer", output_dir])
                            elif platform.system() == "Darwin":
                                subprocess.Popen(["open", output_dir])
                            else:
                                subprocess.Popen(["xdg-open", output_dir])
                        except Exception:
                            pass

        except queue.Empty:
            pass

    # ==================================================================
    # Settings
    # ==================================================================

    def _open_settings(self) -> None:
        def _on_apply(new_settings):
            self.settings = new_settings
            self._log("Settings updated.")
            # Re-render preview with new settings
            self._on_mode_change()

        SettingsDialog(self, self.settings, on_apply=_on_apply)

    # ==================================================================
    # Periodic after() loop
    # ==================================================================

    def _after_loop(self) -> None:
        """Poll queues every 50ms on the main thread."""
        self._process_load_queue()
        self._process_batch_queue()
        self.after(50, self._after_loop)

    # ==================================================================
    # Logging helpers
    # ==================================================================

    def _log(self, msg: str) -> None:
        self._log_text.configure(state=tk.NORMAL)
        self._log_text.insert(tk.END, msg + "\n")
        self._log_text.see(tk.END)
        self._log_text.configure(state=tk.DISABLED)
        logger.info(msg)

    def _set_progress_label(self, text: str) -> None:
        self._progress_label.configure(text=text)
