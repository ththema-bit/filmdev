"""
settings_dialog.py — Modal settings popup for FilmDev.
"""

import tkinter as tk
from tkinter import ttk
from typing import Callable, Dict, Any


DEFAULT_SETTINGS: Dict[str, Any] = {
    "levels_method": "percentile",   # "percentile" | "iterative"
    "bp_percentile": 0.5,
    "wp_percentile": 99.5,
    "inversion_method": "linear",    # "linear" | "log"
    "export_tiff": True,
    "export_jpeg": False,
    "jpeg_quality": 92,
    "sample_radius": 7,              # patch half-size in pixels
    "auto_skip_wb_set": False,       # skip WB click if already set
    "per_channel_levels": False,
    "filmstock": "auto",             # default filmstock profile key
}


class SettingsDialog(tk.Toplevel):
    """Modal dialog for application settings."""

    def __init__(self, parent: tk.Tk, settings: Dict[str, Any], on_apply: Callable[[Dict[str, Any]], None]):
        super().__init__(parent)
        self.title("FilmDev — Settings")
        self.resizable(False, False)
        self.grab_set()  # modal
        self.on_apply = on_apply

        # Work on a copy
        self.settings = dict(settings)

        self._build_ui()
        self._center_on_parent(parent)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        pad = {"padx": 10, "pady": 6}
        s = self.settings

        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # ---- Tab: Levels -------------------------------------------
        tab_levels = ttk.Frame(notebook)
        notebook.add(tab_levels, text="Levels")

        # Levels method
        ttk.Label(tab_levels, text="Auto BP/WP Method:").grid(row=0, column=0, sticky="w", **pad)
        self._levels_method = tk.StringVar(value=s["levels_method"])
        frame_lvl = ttk.Frame(tab_levels)
        frame_lvl.grid(row=0, column=1, sticky="w", **pad)
        ttk.Radiobutton(frame_lvl, text="Percentile (fast)", variable=self._levels_method,
                        value="percentile", command=self._on_levels_change).pack(side=tk.LEFT)
        ttk.Radiobutton(frame_lvl, text="Iterative (no-clip)", variable=self._levels_method,
                        value="iterative", command=self._on_levels_change).pack(side=tk.LEFT, padx=(12, 0))

        # Percentile frame
        self._pct_frame = ttk.LabelFrame(tab_levels, text="Percentile Settings")
        self._pct_frame.grid(row=1, column=0, columnspan=2, sticky="ew", **pad)

        ttk.Label(self._pct_frame, text="Black Point %:").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        self._bp_pct = tk.DoubleVar(value=s["bp_percentile"])
        bp_spin = ttk.Spinbox(self._pct_frame, from_=0.0, to=10.0, increment=0.1,
                              textvariable=self._bp_pct, width=8, format="%.1f")
        bp_spin.grid(row=0, column=1, sticky="w", padx=6, pady=4)

        ttk.Label(self._pct_frame, text="White Point %:").grid(row=1, column=0, sticky="w", padx=6, pady=4)
        self._wp_pct = tk.DoubleVar(value=s["wp_percentile"])
        wp_spin = ttk.Spinbox(self._pct_frame, from_=90.0, to=100.0, increment=0.1,
                              textvariable=self._wp_pct, width=8, format="%.1f")
        wp_spin.grid(row=1, column=1, sticky="w", padx=6, pady=4)

        # Per-channel
        self._per_channel = tk.BooleanVar(value=s["per_channel_levels"])
        ttk.Checkbutton(self._pct_frame, text="Per-channel (independent R/G/B)",
                        variable=self._per_channel).grid(row=2, column=0, columnspan=2,
                                                          sticky="w", padx=6, pady=4)

        # ---- Tab: Inversion ----------------------------------------
        tab_inv = ttk.Frame(notebook)
        notebook.add(tab_inv, text="Inversion")

        ttk.Label(tab_inv, text="Inversion Method:").grid(row=0, column=0, sticky="w", **pad)
        self._inv_method = tk.StringVar(value=s["inversion_method"])
        frame_inv = ttk.Frame(tab_inv)
        frame_inv.grid(row=0, column=1, sticky="w", **pad)
        ttk.Radiobutton(frame_inv, text="Transmittance (1/x) — correct", variable=self._inv_method, value="transmittance").pack(side=tk.LEFT)
        ttk.Radiobutton(frame_inv, text="Linear  (1–x)", variable=self._inv_method, value="linear").pack(side=tk.LEFT, padx=(10, 0))
        ttk.Radiobutton(frame_inv, text="Log  (–ln x)", variable=self._inv_method, value="log").pack(side=tk.LEFT, padx=(10, 0))

        ttk.Label(tab_inv, text="(Log inversion gives better tonal compression\nfor high-density negatives.)",
                  foreground="gray", justify=tk.LEFT).grid(row=1, column=0, columnspan=2, sticky="w", **pad)

        # ---- Tab: Export -------------------------------------------
        tab_exp = ttk.Frame(notebook)
        notebook.add(tab_exp, text="Export")

        self._exp_tiff = tk.BooleanVar(value=s["export_tiff"])
        ttk.Checkbutton(tab_exp, text="Export 16-bit TIFF", variable=self._exp_tiff).grid(
            row=0, column=0, columnspan=2, sticky="w", **pad)

        self._exp_jpeg = tk.BooleanVar(value=s["export_jpeg"])
        ttk.Checkbutton(tab_exp, text="Export JPEG preview", variable=self._exp_jpeg,
                        command=self._on_jpeg_toggle).grid(row=1, column=0, columnspan=2, sticky="w", **pad)

        ttk.Label(tab_exp, text="JPEG Quality:").grid(row=2, column=0, sticky="w", **pad)
        self._jpeg_q = tk.IntVar(value=s["jpeg_quality"])
        self._jpeg_spin = ttk.Spinbox(tab_exp, from_=50, to=100, increment=1,
                                      textvariable=self._jpeg_q, width=6)
        self._jpeg_spin.grid(row=2, column=1, sticky="w", **pad)
        self._on_jpeg_toggle()  # sync state

        # ---- Tab: Sampling -----------------------------------------
        tab_smp = ttk.Frame(notebook)
        notebook.add(tab_smp, text="Sampling")

        ttk.Label(tab_smp, text="WB Sample Radius (px):").grid(row=0, column=0, sticky="w", **pad)
        self._sample_radius = tk.IntVar(value=s["sample_radius"])
        radius_frame = ttk.Frame(tab_smp)
        radius_frame.grid(row=0, column=1, sticky="w", **pad)
        for label, val in [("5×5 (2px)", 2), ("15×15 (7px)", 7), ("25×25 (12px)", 12)]:
            ttk.Radiobutton(radius_frame, text=label, variable=self._sample_radius, value=val).pack(side=tk.LEFT, padx=4)

        self._auto_skip = tk.BooleanVar(value=s["auto_skip_wb_set"])
        ttk.Checkbutton(tab_smp, text="Auto-skip WB click if already set for this image",
                        variable=self._auto_skip).grid(row=1, column=0, columnspan=2, sticky="w", **pad)

        # ---- Buttons -----------------------------------------------
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        ttk.Button(btn_frame, text="Cancel", command=self.destroy).pack(side=tk.RIGHT, padx=4)
        ttk.Button(btn_frame, text="Apply", command=self._apply).pack(side=tk.RIGHT, padx=4)
        ttk.Button(btn_frame, text="Reset Defaults", command=self._reset).pack(side=tk.LEFT, padx=4)

        self._on_levels_change()

    def _on_levels_change(self) -> None:
        state = tk.NORMAL if self._levels_method.get() == "percentile" else tk.DISABLED
        for child in self._pct_frame.winfo_children():
            try:
                child.configure(state=state)
            except tk.TclError:
                pass

    def _on_jpeg_toggle(self) -> None:
        state = tk.NORMAL if self._exp_jpeg.get() else tk.DISABLED
        try:
            self._jpeg_spin.configure(state=state)
        except AttributeError:
            pass

    def _apply(self) -> None:
        self.settings.update({
            "levels_method": self._levels_method.get(),
            "bp_percentile": self._bp_pct.get(),
            "wp_percentile": self._wp_pct.get(),
            "inversion_method": self._inv_method.get(),
            "export_tiff": self._exp_tiff.get(),
            "export_jpeg": self._exp_jpeg.get(),
            "jpeg_quality": self._jpeg_q.get(),
            "sample_radius": self._sample_radius.get(),
            "auto_skip_wb_set": self._auto_skip.get(),
            "per_channel_levels": self._per_channel.get(),
        })
        self.on_apply(self.settings)
        self.destroy()

    def _reset(self) -> None:
        """Reset to defaults."""
        self.settings = dict(DEFAULT_SETTINGS)
        self.destroy()
        # Re-open with defaults (caller will re-instantiate if needed)
        # For simplicity, just apply defaults
        self.on_apply(dict(DEFAULT_SETTINGS))

    def _center_on_parent(self, parent: tk.Tk) -> None:
        self.update_idletasks()
        pw = parent.winfo_width()
        ph = parent.winfo_height()
        px = parent.winfo_x()
        py = parent.winfo_y()
        w = self.winfo_width()
        h = self.winfo_height()
        x = px + (pw - w) // 2
        y = py + (ph - h) // 2
        self.geometry(f"+{x}+{y}")
