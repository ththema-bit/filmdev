"""
image_viewer.py — Canvas-based image viewer with click-to-sample functionality.

Features:
  - Displays a numpy float32 or PIL image scaled to fit the canvas.
  - Tracks click coordinates and maps canvas → original pixel coords.
  - Shows a crosshair marker at the sampled point.
  - Supports cursor change for "waiting for click" mode.
  - Zooms fit-to-canvas on resize.
"""

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional, Tuple

import numpy as np


class ImageViewer(tk.Frame):
    """
    A Tkinter Frame containing a Canvas for displaying images and
    handling WB-sample clicks.
    """

    MARKER_SIZE = 10          # crosshair arm length in canvas pixels
    MARKER_COLOR = "#00FF44"  # bright green
    SAMPLE_BOX_COLOR = "#FFCC00"  # yellow sample region indicator

    def __init__(
        self,
        parent,
        on_click: Optional[Callable[[int, int, np.ndarray], None]] = None,
        **kwargs,
    ):
        """
        Args:
            parent:   Parent Tkinter widget.
            on_click: Callback(orig_x, orig_y, image_array) when user clicks
                      in mask-selection mode.
        """
        super().__init__(parent, **kwargs)
        self.on_click = on_click

        self._image_array: Optional[np.ndarray] = None  # float32 [0,1] (H,W,3)
        self._tk_image = None                            # PhotoImage kept alive
        self._scale: float = 1.0                        # canvas/original ratio (fit)
        self._zoom: float = 1.0                          # extra zoom multiplier (scroll)
        self._zoom_cx: float = 0.5                       # zoom anchor x (0-1 of canvas)
        self._zoom_cy: float = 0.5                       # zoom anchor y
        self._pan_x: int = 0                             # pan offset x
        self._pan_y: int = 0                             # pan offset y
        self._pan_last: tuple = (0, 0)                   # last drag pos
        self._offset_x: int = 0                         # canvas left margin
        self._offset_y: int = 0                         # canvas top margin
        self._orig_w: int = 0
        self._orig_h: int = 0

        self._click_mode: bool = False      # if True, next click samples WB
        self._marker_ids: list = []         # canvas item IDs for the crosshair

        self._build_ui()

    # ------------------------------------------------------------------
    # UI layout
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self.canvas = tk.Canvas(self, bg="#1a1a1a", cursor="arrow")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Info bar at bottom
        info_frame = ttk.Frame(self)
        info_frame.pack(fill=tk.X)

        self._wb_label = ttk.Label(info_frame, text="WB Sample: —")
        self._wb_label.pack(side=tk.LEFT, padx=8)

        self._status_label = ttk.Label(info_frame, text="○ Waiting for image", foreground="gray")
        self._status_label.pack(side=tk.RIGHT, padx=8)

        # Bindings
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<Configure>", self._on_resize)
        # Scroll zoom (Windows + Linux)
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)        # Windows
        self.canvas.bind("<Button-4>", self._on_mousewheel)           # Linux scroll up
        self.canvas.bind("<Button-5>", self._on_mousewheel)           # Linux scroll down
        # Pan with middle mouse or right drag
        self.canvas.bind("<ButtonPress-2>", self._on_pan_start)
        self.canvas.bind("<B2-Motion>", self._on_pan_move)
        self.canvas.bind("<ButtonPress-3>", self._on_pan_start)
        self.canvas.bind("<B3-Motion>", self._on_pan_move)
        # Double-click resets zoom
        self.canvas.bind("<Double-Button-1>", self._on_zoom_reset)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def display_array(self, image: np.ndarray) -> None:
        """
        Display a float32 (H,W,3) numpy array.
        Scales to fit the current canvas size.
        """
        self._image_array = image
        self._orig_h, self._orig_w = image.shape[:2]
        # Reset zoom when a new image is loaded
        self._zoom = 1.0
        self._pan_x = 0
        self._pan_y = 0
        self._render()

    def set_click_mode(self, active: bool) -> None:
        """Switch cursor and status to indicate waiting for mask click."""
        self._click_mode = active
        if active:
            self.canvas.configure(cursor="crosshair")
            self._set_status("● Click on the film mask border", "#FF9900")
        else:
            self.canvas.configure(cursor="arrow")
            self._set_status("○ Navigation mode", "gray")

    def update_wb_info(
        self,
        sample: Optional[np.ndarray] = None,
        mults: Optional[np.ndarray] = None,
    ) -> None:
        """Update the WB info bar with the latest sample / multiplier values."""
        if sample is None:
            self._wb_label.configure(text="WB Sample: —")
            return
        r, g, b = sample
        text = f"WB Sample  R:{r:.3f}  G:{g:.3f}  B:{b:.3f}"
        if mults is not None:
            mr, mg, mb = mults
            text += f"  │  Mults  R:×{mr:.2f}  G:×{mg:.2f}  B:×{mb:.2f}"
        self._wb_label.configure(text=text)

    def show_marker(self, orig_x: int, orig_y: int) -> None:
        """Draw a crosshair marker at the given original-image coordinates."""
        self._clear_markers()
        cx, cy = self._orig_to_canvas(orig_x, orig_y)
        s = self.MARKER_SIZE
        c = self.MARKER_COLOR

        ids = [
            self.canvas.create_line(cx - s, cy, cx + s, cy, fill=c, width=2),
            self.canvas.create_line(cx, cy - s, cx, cy + s, fill=c, width=2),
            self.canvas.create_oval(cx - 3, cy - 3, cx + 3, cy + 3, outline=c, width=2),
        ]
        self._marker_ids = ids

    def clear_marker(self) -> None:
        self._clear_markers()

    def get_displayed_array(self) -> Optional[np.ndarray]:
        return self._image_array

    # ------------------------------------------------------------------
    # Internal rendering
    # ------------------------------------------------------------------

    def _render(self) -> None:
        """Convert numpy array to PhotoImage and display on canvas."""
        if self._image_array is None:
            return

        from PIL import Image, ImageTk

        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10 or ch < 10:
            # Canvas not yet realized; defer
            self.after(50, self._render)
            return

        # Fit-to-canvas base scale
        scale_x = cw / self._orig_w
        scale_y = ch / self._orig_h
        self._scale = min(scale_x, scale_y)

        # Combined scale including zoom
        combined = self._scale * self._zoom
        new_w = int(self._orig_w * combined)
        new_h = int(self._orig_h * combined)

        # Center when fit; use pan offset when zoomed in
        if self._zoom <= 1.0:
            self._offset_x = (cw - new_w) // 2
            self._offset_y = (ch - new_h) // 2
            self._pan_x = 0
            self._pan_y = 0
        else:
            self._offset_x = int((cw - new_w) // 2 + self._pan_x)
            self._offset_y = int((ch - new_h) // 2 + self._pan_y)

        # Convert float32 → uint8
        img8 = (np.clip(self._image_array, 0.0, 1.0) * 255.0).astype(np.uint8)
        pil_img = Image.fromarray(img8, "RGB")
        pil_img = pil_img.resize((new_w, new_h), Image.BILINEAR)  # BILINEAR faster for live zoom

        self._tk_image = ImageTk.PhotoImage(pil_img)
        self.canvas.delete("all")
        self._marker_ids = []
        self.canvas.create_image(
            self._offset_x, self._offset_y,
            anchor=tk.NW,
            image=self._tk_image,
        )

        # Zoom indicator (show when zoomed)
        if self._zoom > 1.05:
            pct = int(self._zoom * 100)
            self.canvas.create_text(
                cw - 8, ch - 8, anchor=tk.SE,
                text=f"{pct}%  (dblclick = reset)",
                fill="#a855f7", font=("Consolas", 9),
            )

    def _on_resize(self, event) -> None:
        if self._image_array is not None:
            self.after(20, self._render)  # debounce

    def _on_mousewheel(self, event) -> None:
        """Scroll to zoom. Anchor under cursor."""
        if self._image_array is None:
            return
        # Delta: Windows gives ±120 units, Linux Button-4/5
        if event.num == 4:
            delta = 1
        elif event.num == 5:
            delta = -1
        else:
            delta = 1 if event.delta > 0 else -1

        factor = 1.15 if delta > 0 else (1 / 1.15)
        new_zoom = max(0.5, min(8.0, self._zoom * factor))

        # Anchor zoom to mouse position
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw > 0 and ch > 0:
            mx = event.x - cw / 2
            my = event.y - ch / 2
            scale_change = new_zoom / self._zoom
            self._pan_x = mx + (self._pan_x - mx) * scale_change
            self._pan_y = my + (self._pan_y - my) * scale_change

        self._zoom = new_zoom
        self._render()

    def _on_zoom_reset(self, event) -> None:
        """Double-click resets zoom to fit."""
        self._zoom = 1.0
        self._pan_x = 0
        self._pan_y = 0
        self._render()

    def _on_pan_start(self, event) -> None:
        self._pan_last = (event.x, event.y)

    def _on_pan_move(self, event) -> None:
        if self._zoom > 1.0:
            dx = event.x - self._pan_last[0]
            dy = event.y - self._pan_last[1]
            self._pan_x += dx
            self._pan_y += dy
            self._pan_last = (event.x, event.y)
            self._render()

    def _on_canvas_click(self, event) -> None:
        if self._image_array is None:
            return

        # Convert canvas coords → original image coords
        orig_x, orig_y = self._canvas_to_orig(event.x, event.y)

        # Bounds check
        if not (0 <= orig_x < self._orig_w and 0 <= orig_y < self._orig_h):
            return

        if self._click_mode and self.on_click:
            self.show_marker(orig_x, orig_y)
            self._set_status(f"● Sampled at ({orig_x}, {orig_y})", "#00CC44")
            self.on_click(orig_x, orig_y, self._image_array)

    # ------------------------------------------------------------------
    # Coordinate mapping
    # ------------------------------------------------------------------

    def _canvas_to_orig(self, cx: int, cy: int) -> Tuple[int, int]:
        """Map canvas pixel → original image pixel (zoom+pan aware)."""
        combined = self._scale * self._zoom
        if combined == 0:
            return 0, 0
        ox = int((cx - self._offset_x) / combined)
        oy = int((cy - self._offset_y) / combined)
        return ox, oy

    def _orig_to_canvas(self, ox: int, oy: int) -> Tuple[int, int]:
        """Map original image pixel → canvas pixel (zoom+pan aware)."""
        combined = self._scale * self._zoom
        cx = int(ox * combined) + self._offset_x
        cy = int(oy * combined) + self._offset_y
        return cx, cy

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _clear_markers(self) -> None:
        for mid in self._marker_ids:
            try:
                self.canvas.delete(mid)
            except tk.TclError:
                pass
        self._marker_ids = []

    def _set_status(self, text: str, color: str = "gray") -> None:
        self._status_label.configure(text=text, foreground=color)
