#!/usr/bin/env python3
"""
filmdev.py — FilmDev: Analog Scan Processor
Entry point. Configures logging and launches the Tkinter application.

Usage:
    python filmdev.py
    python filmdev.py --debug     (verbose logging)
"""

import argparse
import logging
import sys


def _configure_logging(debug: bool = False) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    # Suppress noisy library loggers unless in debug mode
    if not debug:
        logging.getLogger("PIL").setLevel(logging.WARNING)
        logging.getLogger("rawpy").setLevel(logging.WARNING)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FilmDev — Analog Scan Processor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Workflow:
  1. Click 'Open Folder' and select a folder with RAW files.
  2. Choose 'Negative' or 'Positive' mode.
  3. For each image, click '🎯 Click Film Mask' then click on the film border.
  4. Press 'Next ▶' (or →) to advance to the next image.
  5. When all WB points are set, click '✅ Process All'.

Keyboard shortcuts:
  ← / →  Navigate images
  Enter   Advance to next image
  R       Re-activate mask click mode
  S       Skip current image
""",
    )
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging")
    args = parser.parse_args()

    _configure_logging(args.debug)
    logger = logging.getLogger("filmdev")
    logger.info("FilmDev starting…")

    try:
        from ui.main_window import MainWindow
    except ImportError as e:
        print(f"ERROR: Could not import UI: {e}", file=sys.stderr)
        print("Make sure you are running from the filmdev/ directory.", file=sys.stderr)
        sys.exit(1)

    try:
        app = MainWindow()
        app.mainloop()
    except KeyboardInterrupt:
        logger.info("Exiting (KeyboardInterrupt).")
    except Exception as e:
        logger.exception("Unhandled exception: %s", e)
        sys.exit(1)

    logger.info("FilmDev closed.")


if __name__ == "__main__":
    main()
