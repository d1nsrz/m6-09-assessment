"""
cli.py — standardised CLI for the leaderboard runner.

Usage inside the container:
    python /app/app/cli.py info
    python /app/app/cli.py predict
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (fixed by the spec)
# ---------------------------------------------------------------------------
STUDENT_JSON = Path("/app/STUDENT.json")
MODEL_PATH   = Path("/app/models/best.onnx")
INPUT_DIR    = Path("/data/input")
OUTPUT_DIR   = Path("/data/output")
OUTPUT_CSV   = OUTPUT_DIR / "predictions.csv"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

CSV_HEADER = ["image_path", "xmin", "ymin", "xmax", "ymax", "confidence", "class"]


# ---------------------------------------------------------------------------
# Subcommand: info
# ---------------------------------------------------------------------------

def cmd_info() -> None:
    """Print STUDENT.json to stdout and exit 0."""
    print(STUDENT_JSON.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Subcommand: predict
# ---------------------------------------------------------------------------

def cmd_predict() -> None:
    """Run the detector on every image under /data/input and write the CSV."""
    # Late import so `info` works even if onnxruntime isn't installed (unlikely,
    # but keeps the dependency surface clear).
    from app.detector import CatDetector  # noqa: PLC0415

    detector = CatDetector(
        onnx_path=MODEL_PATH,
        imgsz=640,
        conf_threshold=0.25,
        class_names=("cat",),
    )

    # Collect all image paths, preserving relative structure
    image_paths: list[Path] = sorted(
        p
        for p in INPUT_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )

    if not image_paths:
        print("[warn] No images found in /data/input", file=sys.stderr)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_HEADER)
        writer.writeheader()

        for img_path in image_paths:
            rel = img_path.relative_to(INPUT_DIR).as_posix()

            try:
                boxes = detector.predict(img_path)
            except Exception as exc:  # noqa: BLE001
                print(f"[error] {rel}: {exc}", file=sys.stderr)
                # Write an empty-detection row so the image appears in results
                writer.writerow({"image_path": rel, "xmin": "", "ymin": "",
                                 "xmax": "", "ymax": "", "confidence": "", "class": ""})
                continue

            if not boxes:
                # No detections: single row with empty bbox fields
                writer.writerow({"image_path": rel, "xmin": "", "ymin": "",
                                 "xmax": "", "ymax": "", "confidence": "", "class": ""})
            else:
                for box in boxes:
                    writer.writerow(
                        {
                            "image_path": rel,
                            "xmin":       f"{box['xmin']:.4f}",
                            "ymin":       f"{box['ymin']:.4f}",
                            "xmax":       f"{box['xmax']:.4f}",
                            "ymax":       f"{box['ymax']:.4f}",
                            "confidence": f"{box['confidence']:.6f}",
                            "class":      box["class"],
                        }
                    )

    print(f"[info] Wrote {OUTPUT_CSV}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: cli.py <info|predict>", file=sys.stderr)
        sys.exit(1)

    subcmd = sys.argv[1].lower()

    if subcmd == "info":
        cmd_info()
    elif subcmd == "predict":
        cmd_predict()
    else:
        print(f"Unknown subcommand: {subcmd!r}. Expected 'info' or 'predict'.",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
