"""
detector.py — wraps best.onnx for single-class cat detection.

Supports the default YOLO26 end-to-end ONNX output shape: (1, 300, 6)
  columns: [x1, y1, x2, y2, score, class_idx]

If you exported with end2end=False the shape will be (1, nc+4, 8400) and
you would need to add NMS here. Stick with the e2e export to avoid that.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

import numpy as np
import onnxruntime as ort
from PIL import Image


class CatDetector:
    """Load an ONNX model once, expose predict() for per-image inference."""

    def __init__(
        self,
        onnx_path: str | Path,
        imgsz: int = 640,
        conf_threshold: float = 0.25,
        class_names: tuple[str, ...] = ("cat",),
    ) -> None:
        self.imgsz = imgsz
        self.conf = conf_threshold
        self.class_names = class_names

        self.session = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )

        # Validate output shape so surprises surface early
        out_shape = self.session.get_outputs()[0].shape
        # Expect (batch, 300, 6) for e2e head
        if len(out_shape) != 3 or out_shape[2] != 6:
            raise RuntimeError(
                f"Unexpected ONNX output shape {out_shape}. "
                "Re-export with end2end=True (default) to get (N, 300, 6)."
            )

        self.input_name = self.session.get_inputs()[0].name

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, image_path: str | Path) -> list[dict]:
        """
        Run inference on one image.

        Returns a list of dicts, each with keys:
            xmin, ymin, xmax, ymax  — absolute pixel coords in original image
            confidence              — float in [0, 1]
            class                   — string class name
        An empty list means no detections above conf_threshold.
        """
        img = Image.open(str(image_path)).convert("RGB")
        orig_w, orig_h = img.size

        # --- pre-process -----------------------------------------------
        img_lb, scale, (pad_x, pad_y) = self._letterbox(img, self.imgsz)
        x = np.array(img_lb, dtype=np.float32) / 255.0          # [0,1]
        x = x.transpose(2, 0, 1)[np.newaxis, ...]               # NCHW

        # --- inference -------------------------------------------------
        raw = self.session.run(None, {self.input_name: x})[0]    # (1,300,6)
        detections = raw[0]                                       # (300,6)

        # --- post-process ----------------------------------------------
        results: list[dict] = []
        for x1, y1, x2, y2, score, cls_idx in detections:
            if float(score) < self.conf:
                continue

            # Undo letterbox: input-space px → original-image px
            x1 = (float(x1) - pad_x) / scale
            y1 = (float(y1) - pad_y) / scale
            x2 = (float(x2) - pad_x) / scale
            y2 = (float(y2) - pad_y) / scale

            # Clip to image bounds
            x1 = max(0.0, min(float(orig_w), x1))
            y1 = max(0.0, min(float(orig_h), y1))
            x2 = max(0.0, min(float(orig_w), x2))
            y2 = max(0.0, min(float(orig_h), y2))

            cls_name = (
                self.class_names[int(cls_idx)]
                if int(cls_idx) < len(self.class_names)
                else str(int(cls_idx))
            )

            results.append(
                {
                    "xmin": x1,
                    "ymin": y1,
                    "xmax": x2,
                    "ymax": y2,
                    "confidence": float(score),
                    "class": cls_name,
                }
            )

        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _letterbox(
        img: Image.Image,
        target: int,
    ) -> Tuple[Image.Image, float, Tuple[float, float]]:
        """
        Resize image to (target × target) with grey padding, preserving aspect ratio.

        Returns:
            letterboxed PIL image of size (target, target)
            scale  — ratio by which original was scaled before padding
            (pad_x, pad_y) — pixels of left / top padding added
        """
        orig_w, orig_h = img.size
        scale = min(target / orig_w, target / orig_h)

        new_w = int(math.floor(orig_w * scale))
        new_h = int(math.floor(orig_h * scale))
        img_resized = img.resize((new_w, new_h), Image.BILINEAR)

        pad_x = (target - new_w) / 2.0
        pad_y = (target - new_h) / 2.0

        canvas = Image.new("RGB", (target, target), (114, 114, 114))
        canvas.paste(img_resized, (int(pad_x), int(pad_y)))

        return canvas, scale, (pad_x, pad_y)
