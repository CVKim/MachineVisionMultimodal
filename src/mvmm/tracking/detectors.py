"""Pluggable object detectors for the tracking pipeline.

Backends:
    * ``yolo``           — Ultralytics YOLOv8/v11 (closed-vocabulary, fast).
    * ``yolo_world``     — YOLO-World open-vocabulary (text-conditioned YOLO).
    * ``rtdetr``         — Ultralytics RT-DETR (DETR family, real-time).
    * ``grounding_dino`` — Open-vocabulary detection via HuggingFace transformers.

All detectors expose the same API::

    det = build_detector("yolo", model="yolov8n.pt")
    out = det(image_rgb, classes=["person", "forklift"])
    # out: dict with keys "boxes" (N,4 xyxy), "scores" (N,), "labels" (N,), "class_names"

Heavy backends are lazy-imported so the package stays light.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Detections:
    """Container compatible with `supervision.Detections.from_*`."""

    boxes: np.ndarray  # (N, 4) xyxy in pixels
    scores: np.ndarray  # (N,)
    labels: np.ndarray  # (N,) int class indices
    class_names: list[str]  # length C (master class list)

    def filter_by_class(self, allowed: list[str]) -> Detections:
        if not allowed:
            return self
        allowed_idx = {self.class_names.index(c) for c in allowed if c in self.class_names}
        keep = np.array([int(lab) in allowed_idx for lab in self.labels], dtype=bool)
        return Detections(self.boxes[keep], self.scores[keep], self.labels[keep], self.class_names)

    def filter_by_score(self, threshold: float) -> Detections:
        keep = self.scores >= threshold
        return Detections(self.boxes[keep], self.scores[keep], self.labels[keep], self.class_names)


# ---------------------------------------------------------------------------
# YOLOv8 / YOLOv11 (Ultralytics)
# ---------------------------------------------------------------------------
class UltralyticsYOLO:
    """Thin wrapper around an Ultralytics YOLO checkpoint."""

    def __init__(self, model: str = "yolov8n.pt", device: str = "cuda", conf: float = 0.25):
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as e:
            raise ImportError(
                "ultralytics is required for the YOLO backend — `pip install ultralytics`"
            ) from e
        self.model = YOLO(model)
        self.device = device
        self.conf = conf
        self.class_names = (
            list(self.model.names.values()) if isinstance(self.model.names, dict) else list(self.model.names)
        )

    def __call__(self, image_rgb: np.ndarray, classes: list[str] | None = None) -> Detections:
        # Ultralytics accepts class-index filter; we'll filter by name later.
        result = self.model.predict(image_rgb, conf=self.conf, device=self.device, verbose=False)[0]
        if result.boxes is None or len(result.boxes) == 0:
            empty = np.zeros((0, 4), dtype=np.float32)
            return Detections(
                empty, np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int64), self.class_names
            )
        boxes = result.boxes.xyxy.cpu().numpy().astype(np.float32)
        scores = result.boxes.conf.cpu().numpy().astype(np.float32)
        labels = result.boxes.cls.cpu().numpy().astype(np.int64)
        dets = Detections(boxes, scores, labels, self.class_names)
        if classes:
            dets = dets.filter_by_class(classes)
        return dets


# ---------------------------------------------------------------------------
# GroundingDINO (open-vocabulary via HuggingFace transformers)
# ---------------------------------------------------------------------------
class GroundingDINODetector:
    """Open-vocabulary detection: feed a text prompt, get boxes back.

    Uses ``IDEA-Research/grounding-dino-tiny`` (or any GroundingDINO checkpoint
    on the HF Hub). The text prompt is a period-separated list of phrases,
    e.g. ``"person. forklift. pallet."`` — each phrase becomes a "class".
    """

    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-tiny",
        device: str = "cuda",
        box_threshold: float = 0.30,
        text_threshold: float = 0.25,
    ):
        try:
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor  # type: ignore
        except ImportError as e:
            raise ImportError(
                "transformers is required for GroundingDINO — `pip install transformers`"
            ) from e
        import torch

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device).eval()

    @staticmethod
    def _build_prompt(classes: list[str]) -> str:
        # Convention: lower-case + trailing period.
        return ". ".join(c.lower().strip(".") for c in classes) + "."

    def __call__(self, image_rgb: np.ndarray, classes: list[str] | None = None) -> Detections:
        import torch
        from PIL import Image as _Image

        if not classes:
            raise ValueError("GroundingDINO requires at least one text class.")
        prompt = self._build_prompt(classes)
        pil = _Image.fromarray(image_rgb)
        inputs = self.processor(images=pil, text=prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # The kwarg name for the box-confidence threshold changed across
        # transformers versions: older releases use ``box_threshold``,
        # newer ones use ``threshold``. Try the new spelling first.
        post = self.processor.post_process_grounded_object_detection
        try:
            results = post(
                outputs,
                inputs.input_ids,
                threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=[pil.size[::-1]],
            )[0]
        except TypeError:
            results = post(
                outputs,
                inputs.input_ids,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=[pil.size[::-1]],
            )[0]

        # Each detected text-phrase is matched back to the closest class.
        boxes = results["boxes"].detach().cpu().numpy().astype(np.float32)
        scores = results["scores"].detach().cpu().numpy().astype(np.float32)
        phrases = [str(p) for p in results.get("labels", [])]
        labels = np.zeros(len(boxes), dtype=np.int64)
        # ``phrases`` and ``boxes`` should align, but some transformers versions
        # emit a phrases list with a trailing empty token. Only iterate up to
        # the boxes length to avoid index-out-of-bounds on size-0 outputs.
        for i in range(min(len(phrases), len(boxes))):
            ph = phrases[i]
            matched = next((j for j, c in enumerate(classes) if c.lower() in ph.lower()), 0)
            labels[i] = matched
        return Detections(boxes=boxes, scores=scores, labels=labels, class_names=list(classes))


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------
def build_detector(backend: str = "yolo", **kwargs: Any) -> Any:
    """Construct a detector by name.

    Examples:
        build_detector("yolo", model="yolov8n.pt")
        build_detector("yolo", model="yolo11s.pt", conf=0.4)
        build_detector("grounding_dino", model_id="IDEA-Research/grounding-dino-tiny")
    """
    backend = backend.lower()
    if backend in {"yolo", "yolov8", "yolov11", "yolo11", "rtdetr"}:
        return UltralyticsYOLO(**kwargs)
    if backend in {"grounding_dino", "gdino", "groundingdino"}:
        return GroundingDINODetector(**kwargs)
    raise ValueError(f"Unknown detector backend: {backend}")
