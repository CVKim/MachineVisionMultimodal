"""OWLv2 — Open-Vocabulary Detection via HuggingFace transformers.

Reference:
    Minderer et al. "OWLv2: Scaling Open-Vocabulary Object Detection." 2023.
    https://arxiv.org/abs/2306.09683

Compared to GroundingDINO:
    * OWLv2 ships pre-quantized variants on the HF Hub; fewer external deps.
    * Slightly weaker on dense, cluttered scenes than DINO; faster.
"""

from __future__ import annotations

import numpy as np

from mvmm.tracking.detectors import Detections


class OWLv2Detector:
    """Zero-shot detector via ``google/owlv2-base-patch16-ensemble``."""

    def __init__(
        self,
        model_id: str = "google/owlv2-base-patch16-ensemble",
        device: str = "cuda",
        score_threshold: float = 0.15,
    ):
        try:
            from transformers import Owlv2ForObjectDetection, Owlv2Processor  # type: ignore
        except ImportError as e:
            raise ImportError("transformers is required — `pip install transformers`") from e
        import torch

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self.score_threshold = score_threshold
        self.processor = Owlv2Processor.from_pretrained(model_id)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_id).to(device).eval()

    def __call__(self, image_rgb: np.ndarray, classes: list[str]) -> Detections:
        import torch
        from PIL import Image

        if not classes:
            raise ValueError("OWLv2 requires at least one text class.")
        pil = Image.fromarray(image_rgb)
        # OWLv2 wants a list[list[str]] — one inner list per image.
        inputs = self.processor(text=[classes], images=pil, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        target_sizes = torch.tensor([pil.size[::-1]], device=self.device)
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs,
            threshold=self.score_threshold,
            target_sizes=target_sizes,
        )[0]
        boxes = results["boxes"].detach().cpu().numpy().astype(np.float32)
        scores = results["scores"].detach().cpu().numpy().astype(np.float32)
        labels = results["labels"].detach().cpu().numpy().astype(np.int64)
        return Detections(boxes=boxes, scores=scores, labels=labels, class_names=list(classes))
