"""Appearance embeddings for cross-camera ReID.

For single-camera tracking, ByteTrack's motion model is usually enough.
ReID enters when:
    * objects leave and re-enter the FOV after >lost_track_buffer frames
    * you must hand off identities across non-overlapping CCTV cameras
    * crowded scenes cause many ID switches

We expose two lightweight backbones:

    * ``osnet``   — OSNet (Zhou et al. ICCV'19) via torchreid (most popular)
    * ``clip``    — image embedding from a CLIP ViT-B/16 — strong zero-shot
                    appearance signal without ReID training

Both produce a unit-norm L2 embedding (N, D) — feed those to a tracker
with cosine-distance association.
"""

from __future__ import annotations

import numpy as np


class CLIPReID:
    """Use the CLIP image encoder as an appearance feature extractor."""

    def __init__(
        self,
        clip_model: str = "ViT-B-16",
        pretrained: str = "openai",
        device: str = "cuda",
    ):
        import open_clip  # type: ignore
        import torch

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            clip_model, pretrained=pretrained, device=device
        )
        self.model.eval()

    def embed(self, crops_rgb: list[np.ndarray]) -> np.ndarray:
        """Encode a batch of person/object crops to L2-normalized embeddings."""
        import torch
        from PIL import Image

        if not crops_rgb:
            return np.zeros((0, 512), dtype=np.float32)
        pil = [Image.fromarray(c) for c in crops_rgb]
        x = torch.stack([self.preprocess(p) for p in pil]).to(self.device)
        with torch.no_grad():
            f = self.model.encode_image(x)
            f = f / (f.norm(dim=-1, keepdim=True) + 1e-9)
        return f.detach().cpu().numpy().astype(np.float32)


class OSNetReID:
    """OSNet ReID via torchreid. Requires `pip install torchreid`."""

    def __init__(self, model_name: str = "osnet_x0_25", device: str = "cuda"):
        try:
            import torchreid  # type: ignore
        except ImportError as e:
            raise ImportError("torchreid is required for OSNet ReID — `pip install torchreid`") from e
        import torch

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self.model = torchreid.models.build_model(name=model_name, num_classes=1, pretrained=True)
        self.model.eval().to(device)
        self._mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 1, 3)
        self._std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 1, 3)

    def embed(self, crops_rgb: list[np.ndarray]) -> np.ndarray:
        import cv2
        import torch

        if not crops_rgb:
            return np.zeros((0, 512), dtype=np.float32)
        resized = np.stack([cv2.resize(c, (128, 256)) for c in crops_rgb]).astype(np.float32) / 255.0
        normed = (resized - self._mean) / self._std
        x = torch.from_numpy(normed).permute(0, 3, 1, 2).float().to(self.device)
        with torch.no_grad():
            f = self.model(x)
            if isinstance(f, (list, tuple)):
                f = f[0]
            f = f / (f.norm(dim=-1, keepdim=True) + 1e-9)
        return f.detach().cpu().numpy().astype(np.float32)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise cosine distance between two sets of unit-norm vectors."""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    sim = a @ b.T
    return (1.0 - sim).astype(np.float32)


def build_reid(backend: str = "clip", **kwargs) -> object:
    backend = backend.lower()
    if backend == "clip":
        return CLIPReID(**kwargs)
    if backend == "osnet":
        return OSNetReID(**kwargs)
    raise ValueError(f"Unknown ReID backend: {backend}")
