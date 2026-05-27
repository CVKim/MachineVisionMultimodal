"""Additional monocular depth backends.

Reference (2024–2025):
    * Marigold (Ke et al. CVPR'24): diffusion-prior depth — sharp edges
    * MoGe (Wang et al. CVPR'25): geometric foundation model — metric scale
    * UniDepth: metric depth without intrinsics

All wrappers expose the same ``__call__(image_rgb) -> HxW float depth (m)``
API as ``mvmm.three_d.depth.depth.DepthAnythingV2``.
"""

from __future__ import annotations

import cv2
import numpy as np


class Marigold:
    """Marigold diffusion-based monocular depth. ``pip install diffusers``.

    Slower than Depth Anything but produces sharper depth edges, useful for
    metrology on parts with thin features.
    """

    def __init__(
        self,
        model_id: str = "prs-eth/marigold-depth-v1-0",
        device: str = "cuda",
        denoising_steps: int = 10,
    ):
        try:
            from diffusers import DiffusionPipeline  # type: ignore
        except ImportError as e:
            raise ImportError("diffusers is required for Marigold — `pip install diffusers`") from e
        import torch

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self.denoising_steps = denoising_steps
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.pipe = DiffusionPipeline.from_pretrained(
            model_id, torch_dtype=dtype, custom_pipeline=model_id
        ).to(device)

    def __call__(self, image_rgb: np.ndarray) -> np.ndarray:
        from PIL import Image

        pil = Image.fromarray(image_rgb)
        out = self.pipe(pil, denoising_steps=self.denoising_steps, ensemble_size=1)
        depth = (
            np.asarray(out.depth_np, dtype=np.float32)
            if hasattr(out, "depth_np")
            else np.asarray(out["depth_np"], dtype=np.float32)
        )
        if depth.shape[:2] != image_rgb.shape[:2]:
            depth = cv2.resize(
                depth, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_LINEAR
            )
        return depth


class MoGe:
    """MoGe — Monocular Geometric foundation model (Wang et al. CVPR 2025).

    Produces metric-scale depth + point cloud + camera intrinsics from a
    single image. Wrapper requires ``pip install moge`` (Microsoft).
    """

    def __init__(self, model_id: str = "Ruicheng/moge-vitl", device: str = "cuda"):
        try:
            from moge.model import MoGeModel  # type: ignore
        except ImportError as e:
            raise ImportError("moge is required — install per https://github.com/microsoft/MoGe") from e
        import torch

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self.model = MoGeModel.from_pretrained(model_id).to(device).eval()

    def __call__(self, image_rgb: np.ndarray) -> dict:
        import torch

        x = torch.from_numpy(image_rgb).permute(2, 0, 1).float().to(self.device) / 255.0
        with torch.no_grad():
            out = self.model.infer(x)
        return {
            "depth": out["depth"].detach().cpu().numpy().astype(np.float32),
            "points": out["points"].detach().cpu().numpy().astype(np.float32),
            "intrinsics": out.get("intrinsics", torch.eye(3)).detach().cpu().numpy().astype(np.float32),
        }
