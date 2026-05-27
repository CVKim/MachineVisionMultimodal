"""3D Gaussian Splatting wrapper.

Reference:
    Kerbl et al. "3D Gaussian Splatting for Real-Time Radiance Field
    Rendering." SIGGRAPH 2023. https://github.com/graphdeco-inria/gaussian-splatting

Why 3DGS in a CCTV / industrial context:
    * Photometric scene reconstruction from a multi-view CCTV array.
    * Digital twin generation — bake a factory floor into a renderable
      asset, then overlay tracked objects on top in real time.
    * Compared to NeRF: 10–100x faster training, real-time rendering.

The actual 3DGS training requires CUDA-compiled rasterizers (e.g.
``diff-gaussian-rasterization``). We provide a *wrapper* that defers
to an installed implementation rather than re-implementing the kernel.

Supported backends (auto-detected):
    * ``gsplat`` (https://github.com/nerfstudio-project/gsplat) — recommended,
      pip-installable.
    * ``inria-3dgs`` (legacy) — original SIGGRAPH code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class GSReconstructionResult:
    ply_path: Path
    n_gaussians: int
    psnr: float | None


class GaussianSplattingTrainer:
    """Train a 3DGS model from a directory of images + COLMAP poses.

    Args:
        backend: which implementation to use; "gsplat" is recommended.
        device: "cuda" (required for training).

    Notes:
        The wrapper expects a typical COLMAP layout:
            scene/
                images/000.png ...
                sparse/0/  (cameras.bin, images.bin, points3D.bin)
    """

    def __init__(self, backend: str = "gsplat", device: str = "cuda"):
        self.backend = backend.lower()
        self.device = device
        self._train_fn: Any = None
        if self.backend == "gsplat":
            try:
                import gsplat  # type: ignore # noqa: F401
            except ImportError as e:
                raise ImportError(
                    "gsplat is not installed. Install with: pip install gsplat (CUDA toolkit required)"
                ) from e
            # Lazy: we import the actual training entrypoint inside `train`.
        elif self.backend == "inria-3dgs":
            pass  # legacy — user is on their own for the install
        else:
            raise ValueError(f"Unknown 3DGS backend: {backend}")

    def train(
        self,
        scene_dir: str | Path,
        output_dir: str | Path,
        iterations: int = 7000,
    ) -> GSReconstructionResult:
        """Train a 3DGS model. Returns the path of the exported PLY."""
        scene_dir = Path(scene_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.backend == "gsplat":
            # gsplat ships training scripts in its examples/ directory.
            # We give a stable entry point by calling that script via subprocess
            # rather than depending on private internals.
            import subprocess
            import sys

            cmd = [
                sys.executable,
                "-m",
                "gsplat.examples.simple_trainer",
                "--scene-dir",
                str(scene_dir),
                "--result-dir",
                str(output_dir),
                "--max-steps",
                str(iterations),
            ]
            subprocess.check_call(cmd)
            ply = output_dir / "point_cloud" / f"iteration_{iterations}" / "point_cloud.ply"
            return GSReconstructionResult(ply_path=ply, n_gaussians=-1, psnr=None)

        raise NotImplementedError(f"Backend {self.backend} not wired yet — see ROADMAP.md")
