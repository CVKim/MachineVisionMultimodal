"""Typer-based CLI: `mvmm <command>`.

Subcommands:
    anomaly train      — fit PatchCore / EfficientAD on a category
    anomaly eval       — score test split and compute AUROC/PRO
    anomaly zero-shot  — AnomalyCLIP zero-shot inference
    metrology measure  — segment + measure a single image
    pose bin-pick      — run the (mock) bin-pick pipeline on RGB+depth
    pdm train          — train the multimodal PdM model on a CSV
    info               — print environment and version
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Annotated

import typer
from rich import print as rprint

from mvmm import __version__

app = typer.Typer(help="Machine Vision MultiModal — manufacturing AI CLI.", no_args_is_help=True)

anomaly_app = typer.Typer(help="Anomaly / defect detection commands.", no_args_is_help=True)
metrology_app = typer.Typer(help="Dimensional metrology commands.", no_args_is_help=True)
pose_app = typer.Typer(help="6D pose & bin-picking commands.", no_args_is_help=True)
pdm_app = typer.Typer(help="Predictive maintenance commands.", no_args_is_help=True)

app.add_typer(anomaly_app, name="anomaly")
app.add_typer(metrology_app, name="metrology")
app.add_typer(pose_app, name="pose")
app.add_typer(pdm_app, name="pdm")


@app.command()
def info() -> None:
    """Print version & key dependency availability."""
    rprint(f"[bold cyan]mvmm[/] version [yellow]{__version__}[/]")

    def check(name: str) -> str:
        return "[green]ok[/]" if importlib.util.find_spec(name) else "[red]missing[/]"

    rprint(f"  torch:       {check('torch')}")
    rprint(f"  torchvision: {check('torchvision')}")
    rprint(f"  open_clip:   {check('open_clip')}")
    rprint(f"  open3d:      {check('open3d')}")
    rprint(f"  faiss:       {check('faiss')}")

    try:
        import torch

        rprint(f"  cuda avail:  {'[green]yes[/]' if torch.cuda.is_available() else '[yellow]cpu-only[/]'}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                rprint(f"   gpu[{i}]: {torch.cuda.get_device_name(i)}")
    except ImportError:
        pass


@anomaly_app.command("train")
def anomaly_train(
    data_root: Annotated[Path, typer.Option(help="MVTec-AD category root (must contain train/good).")],
    output: Annotated[Path, typer.Option(help="Where to save the trained memory bank.")] = Path(
        "./checkpoints/patchcore.pkl"
    ),
    image_size: Annotated[int, typer.Option()] = 256,
    crop_size: Annotated[int, typer.Option()] = 224,
    coreset_ratio: Annotated[float, typer.Option()] = 0.1,
    batch_size: Annotated[int, typer.Option()] = 16,
    device: Annotated[str, typer.Option()] = "cuda",
) -> None:
    """Train PatchCore on a single category of MVTec-AD."""
    from torch.utils.data import DataLoader

    from mvmm.anomaly.patchcore import PatchCore
    from mvmm.common.data import MVTecADDataset
    from mvmm.common.transforms import build_eval_transform

    tfm = build_eval_transform(image_size, crop_size)
    ds = MVTecADDataset(data_root, split="train", transform=tfm, load_masks=False)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=4, shuffle=False)
    rprint(f"[bold]Fitting PatchCore[/] on {len(ds)} normal samples ({data_root})")

    model = PatchCore(coreset_ratio=coreset_ratio, device=device)
    model.fit(loader)
    model.save(output)
    rprint(
        f"[green]Saved memory bank to[/] {output}  (bank size: {model._bank.shape if model._bank is not None else None})"
    )


@anomaly_app.command("eval")
def anomaly_eval(
    data_root: Annotated[Path, typer.Option()],
    checkpoint: Annotated[Path, typer.Option()],
    image_size: Annotated[int, typer.Option()] = 256,
    crop_size: Annotated[int, typer.Option()] = 224,
    batch_size: Annotated[int, typer.Option()] = 16,
    device: Annotated[str, typer.Option()] = "cuda",
) -> None:
    """Evaluate a saved PatchCore on the test split."""
    import numpy as np
    from torch.utils.data import DataLoader
    from torchvision import transforms as T

    from mvmm.anomaly.patchcore import PatchCore
    from mvmm.common.data import MVTecADDataset
    from mvmm.common.metrics import image_auroc, pixel_auroc, pro_score
    from mvmm.common.transforms import build_eval_transform

    tfm = build_eval_transform(image_size, crop_size)
    mask_tfm = T.Compose([T.Resize(image_size), T.CenterCrop(crop_size), T.PILToTensor()])
    ds = MVTecADDataset(data_root, split="test", transform=tfm, mask_transform=mask_tfm, load_masks=True)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=4, shuffle=False)

    model = PatchCore(device=device)
    model.load(checkpoint)

    all_img_scores, all_labels = [], []
    all_pix_scores, all_pix_masks = [], []
    for batch in loader:
        out = model.predict(batch["image"])
        all_img_scores.append(out.image_scores)
        all_labels.append(batch["label"].cpu().numpy())
        if "mask" in batch:
            all_pix_scores.append(out.score_maps)
            m = (batch["mask"].squeeze(1).cpu().numpy() > 0).astype(np.uint8)
            all_pix_masks.append(m)

    img_scores = np.concatenate(all_img_scores)
    labels = np.concatenate(all_labels)
    img_auc = image_auroc(img_scores, labels)
    rprint(f"[bold]Image AUROC:[/] {img_auc:.4f}")

    if all_pix_scores:
        sm = np.concatenate(all_pix_scores, axis=0)
        gm = np.concatenate(all_pix_masks, axis=0)
        rprint(f"[bold]Pixel AUROC:[/] {pixel_auroc(sm, gm):.4f}")
        rprint(f"[bold]PRO score:[/]   {pro_score(sm, gm):.4f}")


@anomaly_app.command("zero-shot")
def anomaly_zero_shot(
    image: Annotated[Path, typer.Option()],
    object_name: Annotated[
        str, typer.Option(help="Short noun phrase describing the part.")
    ] = "industrial part",
    output: Annotated[Path, typer.Option()] = Path("./outputs/clip_score.png"),
    device: Annotated[str, typer.Option()] = "cuda",
) -> None:
    """Run zero-shot anomaly scoring on a single image using CLIP prompts."""

    from mvmm.anomaly.anomaly_clip import AnomalyCLIP
    from mvmm.common.io import load_image, save_image
    from mvmm.common.transforms import build_eval_transform
    from mvmm.common.viz import overlay_heatmap

    rgb = load_image(image)
    tfm = build_eval_transform()
    x = tfm(__import__("PIL").Image.fromarray(rgb)).unsqueeze(0)

    model = AnomalyCLIP(object_name=object_name, device=device)
    out = model.predict(x)
    rprint(f"[bold]Image anomaly score:[/] {float(out.image_scores[0]):.4f}")
    heat = overlay_heatmap(rgb, out.score_maps[0])
    save_image(output, heat)
    rprint(f"[green]Saved heatmap to[/] {output}")


@metrology_app.command("measure")
def metrology_measure(
    image: Annotated[Path, typer.Option()],
    seed_x: Annotated[int, typer.Option(help="Foreground hint x-pixel.")] = -1,
    seed_y: Annotated[int, typer.Option(help="Foreground hint y-pixel.")] = -1,
    mm_per_px: Annotated[float, typer.Option(help="Scale factor; 0 = report pixel only.")] = 0.0,
    output: Annotated[Path, typer.Option()] = Path("./outputs/measure.json"),
) -> None:
    """Segment with GrabCut then report bounding box + min-circle + line dimensions."""
    from mvmm.common.io import load_image, save_json
    from mvmm.metrology.measure import circle_fit, dimension_from_mask, line_fit
    from mvmm.metrology.segmentation import ClassicalSegmenter

    rgb = load_image(image)
    h, w = rgb.shape[:2]
    if seed_x < 0 or seed_y < 0:
        seed_x, seed_y = w // 2, h // 2
    seg = ClassicalSegmenter()
    mask = seg(rgb, points=[(seed_x, seed_y, 1)])

    scale = mm_per_px if mm_per_px > 0 else None
    dim = dimension_from_mask(mask, scale_mm_per_px=scale)
    circ = circle_fit(mask, scale_mm_per_px=scale)
    line = line_fit(mask, scale_mm_per_px=scale)

    result = {
        "rect_width_px": dim.width_px,
        "rect_height_px": dim.height_px,
        "rect_width_mm": dim.width_mm,
        "rect_height_mm": dim.height_mm,
        "circle_radius_px": circ.radius_px,
        "circle_radius_mm": circ.radius_mm,
        "line_length_px": line.length_px,
        "line_length_mm": line.length_mm,
    }
    save_json(output, result)
    rprint(result)


@pose_app.command("bin-pick")
def pose_bin_pick(
    rgb: Annotated[Path, typer.Option()],
    depth: Annotated[Path, typer.Option(help="16-bit single-channel depth in mm.")],
    intrinsics_npz: Annotated[
        Path, typer.Option(help="Camera intrinsics .npz (saved via CameraIntrinsics).")
    ],
    seed_x: Annotated[int, typer.Option()] = -1,
    seed_y: Annotated[int, typer.Option()] = -1,
    output: Annotated[Path, typer.Option()] = Path("./outputs/binpick.json"),
) -> None:
    """Run a bin-pick on one frame using classical seg + antipodal grasps."""
    import cv2
    import numpy as np

    from mvmm.common.io import load_image, save_json
    from mvmm.metrology.calibration import CameraIntrinsics
    from mvmm.metrology.segmentation import ClassicalSegmenter
    from mvmm.pose.grasp import antipodal_grasps
    from mvmm.pose.pipeline import BinPickFrame, PosePipeline

    rgb_arr = load_image(rgb)
    depth_arr = cv2.imread(str(depth), cv2.IMREAD_UNCHANGED).astype(np.float32)
    intr = CameraIntrinsics.from_npz(intrinsics_npz)

    h, w = rgb_arr.shape[:2]
    if seed_x < 0 or seed_y < 0:
        seed_x, seed_y = w // 2, h // 2

    pipeline = PosePipeline(
        segmenter=ClassicalSegmenter(),
        pose_estimator=None,
        grasp_sampler=antipodal_grasps,
    )
    frame = BinPickFrame(rgb=rgb_arr, depth_mm=depth_arr, intrinsics_3x3=intr.K)
    result = pipeline(frame, seed_point=(seed_x, seed_y))

    payload = {
        "n_grasps": len(result.grasps),
        "best_grasp": (
            None
            if not result.grasps
            else {
                "p1": result.grasps[0].p1.tolist(),
                "p2": result.grasps[0].p2.tolist(),
                "width_mm": result.grasps[0].width,
                "score": result.grasps[0].score,
            }
        ),
    }
    save_json(output, payload)
    rprint(payload)


@pdm_app.command("train")
def pdm_train(
    table: Annotated[Path, typer.Option()],
    sensor_cols: Annotated[str, typer.Option(help="Comma-separated sensor column names.")],
    target_col: Annotated[str, typer.Option()],
    seq_len: Annotated[int, typer.Option()] = 512,
    epochs: Annotated[int, typer.Option()] = 5,
    batch_size: Annotated[int, typer.Option()] = 32,
    device: Annotated[str, typer.Option()] = "cuda",
    output: Annotated[Path, typer.Option()] = Path("./checkpoints/pdm.pt"),
) -> None:
    """Tiny training loop for the multimodal PdM model (smoke / starter)."""
    import torch
    from torch.utils.data import DataLoader

    from mvmm.common.transforms import build_train_transform
    from mvmm.pdm.datasets import PdMSlidingWindowDataset
    from mvmm.pdm.fusion import MultimodalPdMModel, PdMConfig

    cols = [c.strip() for c in sensor_cols.split(",") if c.strip()]
    ds = PdMSlidingWindowDataset(
        table_path=table,
        seq_len=seq_len,
        sensor_cols=cols,
        target_col=target_col,
        transform=build_train_transform(),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2)
    model = MultimodalPdMModel(PdMConfig(n_sensor_channels=len(cols), sensor_seq_len=seq_len)).to(device)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4)
    crit = torch.nn.CrossEntropyLoss()

    for ep in range(epochs):
        model.train()
        total = 0.0
        for batch in loader:
            sensor = batch["sensor"].to(device).float()
            img = batch["image"]
            if img is None or (hasattr(img, "numel") and img.numel() == 0):
                img = torch.zeros(sensor.shape[0], 3, 224, 224, device=device)
            else:
                img = img.to(device).float()
            tgt = batch["target"].long().to(device)
            logits = model(sensor, img)
            loss = crit(logits, tgt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * sensor.shape[0]
        rprint(f"[bold]epoch[/] {ep + 1}/{epochs}  loss={total / max(len(ds), 1):.4f}")

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output)
    rprint(f"[green]Saved PdM model to[/] {output}")


if __name__ == "__main__":  # pragma: no cover
    app()
