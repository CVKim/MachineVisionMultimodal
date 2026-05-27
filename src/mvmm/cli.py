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

app = typer.Typer(
    help="Machine Vision MultiModal — CCTV tracking, zero-shot perception, 3D, video anomaly, PdM.",
    no_args_is_help=True,
)

track_app = typer.Typer(
    help="CCTV / video object tracking (detect + track + analytics).", no_args_is_help=True
)
zeroshot_app = typer.Typer(
    help="Zero-shot / open-vocabulary perception (CLIP, GDINO, SAM2).", no_args_is_help=True
)
depth_app = typer.Typer(help="3D depth estimation + point cloud export.", no_args_is_help=True)
vad_app = typer.Typer(help="Video anomaly detection (frame-AE / MemAE).", no_args_is_help=True)
anomaly_app = typer.Typer(help="Image-level anomaly / defect detection.", no_args_is_help=True)
metrology_app = typer.Typer(help="Dimensional metrology commands.", no_args_is_help=True)
pose_app = typer.Typer(help="6D pose & bin-picking commands.", no_args_is_help=True)
pdm_app = typer.Typer(help="Predictive maintenance commands.", no_args_is_help=True)

app.add_typer(track_app, name="track")
app.add_typer(zeroshot_app, name="zeroshot")
app.add_typer(depth_app, name="depth")
app.add_typer(vad_app, name="vad")
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

    rprint(f"  torch:        {check('torch')}")
    rprint(f"  torchvision:  {check('torchvision')}")
    rprint(f"  open_clip:    {check('open_clip')}")
    rprint(f"  open3d:       {check('open3d')}")
    rprint(f"  faiss:        {check('faiss')}")
    rprint(f"  ultralytics:  {check('ultralytics')}   [for tracking]")
    rprint(f"  supervision:  {check('supervision')}   [for ByteTrack]")
    rprint(f"  transformers: {check('transformers')}  [for GDINO/OWLv2]")
    rprint(f"  sam2:         {check('sam2')}          [for SAM2]")
    rprint(f"  gsplat:       {check('gsplat')}        [for 3D Gaussian Splatting]")

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

    from mvmm.common.io import load_image, save_image
    from mvmm.common.transforms import build_eval_transform
    from mvmm.common.viz import overlay_heatmap
    from mvmm.zeroshot.anomaly_clip import AnomalyCLIP

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
    from mvmm.three_d.metrology.measure import circle_fit, dimension_from_mask, line_fit
    from mvmm.three_d.metrology.segmentation import ClassicalSegmenter

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
    from mvmm.three_d.metrology.calibration import CameraIntrinsics
    from mvmm.three_d.metrology.segmentation import ClassicalSegmenter
    from mvmm.three_d.pose.grasp import antipodal_grasps
    from mvmm.three_d.pose.pipeline import BinPickFrame, PosePipeline

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
    stride: Annotated[int, typer.Option(help="Sliding-window stride (timesteps).")] = 256,
    epochs: Annotated[int, typer.Option()] = 5,
    batch_size: Annotated[int, typer.Option()] = 32,
    num_workers: Annotated[int, typer.Option()] = 0,
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
        stride=stride,
        transform=build_train_transform(),
    )
    rprint(f"[bold]PdM dataset[/]: {len(ds)} windows, seq_len={seq_len}, stride={stride}")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
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


@pdm_app.command("eval")
def pdm_eval(
    table: Annotated[Path, typer.Option()],
    sensor_cols: Annotated[str, typer.Option(help="Comma-separated sensor column names.")],
    target_col: Annotated[str, typer.Option()],
    checkpoint: Annotated[Path, typer.Option()],
    seq_len: Annotated[int, typer.Option()] = 512,
    stride: Annotated[int, typer.Option()] = 256,
    batch_size: Annotated[int, typer.Option()] = 16,
    device: Annotated[str, typer.Option()] = "cuda",
) -> None:
    """Score a saved PdM model on a labeled table and report accuracy + AUROC."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from mvmm.common.metrics import image_auroc
    from mvmm.common.transforms import build_eval_transform
    from mvmm.pdm.datasets import PdMSlidingWindowDataset
    from mvmm.pdm.fusion import MultimodalPdMModel, PdMConfig

    cols = [c.strip() for c in sensor_cols.split(",") if c.strip()]
    ds = PdMSlidingWindowDataset(
        table_path=table,
        seq_len=seq_len,
        sensor_cols=cols,
        target_col=target_col,
        stride=stride,
        transform=build_eval_transform(),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    model = MultimodalPdMModel(PdMConfig(n_sensor_channels=len(cols), sensor_seq_len=seq_len)).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()

    all_pred, all_prob, all_tgt = [], [], []
    with torch.no_grad():
        for batch in loader:
            sensor = batch["sensor"].to(device).float()
            img = batch["image"]
            if img is None or (hasattr(img, "numel") and img.numel() == 0):
                img = torch.zeros(sensor.shape[0], 3, 224, 224, device=device)
            else:
                img = img.to(device).float()
            logits = model(sensor, img)
            prob = torch.softmax(logits, dim=-1)[:, 1]
            all_pred.append(logits.argmax(dim=-1).cpu().numpy())
            all_prob.append(prob.cpu().numpy())
            all_tgt.append(batch["target"].long().numpy())

    pred = np.concatenate(all_pred)
    prob = np.concatenate(all_prob)
    tgt = np.concatenate(all_tgt)
    acc = float((pred == tgt).mean())
    auc = image_auroc(prob, tgt)
    rprint(f"[bold]PdM eval[/]  windows={len(tgt)}  accuracy={acc:.4f}  AUROC={auc:.4f}")
    rprint(f"  predictions: {np.bincount(pred, minlength=2).tolist()} (0 vs 1)")
    rprint(f"  ground truth: {np.bincount(tgt, minlength=2).tolist()} (0 vs 1)")


# ===========================================================================
# Tracking
# ===========================================================================
@track_app.command("video")
def track_video(
    input: Annotated[Path, typer.Option(help="Input video path (mp4 etc.).")],
    output_video: Annotated[Path, typer.Option()] = Path("outputs/track.mp4"),
    output_json: Annotated[Path, typer.Option()] = Path("outputs/track.json"),
    detector: Annotated[str, typer.Option(help="yolo | grounding_dino")] = "yolo",
    model: Annotated[
        str, typer.Option(help="YOLO checkpoint name (e.g. yolov8n.pt, yolo11s.pt).")
    ] = "yolov8n.pt",
    classes: Annotated[str, typer.Option(help="Comma-separated class names to keep (empty=all).")] = "",
    score_threshold: Annotated[float, typer.Option()] = 0.25,
    frame_rate: Annotated[int, typer.Option()] = 30,
    device: Annotated[str, typer.Option()] = "cuda",
) -> None:
    """Run an end-to-end CCTV tracking pipeline on a video."""
    from mvmm.tracking import ByteTrackTracker, TrackingPipeline, build_detector

    cls_list = [c.strip() for c in classes.split(",") if c.strip()] or None
    det = (
        build_detector(detector, model=model, device=device)
        if detector == "yolo"
        else build_detector(detector, device=device)
    )
    tracker = ByteTrackTracker(frame_rate=frame_rate, track_activation_threshold=score_threshold)
    pipeline = TrackingPipeline(
        detector=det, tracker=tracker, classes=cls_list, score_threshold=score_threshold
    )
    stats = pipeline.process_video(input, output_video=output_video, output_json=output_json)
    rprint(f"[bold]Tracking done[/]: {stats.as_dict()}")
    rprint(f"  annotated video: {output_video}")
    rprint(f"  per-frame JSON:  {output_json}")


# ===========================================================================
# Zero-shot
# ===========================================================================
@zeroshot_app.command("detect")
def zeroshot_detect(
    image: Annotated[Path, typer.Option()],
    classes: Annotated[str, typer.Option(help="Comma-separated text classes.")],
    detector: Annotated[str, typer.Option(help="grounding_dino | owlv2")] = "grounding_dino",
    output: Annotated[Path, typer.Option()] = Path("outputs/zeroshot_detect.png"),
    device: Annotated[str, typer.Option()] = "cuda",
    score_threshold: Annotated[float, typer.Option()] = 0.25,
) -> None:
    """Open-vocabulary detection on a single image: text prompt → boxes overlay."""
    import cv2

    from mvmm.common.io import load_image, save_image
    from mvmm.tracking.detectors import GroundingDINODetector
    from mvmm.zeroshot.owl_v2 import OWLv2Detector

    cls_list = [c.strip() for c in classes.split(",") if c.strip()]
    if not cls_list:
        raise typer.BadParameter("--classes must contain at least one class")
    rgb = load_image(image)
    if detector == "grounding_dino":
        det = GroundingDINODetector(device=device, box_threshold=score_threshold)
    else:
        det = OWLv2Detector(device=device, score_threshold=score_threshold)
    dets = det(rgb, classes=cls_list)

    overlay = rgb.copy()
    for b, s, lab in zip(dets.boxes, dets.scores, dets.labels, strict=False):
        x1, y1, x2, y2 = (int(v) for v in b)
        name = dets.class_names[int(lab)] if 0 <= int(lab) < len(dets.class_names) else "?"
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 64, 64), 2)
        cv2.putText(
            overlay,
            f"{name} {s:.2f}",
            (x1, max(y1 - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 64, 64),
            2,
        )
    save_image(output, overlay)
    rprint(f"[bold]Detected {len(dets.boxes)} objects[/] -> overlay saved to {output}")
    for b, s, lab in zip(dets.boxes, dets.scores, dets.labels, strict=False):
        name = dets.class_names[int(lab)] if 0 <= int(lab) < len(dets.class_names) else "?"
        rprint(f"  {name:<24s} score={float(s):.3f}  bbox={[round(float(v), 1) for v in b]}")


# ===========================================================================
# Depth
# ===========================================================================
@depth_app.command("infer")
def depth_infer(
    image: Annotated[Path, typer.Option()],
    output: Annotated[Path, typer.Option()] = Path("outputs/depth.png"),
    output_ply: Annotated[Path, typer.Option()] = Path(""),
    model_id: Annotated[str, typer.Option()] = "depth-anything/Depth-Anything-V2-Small-hf",
    device: Annotated[str, typer.Option()] = "cuda",
) -> None:
    """Monocular depth (Depth Anything v2). Optionally export a PLY point cloud."""
    import cv2
    import numpy as np

    from mvmm.common.io import load_image
    from mvmm.common.viz import normalize01
    from mvmm.three_d.depth import DepthAnythingV2
    from mvmm.three_d.reconstruction import back_project, save_ply

    rgb = load_image(image)
    estimator = DepthAnythingV2(model_id=model_id, device=device)
    depth = estimator(rgb)

    # Save false-color depth visualization
    vis = (normalize01(depth) * 255).astype(np.uint8)
    vis_color = cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), vis_color)
    rprint(f"[bold]Depth saved[/] {output}  (min={float(depth.min()):.3f}, max={float(depth.max()):.3f})")

    if str(output_ply):
        h, w = depth.shape
        K = np.array([[w, 0, w / 2.0], [0, w, h / 2.0], [0, 0, 1.0]])  # rough default intrinsics
        result = back_project(depth, K, rgb=rgb)
        save_ply(str(output_ply), result["points"], result["colors"])
        rprint(f"  PLY saved: {output_ply}  ({len(result['points'])} points)")


# ===========================================================================
# Video anomaly detection
# ===========================================================================
@vad_app.command("train")
def vad_train(
    train_root: Annotated[Path, typer.Option()],
    epochs: Annotated[int, typer.Option()] = 10,
    batch_size: Annotated[int, typer.Option()] = 32,
    lr: Annotated[float, typer.Option()] = 1e-3,
    image_size: Annotated[int, typer.Option()] = 128,
    output: Annotated[Path, typer.Option()] = Path("checkpoints/vad_ae.pt"),
    device: Annotated[str, typer.Option()] = "cuda",
) -> None:
    """Train the frame-AE VAD baseline on a folder of normal frames."""
    import torch
    from torch.utils.data import DataLoader
    from torchvision import transforms as T

    from mvmm.vad.conv_autoencoder import ConvAutoEncoder
    from mvmm.vad.datasets import VideoFrameDataset

    dev = device if torch.cuda.is_available() else "cpu"
    tfm = T.Compose([T.Resize(image_size), T.CenterCrop(image_size), T.ToTensor()])
    ds = VideoFrameDataset(train_root, transform=tfm, normals_only=True)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    model = ConvAutoEncoder().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    crit = torch.nn.MSELoss()

    rprint(f"[bold]VAD train[/]: samples={len(ds)} device={dev}")
    for ep in range(epochs):
        total = 0.0
        for batch in loader:
            x = batch["image"].to(dev).float()
            loss = crit(model(x), x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * x.shape[0]
        rprint(f"  epoch {ep + 1}/{epochs}  recon_loss={total / max(len(ds), 1):.5f}")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output)
    rprint(f"[green]Saved VAD model to[/] {output}")


@vad_app.command("eval")
def vad_eval(
    test_root: Annotated[Path, typer.Option()],
    labels: Annotated[Path, typer.Option()],
    checkpoint: Annotated[Path, typer.Option()],
    image_size: Annotated[int, typer.Option()] = 128,
    batch_size: Annotated[int, typer.Option()] = 64,
    device: Annotated[str, typer.Option()] = "cuda",
) -> None:
    """Score a folder of test frames; print AUROC + per-clip means."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from torchvision import transforms as T

    from mvmm.common.metrics import image_auroc
    from mvmm.vad.conv_autoencoder import ConvAutoEncoder
    from mvmm.vad.datasets import VideoFrameDataset

    dev = device if torch.cuda.is_available() else "cpu"
    tfm = T.Compose([T.Resize(image_size), T.CenterCrop(image_size), T.ToTensor()])
    ds = VideoFrameDataset(test_root, transform=tfm, labels_csv=labels, normals_only=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    model = ConvAutoEncoder().to(dev)
    model.load_state_dict(torch.load(checkpoint, map_location=dev, weights_only=True))
    model.eval()

    scores, lbls, vids = [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(dev).float()
            s = model.anomaly_score(x).cpu().numpy()
            scores.append(s)
            lbls.append(np.asarray(batch["label"]))
            vids.extend(batch["video"])
    scores = np.concatenate(scores)
    lbls = np.concatenate(lbls)
    rprint(f"[bold]VAD eval[/]  frames={len(lbls)}  AUROC={image_auroc(scores, lbls):.4f}")
    by_video: dict[str, list[float]] = {}
    for v, s in zip(vids, scores, strict=False):
        by_video.setdefault(v, []).append(float(s))
    for v, ss in sorted(by_video.items()):
        rprint(f"  {v:<28s}  mean={np.mean(ss):.5f}  max={np.max(ss):.5f}")


if __name__ == "__main__":  # pragma: no cover
    app()
