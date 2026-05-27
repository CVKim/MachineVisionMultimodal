"""Gradio demo: upload an image / video, pick a task, see the result.

Run:
    python -m mvmm.serving.gradio_app
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np


def _tracking_demo(video_path: str, classes_csv: str, model: str = "yolov8n.pt") -> str:
    from mvmm.tracking import ByteTrackTracker, TrackingPipeline, build_detector

    classes = [c.strip() for c in classes_csv.split(",") if c.strip()] or None
    det = build_detector("yolo", model=model)
    tracker = ByteTrackTracker()
    pipeline = TrackingPipeline(detector=det, tracker=tracker, classes=classes)
    out_path = Path(tempfile.gettempdir()) / "mvmm_track_demo.mp4"
    pipeline.process_video(video_path, output_video=out_path, output_json=None, show_progress=False)
    return str(out_path)


def _zeroshot_demo(image: np.ndarray, classes_csv: str, detector_name: str) -> np.ndarray:
    import cv2

    from mvmm.tracking.detectors import GroundingDINODetector
    from mvmm.zeroshot.owl_v2 import OWLv2Detector

    classes = [c.strip() for c in classes_csv.split(",") if c.strip()]
    det = GroundingDINODetector() if detector_name == "GroundingDINO" else OWLv2Detector()
    dets = det(image, classes=classes)
    overlay = image.copy()
    for b, s, lab in zip(dets.boxes, dets.scores, dets.labels, strict=False):
        x1, y1, x2, y2 = (int(v) for v in b)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 64, 64), 2)
        cv2.putText(
            overlay,
            f"{dets.class_names[int(lab)]} {float(s):.2f}",
            (x1, max(y1 - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 64, 64),
            2,
        )
    return overlay


def _depth_demo(image: np.ndarray) -> np.ndarray:
    import cv2

    from mvmm.common.viz import normalize01
    from mvmm.three_d.depth import DepthAnythingV2

    depth = DepthAnythingV2()(image)
    vis = (normalize01(depth) * 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)


def build_app():
    import gradio as gr  # type: ignore

    with gr.Blocks(title="mvmm — Industrial Vision MultiModal Demo") as app:
        gr.Markdown("# mvmm — Industrial Vision MultiModal Demo")
        with gr.Tab("CCTV Tracking"):
            vid = gr.Video(label="Input video")
            cls = gr.Textbox(label="Classes (comma-separated)", value="person,forklift,pallet")
            model = gr.Dropdown(
                ["yolov8n.pt", "yolov8s.pt", "yolo11s.pt"], value="yolov8n.pt", label="Detector"
            )
            out_vid = gr.Video(label="Tracked video")
            gr.Button("Run").click(_tracking_demo, [vid, cls, model], out_vid)

        with gr.Tab("Zero-shot Detection"):
            img = gr.Image(type="numpy", label="Input image")
            cls2 = gr.Textbox(label="Text classes (comma-separated)", value="person,forklift,box")
            det = gr.Radio(["GroundingDINO", "OWLv2"], value="GroundingDINO", label="Detector")
            out_img = gr.Image(label="Detections")
            gr.Button("Run").click(_zeroshot_demo, [img, cls2, det], out_img)

        with gr.Tab("Monocular Depth"):
            img2 = gr.Image(type="numpy", label="Input image")
            out_img2 = gr.Image(label="Depth (Depth Anything v2)")
            gr.Button("Run").click(_depth_demo, [img2], out_img2)
    return app


if __name__ == "__main__":  # pragma: no cover
    build_app().launch(server_name="0.0.0.0", server_port=7860, show_error=True)
