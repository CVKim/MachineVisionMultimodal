"""Pose-based activity recognition layered on top of ByteTrack.

Pipeline:
    frame ─▶ YOLOv8-pose ─▶ boxes + 17 COCO keypoints per person
              + built-in ByteTrack (persistent track ids)
              ─▶ ActivityClassifier (rule-based on keypoint motion)
              ─▶ per-track state ∈ { idle, walking, working, lifting, unknown }
              ─▶ rolling smoothing (majority vote over last K frames)
              ─▶ time-in-state accumulator (per track, per state)

The classifier is intentionally rule-based + interpretable rather than
a deep action-recognition model, so:
    * it works on a single RTX 3080 in real time,
    * each decision is traceable (which keypoint motion fired what rule),
    * the thresholds are bbox-height-normalized so far/near doesn't matter.

If you want a SOTA video-clip classifier (VideoMAE, SlowFast, X3D, MViT)
later, plug it into `ActivityClassifier._classify_dl` — the rest of the
pipeline (history, smoothing, time accounting, rendering) stays the
same.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# COCO 17-keypoint indices.
KP_NOSE = 0
KP_LSHOULDER, KP_RSHOULDER = 5, 6
KP_LELBOW, KP_RELBOW = 7, 8
KP_LWRIST, KP_RWRIST = 9, 10
KP_LHIP, KP_RHIP = 11, 12
KP_LKNEE, KP_RKNEE = 13, 14
KP_LANKLE, KP_RANKLE = 15, 16

# 17-keypoint skeleton edges for rendering.
SKELETON: tuple[tuple[int, int], ...] = (
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),  # arms
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),  # legs
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),  # torso
    (0, 5),
    (0, 6),  # nose-to-shoulders
)

# Color codes per state — used by the renderer.
STATE_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "idle": (160, 160, 160),
    "walking": (255, 120, 60),
    "working": (60, 200, 60),
    "lifting": (0, 140, 255),
    "unknown": (180, 0, 180),
}
STATE_ORDER = ("idle", "walking", "working", "lifting", "unknown")


@dataclass
class PoseTrack:
    """One person in one frame with a stable id + pose."""

    track_id: int
    bbox: np.ndarray  # (4,) xyxy
    keypoints: np.ndarray  # (17, 2) image-space xy
    kp_conf: np.ndarray  # (17,) per-keypoint confidence
    score: float = 0.0
    state: str = "unknown"


@dataclass
class _TrackHistory:
    """Per-track ring buffer + state log for one tracked person."""

    track_id: int
    window: deque[tuple[float, np.ndarray, np.ndarray, np.ndarray]] = field(
        default_factory=lambda: deque(maxlen=15)
    )  # (timestamp, kps, kp_conf, bbox)
    recent_states: deque[str] = field(default_factory=lambda: deque(maxlen=8))
    state_log: list[tuple[float, str]] = field(default_factory=list)


# ---------------------------------------------------------------------------
class PoseTracker:
    """Wrap Ultralytics YOLOv8-pose + its built-in ByteTrack.

    Args:
        model:   Ultralytics pose checkpoint (e.g. yolov8n-pose.pt).
        device:  "cuda" / "cpu".
        conf:    minimum confidence for keeping a detection.
        tracker: bytetrack.yaml | botsort.yaml (built-in configs).
        persist: keep tracker state across frames (set False on each new clip).
    """

    def __init__(
        self,
        model: str = "yolov8n-pose.pt",
        device: str = "cuda",
        conf: float = 0.25,
        tracker: str = "bytetrack.yaml",
        persist: bool = True,
    ):
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as e:
            raise ImportError("ultralytics is required for PoseTracker — `pip install ultralytics`") from e
        self.model = YOLO(model)
        self.device = device
        self.conf = conf
        self.tracker_cfg = tracker
        self.persist = persist

    def update(self, frame_rgb: np.ndarray) -> list[PoseTrack]:
        result = self.model.track(
            frame_rgb,
            persist=self.persist,
            tracker=self.tracker_cfg,
            conf=self.conf,
            device=self.device,
            verbose=False,
        )[0]

        out: list[PoseTrack] = []
        if result.boxes is None or len(result.boxes) == 0 or result.keypoints is None:
            return out
        boxes = result.boxes.xyxy.cpu().numpy().astype(np.float32)
        scores = result.boxes.conf.cpu().numpy().astype(np.float32)
        ids = (
            result.boxes.id.cpu().numpy().astype(np.int64)
            if result.boxes.id is not None
            else np.full(len(boxes), -1, dtype=np.int64)
        )
        kps_xy = result.keypoints.xy.cpu().numpy().astype(np.float32)  # (N, 17, 2)
        kps_conf = (
            result.keypoints.conf.cpu().numpy().astype(np.float32)
            if result.keypoints.conf is not None
            else np.ones((len(boxes), 17), dtype=np.float32)
        )

        for i in range(len(boxes)):
            out.append(
                PoseTrack(
                    track_id=int(ids[i]),
                    bbox=boxes[i],
                    keypoints=kps_xy[i],
                    kp_conf=kps_conf[i],
                    score=float(scores[i]),
                )
            )
        return out


# ---------------------------------------------------------------------------
class ActivityClassifier:
    """Rule-based activity classifier with bbox-height-normalized thresholds.

    Args:
        fps:             source video FPS, used to convert frame counts to time.
        window_frames:   how many past frames to look at for motion features.
        smooth_frames:   majority-vote window for the *displayed* state.
        walk_disp_ratio: ``hip_displacement / bbox_height`` above which we call walking.
        wrist_speed_ratio: per-frame wrist motion threshold (working signal).
        idle_total_ratio: total keypoint motion below which we call idle.
        min_kp_conf:     keypoints below this confidence are ignored for motion.
    """

    STATES: tuple[str, ...] = STATE_ORDER

    def __init__(
        self,
        fps: float = 25.0,
        window_frames: int = 15,
        smooth_frames: int = 8,
        walk_disp_ratio: float = 0.35,
        wrist_speed_ratio: float = 0.04,
        idle_total_ratio: float = 0.012,
        min_kp_conf: float = 0.35,
    ):
        self.fps = float(fps)
        self.window_frames = int(window_frames)
        self.smooth_frames = int(smooth_frames)
        self.walk_disp_ratio = float(walk_disp_ratio)
        self.wrist_speed_ratio = float(wrist_speed_ratio)
        self.idle_total_ratio = float(idle_total_ratio)
        self.min_kp_conf = float(min_kp_conf)
        self.histories: dict[int, _TrackHistory] = {}

    # -------------------------- core update -------------------------
    def update(self, tracks: list[PoseTrack], timestamp: float) -> None:
        seen: set[int] = set()
        for t in tracks:
            if t.track_id < 0:
                t.state = "unknown"
                continue
            h = self.histories.setdefault(t.track_id, _TrackHistory(track_id=t.track_id))
            h.window.append((timestamp, t.keypoints.copy(), t.kp_conf.copy(), t.bbox.copy()))
            raw = self._classify(h, t.bbox)
            h.recent_states.append(raw)
            t.state = Counter(list(h.recent_states)[-self.smooth_frames :]).most_common(1)[0][0]
            h.state_log.append((timestamp, t.state))
            seen.add(t.track_id)
        # tracks that have disappeared this frame keep their last state but
        # do not extend the time-in-state — handled by the time_in_state aggregator.

    # ------------------------ classification ------------------------
    def _classify(self, history: _TrackHistory, bbox: np.ndarray) -> str:
        if len(history.window) < 3:
            return "unknown"
        bbox_h = float(max(bbox[3] - bbox[1], 1.0))

        # Stack window: (T, 17, 2) and (T, 17)
        kps = np.stack([w[1] for w in history.window])
        confs = np.stack([w[2] for w in history.window])

        # Hip-center trajectory (mean of L/R hips when both visible).
        valid_lhip = confs[:, KP_LHIP] >= self.min_kp_conf
        valid_rhip = confs[:, KP_RHIP] >= self.min_kp_conf
        hip = np.zeros((kps.shape[0], 2), dtype=np.float32)
        for i in range(kps.shape[0]):
            pts, w = [], []
            if valid_lhip[i]:
                pts.append(kps[i, KP_LHIP])
                w.append(1.0)
            if valid_rhip[i]:
                pts.append(kps[i, KP_RHIP])
                w.append(1.0)
            if pts:
                hip[i] = np.average(np.stack(pts), axis=0, weights=w)
            else:
                hip[i] = (bbox[0] + bbox[2]) / 2, (bbox[1] + 3 * bbox[3]) / 4  # fallback

        # Hip displacement across the full window.
        hip_disp = float(np.linalg.norm(hip[-1] - hip[0]))

        # Wrist speeds (frame-to-frame), averaged across both wrists.
        wrist_motion: list[float] = []
        for idx in (KP_LWRIST, KP_RWRIST):
            for i in range(1, kps.shape[0]):
                if confs[i, idx] < self.min_kp_conf or confs[i - 1, idx] < self.min_kp_conf:
                    continue
                wrist_motion.append(float(np.linalg.norm(kps[i, idx] - kps[i - 1, idx])))
        wrist_speed = float(np.mean(wrist_motion)) if wrist_motion else 0.0

        # Total keypoint motion (averaged over all reliable points).
        total_motion: list[float] = []
        for i in range(1, kps.shape[0]):
            mask = (confs[i] >= self.min_kp_conf) & (confs[i - 1] >= self.min_kp_conf)
            if mask.any():
                d = np.linalg.norm(kps[i, mask] - kps[i - 1, mask], axis=1)
                total_motion.append(float(d.mean()))
        total_speed = float(np.mean(total_motion)) if total_motion else 0.0

        # Lifting heuristic: at least one wrist below the hip y-coordinate (image y grows downward → wrist_y > hip_y)
        # AND elbow is below shoulder AND the wrist has motion.
        last = kps[-1]
        last_conf = confs[-1]
        lift_signal = False
        hip_y = float(hip[-1, 1])
        for w_idx, e_idx, s_idx in (
            (KP_LWRIST, KP_LELBOW, KP_LSHOULDER),
            (KP_RWRIST, KP_RELBOW, KP_RSHOULDER),
        ):
            if (
                last_conf[w_idx] >= self.min_kp_conf
                and last_conf[e_idx] >= self.min_kp_conf
                and last_conf[s_idx] >= self.min_kp_conf
            ):
                wrist_y = float(last[w_idx, 1])
                elbow_y = float(last[e_idx, 1])
                shoulder_y = float(last[s_idx, 1])
                if wrist_y > hip_y and elbow_y > shoulder_y:
                    lift_signal = True
                    break

        # Normalize motion features by bbox height.
        walk_ratio = hip_disp / bbox_h
        wrist_ratio = wrist_speed / bbox_h
        total_ratio = total_speed / bbox_h

        # Rule cascade (priority order).
        if walk_ratio > self.walk_disp_ratio:
            return "walking"
        if lift_signal and wrist_ratio > 0.5 * self.wrist_speed_ratio:
            return "lifting"
        if wrist_ratio > self.wrist_speed_ratio and walk_ratio < 0.15:
            return "working"
        if total_ratio < self.idle_total_ratio:
            return "idle"
        return "unknown"

    # ----------------------- time accounting ------------------------
    def time_in_state(self) -> dict[int, dict[str, float]]:
        """Convert each track's state log into total seconds per state.

        For consecutive (timestamp, state) entries within the same state,
        we add the inter-frame delta. A state change ends the previous
        interval and starts a new one at the new timestamp.
        """
        out: dict[int, dict[str, float]] = {}
        for tid, h in self.histories.items():
            per_state: dict[str, float] = dict.fromkeys(self.STATES, 0.0)
            for i in range(1, len(h.state_log)):
                t_prev, s_prev = h.state_log[i - 1]
                t_cur, _s_cur = h.state_log[i]
                dt = max(t_cur - t_prev, 0.0)
                per_state[s_prev] = per_state.get(s_prev, 0.0) + dt
            out[tid] = per_state
        return out

    def total_time_in_state(self) -> dict[str, float]:
        """Aggregate seconds-per-state summed across all tracks."""
        totals: dict[str, float] = dict.fromkeys(self.STATES, 0.0)
        for per_track in self.time_in_state().values():
            for s, secs in per_track.items():
                totals[s] = totals.get(s, 0.0) + secs
        return totals

    # ------------------------- export -------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Machine-readable summary — state log per track + time totals."""
        return {
            "fps": self.fps,
            "tracks": {
                str(tid): {
                    "state_log": [(round(t, 4), s) for (t, s) in h.state_log],
                    "time_in_state_s": self.time_in_state().get(tid, {}),
                }
                for tid, h in self.histories.items()
            },
            "total_time_in_state_s": self.total_time_in_state(),
        }


# ---------------------------------------------------------------------------
def draw_pose_track(
    frame_bgr: np.ndarray,
    track: PoseTrack,
    kp_conf_thresh: float = 0.35,
) -> np.ndarray:
    """Render skeleton + bbox + state badge onto a BGR frame in place."""
    import cv2

    color = STATE_COLORS_BGR.get(track.state, STATE_COLORS_BGR["unknown"])

    # bbox
    x1, y1, x2, y2 = (int(v) for v in track.bbox)
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)

    # skeleton
    for a, b in SKELETON:
        if track.kp_conf[a] < kp_conf_thresh or track.kp_conf[b] < kp_conf_thresh:
            continue
        pa = tuple(int(v) for v in track.keypoints[a])
        pb = tuple(int(v) for v in track.keypoints[b])
        cv2.line(frame_bgr, pa, pb, color, 2, lineType=cv2.LINE_AA)
    for i in range(17):
        if track.kp_conf[i] < kp_conf_thresh:
            continue
        p = tuple(int(v) for v in track.keypoints[i])
        cv2.circle(frame_bgr, p, 3, color, -1)

    # state badge
    label = f"#{track.track_id} {track.state}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame_bgr, (x1, max(y1 - th - 8, 0)), (x1 + tw + 6, y1), color, -1)
    cv2.putText(
        frame_bgr,
        label,
        (x1 + 3, max(y1 - 5, th + 2)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        2,
        lineType=cv2.LINE_AA,
    )
    return frame_bgr


def draw_state_strip(
    frame_bgr: np.ndarray,
    classifier: ActivityClassifier,
    height: int = 56,
    seconds_window: float = 30.0,
    timestamp: float = 0.0,
) -> np.ndarray:
    """Draw a bottom-of-frame gantt strip of recent per-track states.

    Args:
        frame_bgr: BGR frame to overlay onto (returns a new image).
        classifier: source of state logs.
        height: total strip height in pixels (split across tracks).
        seconds_window: how many seconds of history to show.
    """
    import cv2

    _h, w = frame_bgr.shape[:2]
    strip = np.full((height, w, 3), 25, dtype=np.uint8)

    tracks = sorted(classifier.histories.keys())
    if not tracks:
        return np.concatenate([frame_bgr, strip], axis=0)
    row_h = max(height // max(len(tracks), 1), 8)
    t_min = max(timestamp - seconds_window, 0.0)
    t_max = max(timestamp, t_min + 1e-3)

    for i, tid in enumerate(tracks[: height // row_h]):
        h_log = classifier.histories[tid].state_log
        y0 = i * row_h
        y1 = y0 + row_h - 1
        # label
        cv2.putText(
            strip,
            f"#{tid}",
            (4, y0 + row_h - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        for j in range(1, len(h_log)):
            t0, s0 = h_log[j - 1]
            t1, _ = h_log[j]
            if t1 < t_min:
                continue
            x0 = int(40 + (max(t0, t_min) - t_min) / (t_max - t_min) * (w - 50))
            x1px = int(40 + (min(t1, t_max) - t_min) / (t_max - t_min) * (w - 50))
            if x1px <= x0:
                continue
            color = STATE_COLORS_BGR.get(s0, STATE_COLORS_BGR["unknown"])
            cv2.rectangle(strip, (x0, y0 + 1), (x1px, y1), color, -1)

    return np.concatenate([frame_bgr, strip], axis=0)


def save_state_logs_csv(classifier: ActivityClassifier, path: str | Path) -> None:
    """Dump per-(track, frame) state to CSV."""
    import csv

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["track_id", "timestamp_s", "state"])
        for tid, h in classifier.histories.items():
            for t, s in h.state_log:
                w.writerow([tid, f"{t:.4f}", s])
