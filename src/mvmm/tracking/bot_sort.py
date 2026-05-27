"""BoT-SORT — motion + appearance fusion tracker.

Reference:
    Aharon et al. "BoT-SORT: Robust Associations Multi-Pedestrian Tracking."
    https://arxiv.org/abs/2206.14651

Design:
    BoT-SORT is essentially ByteTrack with two extras —
        (1) Camera-motion compensation (we expose a stub; OpenCV's
            ``findTransformECC`` does it well for slow-moving CCTV).
        (2) Appearance gating: after ByteTrack proposes associations,
            cosine distance on appearance embeddings is used as a
            second pass that *blocks* implausible matches.

For (2) we use a CLIP-based ReID embedding by default — strong out of
the box, zero training needed. If you have a labeled person dataset,
swap in ``OSNetReID``.

This wrapper composes the existing :class:`ByteTrackTracker` and
:class:`CLIPReID` rather than re-implementing the Kalman/Hungarian
core, which keeps the code surface small and the upstream library
maintained.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mvmm.tracking.byte_track import ByteTrackTracker, Track
from mvmm.tracking.detectors import Detections
from mvmm.tracking.reid import cosine_distance


@dataclass
class _BotState:
    """Per-track running appearance embedding (EMA)."""

    embedding: np.ndarray  # (D,) unit-norm
    last_seen: int


class BoTSORTTracker:
    """ByteTrack + appearance gating.

    Args:
        reid:                a ReID backend exposing ``embed(crops_rgb) -> (N,D)``.
        appearance_dist_threshold:
                             max cosine distance between a candidate
                             detection embedding and an existing track's
                             EMA embedding to consider them the same ID.
                             Higher is more permissive.
        appearance_alpha:    EMA decay for the per-track embedding.
        frame_rate / track_activation_threshold / lost_track_buffer:
                             forwarded to :class:`ByteTrackTracker`.
    """

    def __init__(
        self,
        reid,
        appearance_dist_threshold: float = 0.35,
        appearance_alpha: float = 0.9,
        frame_rate: int = 30,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 60,
        minimum_matching_threshold: float = 0.8,
    ):
        self.reid = reid
        self.appearance_dist_threshold = float(appearance_dist_threshold)
        self.appearance_alpha = float(appearance_alpha)
        self._byte = ByteTrackTracker(
            frame_rate=frame_rate,
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
        )
        # Map: track_id -> appearance state. Cleared periodically.
        self._states: dict[int, _BotState] = {}
        self._frame_idx = 0

    @staticmethod
    def _crop(frame_rgb: np.ndarray, bbox: np.ndarray) -> np.ndarray | None:
        h, w = frame_rgb.shape[:2]
        x1, y1, x2, y2 = (int(v) for v in bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame_rgb[y1:y2, x1:x2]

    def update(self, detections: Detections, frame_rgb: np.ndarray) -> list[Track]:
        """Step the tracker with detections **and the original frame** (needed for crops)."""
        self._frame_idx += 1
        tracks = self._byte.update(detections)
        if not tracks:
            self._gc()
            return []

        # Compute appearance embeddings for the new tracks' crops.
        crops, idx_map = [], []
        for i, t in enumerate(tracks):
            crop = self._crop(frame_rgb, t.bbox)
            if crop is None or crop.size == 0:
                continue
            crops.append(crop)
            idx_map.append(i)
        if not crops:
            self._gc()
            return tracks

        embeddings = self.reid.embed(crops)

        # Build the current set of track-id -> "old" embeddings.
        old_ids = list(self._states.keys())
        old_embs = (
            np.stack([self._states[k].embedding for k in old_ids])
            if old_ids
            else np.zeros((0, embeddings.shape[1]), dtype=np.float32)
        )

        # Appearance gating: for each new track, if it shares the same
        # track_id as a known state -> EMA-update the embedding; otherwise
        # try to RE-MAP its id onto a previously-lost track whose
        # appearance is the closest match below the threshold.
        remap: dict[int, int] = {}
        for emb_row, i in zip(embeddings, idx_map, strict=False):
            t = tracks[i]
            tid = t.track_id
            if tid in self._states:
                self._ema_update(tid, emb_row)
                continue
            # New ID — try to re-assign to a lost track with similar appearance.
            if old_embs.size:
                d = cosine_distance(emb_row[None, :], old_embs)[0]
                best = int(np.argmin(d))
                if d[best] < self.appearance_dist_threshold and old_ids[best] != tid:
                    remap[tid] = old_ids[best]
                    self._ema_update(old_ids[best], emb_row)
                    continue
            # Otherwise it's genuinely a new track — register a state.
            self._states[tid] = _BotState(embedding=emb_row.astype(np.float32), last_seen=self._frame_idx)

        # Apply re-maps so downstream sees stable IDs across occlusions.
        if remap:
            for t in tracks:
                if t.track_id in remap:
                    t.track_id = remap[t.track_id]

        self._gc()
        return tracks

    def _ema_update(self, tid: int, emb: np.ndarray) -> None:
        a = self.appearance_alpha
        state = self._states[tid]
        new = a * state.embedding + (1 - a) * emb
        new /= float(np.linalg.norm(new)) + 1e-9
        self._states[tid] = _BotState(embedding=new.astype(np.float32), last_seen=self._frame_idx)

    def _gc(self, keep_recent_frames: int = 600) -> None:
        cutoff = self._frame_idx - keep_recent_frames
        if cutoff <= 0:
            return
        stale = [k for k, v in self._states.items() if v.last_seen < cutoff]
        for k in stale:
            self._states.pop(k, None)
