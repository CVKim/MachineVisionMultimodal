"""Convenience re-export of GroundingDINO for the zeroshot namespace.

The detector implementation lives in ``mvmm.tracking.detectors`` (it's
the same model used in the tracking pipeline). Importing from here
makes the open-vocabulary detection API obviously discoverable under
``mvmm.zeroshot``.
"""

from __future__ import annotations

from mvmm.tracking.detectors import GroundingDINODetector

__all__ = ["GroundingDINODetector"]
