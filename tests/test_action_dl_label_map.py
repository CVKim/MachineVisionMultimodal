"""Test only the label-mapper of DeepActionClassifier — no model loaded."""

from __future__ import annotations

from mvmm.tracking.action_dl import map_kinetics_label


def test_walking_family():
    for s in ["walking the dog", "running", "jogging", "marching", "hiking"]:
        assert map_kinetics_label(s) == "walking", s


def test_lifting_family():
    for s in ["lifting weights", "carrying baby", "picking apples", "pushing cart", "loading dishwasher"]:
        assert map_kinetics_label(s) == "lifting", s


def test_working_family():
    for s in [
        "assembling furniture",
        "welding metal",
        "drilling a hole",
        "cleaning floor",
        "typing on keyboard",
    ]:
        assert map_kinetics_label(s) == "working", s


def test_idle_family():
    for s in ["sitting at a table", "standing in line", "waiting in queue", "reading book", "watching tv"]:
        assert map_kinetics_label(s) == "idle", s


def test_unknown_falls_through():
    assert map_kinetics_label("zumba dancing") == "unknown"
    assert map_kinetics_label("blowing bubbles underwater") == "unknown"
