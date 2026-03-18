from __future__ import annotations

from pathlib import Path

from PIL import Image


def test_keypoint_overlay_handles_empty_and_non_empty_points():
    from frontend.components.keypoint_overlay import render_keypoint_overlay

    image = Image.new("L", (128, 128), color=64)
    empty_overlay = render_keypoint_overlay(image, [])
    filled_overlay = render_keypoint_overlay(image, [(32.0, 40.0), (84.0, 88.0)], labels=["P1", "P2"])

    assert empty_overlay.size == image.size
    assert filled_overlay.size == image.size
    assert list(empty_overlay.getdata()) != list(filled_overlay.getdata())


def test_render_viewer_education_mode_handles_missing_keypoints_without_crash(monkeypatch):
    from frontend.components import viewer

    calls: dict[str, list[object]] = {"image": [], "markdown": [], "info": [], "caption": []}
    monkeypatch.setattr(viewer.st, "image", lambda *args, **kwargs: calls["image"].append((args, kwargs)))
    monkeypatch.setattr(viewer.st, "markdown", lambda *args, **kwargs: calls["markdown"].append((args, kwargs)))
    monkeypatch.setattr(viewer.st, "info", lambda message: calls["info"].append(message))
    monkeypatch.setattr(viewer.st, "caption", lambda message: calls["caption"].append(message))

    preview = Image.new("L", (96, 96), color=80)
    preview_metadata = {"modality": "DX", "rows": 96, "columns": 96, "frames": 1, "photometric_interpretation": "MONOCHROME2"}
    result = {
        "disease_detected": False,
        "confidence": 0.4,
        "metrics": {
            "runtime_model_loaded": 1.0,
            "model_probability": 0.4,
            "model_threshold": 0.5,
            "keypoint_model_loaded": 0.0,
            "keypoint_count": 0.0,
        },
        "keypoints": [],
    }

    viewer.render_viewer(preview, preview_metadata, result=result, mode="education", show_keypoints=True)

    assert calls["image"]
    assert calls["info"]
    assert "Анатомические ориентиры недоступны" in str(calls["info"][0])


def test_overlay_keypoint_labels_defaults_to_neutral_labels():
    from frontend.utils.keypoint_labels import overlay_keypoint_labels

    assert overlay_keypoint_labels() == ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"]


def test_render_viewer_compat_handles_stale_render_viewer_signature(monkeypatch):
    from frontend import app

    calls: list[tuple[object, object, object]] = []

    def stale_render_viewer(preview, preview_metadata, *, result=None):
        calls.append((preview, preview_metadata, result))

    monkeypatch.setattr(app, "render_viewer", stale_render_viewer)
    app._render_viewer_compat(
        "preview",
        {"modality": "DX"},
        result={"confidence": 0.5},
        mode="education",
        show_keypoints=True,
    )

    assert calls == [("preview", {"modality": "DX"}, {"confidence": 0.5})]
