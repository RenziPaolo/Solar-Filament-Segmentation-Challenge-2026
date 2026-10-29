"""Tests for spine/centerline target generation."""
from __future__ import annotations

import numpy as np
import pytest

from solarseg.spine import centerlines_from_segmentations, soft_centerline_target, spine_from_annotations


def test_centerlines_from_polygon():
    """Skeletonize a simple rectangular polygon."""
    # A 20x10 rectangle as a COCO polygon
    poly = [[5, 5, 25, 5, 25, 15, 5, 15, 5, 5]]
    canvas = centerlines_from_segmentations(poly, height=64, width=64)
    assert canvas.shape == (64, 64)
    assert canvas.dtype == np.float32
    # Skeleton of a horizontal rectangle should be roughly 1 pixel wide
    assert canvas.sum() > 0, "Centerline should be non-empty"
    assert canvas.max() <= 1.0 and canvas.min() >= 0.0


def test_centerlines_empty_segmentation():
    """Empty segmentations should produce an all-zero canvas."""
    canvas = centerlines_from_segmentations([], height=32, width=32)
    assert canvas.shape == (32, 32)
    assert canvas.sum() == 0


def test_soft_centerline_target():
    """Union-based skeletonization of multiple touching polygons."""
    poly1 = [[2, 2, 10, 2, 10, 5, 2, 5, 2, 2]]
    poly2 = [[2, 6, 10, 6, 10, 9, 2, 9, 2, 6]]  # touches poly1
    canvas = soft_centerline_target([poly1, poly2], height=32, width=32)
    assert canvas.shape == (32, 32)
    assert canvas.sum() > 0


def test_spine_from_annotations_with_endpoints():
    """Draw spine line from endpoint annotations."""
    anns = [
        {"spine": [[10, 10], [30, 30]]},
        {"spine": [[5, 5], [15, 15]]},
    ]
    canvas = spine_from_annotations(anns, height=64, width=64, thickness=1)
    assert canvas.shape == (64, 64)
    assert canvas.sum() > 0, "Spine lines should be drawn"
    # Endpoints should be in the canvas
    assert canvas[10, 10] > 0 and canvas[30, 30] > 0


def test_spine_from_annotations_without_spine():
    """Annotations without spine data should produce an empty canvas."""
    anns = [{"segmentation": [[1, 2, 3, 4]]}]  # no spine field
    canvas = spine_from_annotations(anns, height=32, width=32)
    assert canvas.shape == (32, 32)
    assert canvas.sum() == 0


@pytest.mark.skipif(
    __import__("skimage.morphology", fromlist=["skeletonize"]) is None,
    reason="scikit-image not installed",
)
def test_thickness_affects_canvas():
    """Thicker centerlines should cover more pixels."""
    poly = [[5, 5, 25, 5, 25, 15, 5, 15, 5, 5]]
    thin = centerlines_from_segmentations(poly, height=64, width=64, thickness=1)
    thick = centerlines_from_segmentations(poly, height=64, width=64, thickness=3)
    assert thick.sum() >= thin.sum(), "Thicker centerlines should cover >= pixels"