"""Generate centerline / spine targets for multi-task learning.

The MAGFiLO annotations include per-filament spine endpoints and polygon
segmentations.  Two ways to produce a centerline target:

1. ``centerlines_from_segmentations`` — skeletonize each polygon mask and
   overlay the skeletons into one soft centerline map.
2. ``spine_from_annotations`` — draw a 1-pixel-wide line between each
   filament's annotated spine endpoints (when available).

Both return a float32 array in [0, 1] at image resolution.
"""
from __future__ import annotations

import numpy as np
from pycocotools import mask as mask_utils

try:
    from skimage.morphology import skeletonize
except ImportError:  # pragma: no cover
    skeletonize = None


def _decode_polygon(seg, height: int, width: int) -> np.ndarray:
    """Decode a single COCO polygon or RLE into a binary mask."""
    if isinstance(seg, dict) and "counts" in seg:
        rle = seg
        return mask_utils.decode(rle).astype(bool)
    # polygons: list of [x1, y1, x2, y2, ...]
    if isinstance(seg, list):
        if seg and isinstance(seg[0], (list, tuple)):
            polys = seg
        else:
            polys = [seg]
    else:
        polys = [seg]
    mask = np.zeros((height, width), dtype=np.uint8)
    for poly in polys:
        if len(poly) < 6:
            continue
        arr = np.array(poly, dtype=np.int32).reshape(-1, 2)
        import cv2
        cv2.fillPoly(mask, [arr], 1)
    return mask.astype(bool)


def centerlines_from_segmentations(
    segmentations: list, height: int, width: int, thickness: int = 1
) -> np.ndarray:
    """Skeletonize every filament polygon and merge into one centerline map.

    Parameters
    ----------
    segmentations : list
        List of COCO segmentation entries (polygons or RLE dicts).
    height, width : int
        Image dimensions.
    thickness : int
        Half-width of the centerline band (dilation around the skeleton).
    """
    if skeletonize is None:
        raise ImportError("scikit-image is required for centerline generation")

    canvas = np.zeros((height, width), dtype=np.float32)
    for seg in segmentations:
        mask = _decode_polygon(seg, height, width)
        if not mask.any():
            continue
        skel = skeletonize(mask)
        # thin skeleton to 1-pixel-wide centerline, then optionally thicken
        canvas = np.maximum(canvas, skel.astype(np.float32))
    if thickness > 1:
        import cv2
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * thickness - 1, 2 * thickness - 1))
        canvas = cv2.dilate(canvas.astype(np.uint8), kernel).astype(np.float32)
    return canvas


def spine_from_annotations(
    annotations: list[dict], height: int, width: int, thickness: int = 1
) -> np.ndarray:
    """Draw lines between annotated spine endpoints for each filament.

    The MAGFiLO annotations may include ``spine`` or ``point_coordinates``
    fields containing the two endpoints of the filament's spine.

    Parameters
    ----------
    annotations : list[dict]
        COCO annotation dicts that may contain spine endpoint data.
    height, width : int
        Image dimensions.
    thickness : int
        Line thickness in pixels.
    """
    import cv2

    canvas = np.zeros((height, width), dtype=np.float32)
    for ann in annotations:
        pts = None
        for key in ("spine", "point_coordinates", "line"):
            pts = ann.get(key)
            if pts is not None:
                break
        if pts is None:
            continue
        # Expect [[x1, y1], [x2, y2]] or [(x1, y1), (x2, y2]]
        if len(pts) >= 2:
            pt1 = tuple(int(round(float(v))) for v in pts[0])
            pt2 = tuple(int(round(float(v))) for v in pts[1])
            cv2.line(canvas, pt1, pt2, 1.0, thickness=thickness)
    return canvas


def soft_centerline_target(
    segmentations: list, height: int, width: int
) -> np.ndarray:
    """Combined centerline target: skeleton of the soft-segmentation union.

    Unlike ``centerlines_from_segmentations`` which processes each polygon
    independently, this function first builds a binary union mask and then
    skeletonizes the whole thing — useful when filaments touch.
    """
    if skeletonize is None:
        raise ImportError("scikit-image is required for centerline generation")

    union = np.zeros((height, width), dtype=bool)
    for seg in segmentations:
        union |= _decode_polygon(seg, height, width)
    if not union.any():
        return np.zeros((height, width), dtype=np.float32)
    skel = skeletonize(union)
    return skel.astype(np.float32)
