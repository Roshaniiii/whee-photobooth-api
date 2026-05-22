import os
import cv2
import numpy as np
from typing import Optional, Tuple

# ── Cascade paths ─────────────────────────────────────────────────────────────
_CASCADE_DIR: str = cv2.data.haarcascades
_FACE_XML:    str = _CASCADE_DIR + "haarcascade_frontalface_alt2.xml"

# ── Asset path ────────────────────────────────────────────────────────────────
_ASSET_PATH: str = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "assets", "blush.png"
)

# ── Blush PNG intrinsics (727×519, pink mark centred at 361,261) ──────────────
_SRC_W:        int = 727
_SRC_H:        int = 519
_SRC_ANCHOR_X: int = 361
_SRC_ANCHOR_Y: int = 261

# Module-level cache — loaded once, reused every frame
_blush_bgra_cache: Optional[np.ndarray] = None


# ─────────────────────────────────────────────────────────────────────────────
def _load_blush_bgra() -> np.ndarray:
    """
    Load blush.png and return a BGRA uint8 array.

    The PNG has a neutral grey background (~170,170,166 BGR) with no alpha.
    Alpha is synthesised from the HSV hue+saturation of the pink blush mark:
      • Pink hue 125-178 (OpenCV 0-180 scale) AND saturation > 40 → opaque
      • Grey background (low saturation)                          → transparent
      • Silhouette edges softened with a Gaussian blur.
    Cached after first call — file I/O happens only once.
    """
    global _blush_bgra_cache
    if _blush_bgra_cache is not None:
        return _blush_bgra_cache

    bgr: np.ndarray = cv2.imread(_ASSET_PATH, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(
            "blush.png not found at: {}\n"
            "Place the file at  assets/blush.png  relative to your project root."
            .format(_ASSET_PATH)
        )

    hsv: np.ndarray = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue: np.ndarray = hsv[:, :, 0].astype(np.float32)   # 0–180
    sat: np.ndarray = hsv[:, :, 1].astype(np.float32)   # 0–255

    # Soft hue window 125-178 with 5-px ramps on each side
    hue_mask: np.ndarray = (
        np.clip((hue - 125.0) / 5.0, 0.0, 1.0) *
        np.clip((178.0 - hue) / 5.0, 0.0, 1.0)
    )
    # Saturation ramp: transparent below 40, opaque above 100
    sat_mask: np.ndarray = np.clip((sat - 40.0) / 60.0, 0.0, 1.0)

    alpha_f: np.ndarray = hue_mask * sat_mask * 255.0
    alpha:   np.ndarray = np.clip(alpha_f, 0.0, 255.0).astype(np.uint8)
    alpha = cv2.GaussianBlur(alpha, (7, 7), sigmaX=2.5)

    b, g, r = cv2.split(bgr)
    _blush_bgra_cache = cv2.merge([b, g, r, alpha])
    return _blush_bgra_cache


# ─────────────────────────────────────────────────────────────────────────────
def _overlay_bgra(
    base:    np.ndarray,
    overlay: np.ndarray,
    x:       int,
    y:       int,
    opacity: float = 1.0,
) -> np.ndarray:
    """
    Alpha-composite a BGRA overlay onto a BGR base at top-left (x, y).
    Out-of-bounds positions are handled gracefully.
    """
    result: np.ndarray = base.copy()

    oh: int = overlay.shape[0]
    ow: int = overlay.shape[1]
    fh: int = base.shape[0]
    fw: int = base.shape[1]

    dst_x1: int = max(x, 0)
    dst_y1: int = max(y, 0)
    dst_x2: int = min(x + ow, fw)
    dst_y2: int = min(y + oh, fh)

    if dst_x2 <= dst_x1 or dst_y2 <= dst_y1:
        return result

    src_x1: int = dst_x1 - x
    src_y1: int = dst_y1 - y
    src_x2: int = src_x1 + (dst_x2 - dst_x1)
    src_y2: int = src_y1 + (dst_y2 - dst_y1)

    roi:   np.ndarray = result[dst_y1:dst_y2, dst_x1:dst_x2].astype(np.float32)
    patch: np.ndarray = overlay[src_y1:src_y2, src_x1:src_x2]

    patch_bgr: np.ndarray = patch[:, :, :3].astype(np.float32)
    patch_a:   np.ndarray = patch[:, :, 3:4].astype(np.float32) / 255.0 * float(opacity)

    blended: np.ndarray = roi * (1.0 - patch_a) + patch_bgr * patch_a
    result[dst_y1:dst_y2, dst_x1:dst_x2] = np.clip(blended, 0.0, 255.0).astype(np.uint8)
    return result


# ─────────────────────────────────────────────────────────────────────────────
def _get_cheek_centres(
    fx: int, fy: int, fw: int, fh: int,
) -> Tuple[Tuple[int, int], Tuple[int, int], float]:
    """
    Compute left/right cheek centres using pure face geometry.

    No eye detection — consistent positioning with/without glasses.
    Anatomical cheek position:
      • 58% down face height  (below eye line, above mouth)
      • 22% inward from each face edge (on the cheekbone)
    Returns ((l_cx, l_cy), (r_cx, r_cy), scale).
    """
    l_cx: int = fx + int(fw * 0.22)
    r_cx: int = fx + int(fw * 0.78)
    l_cy: int = fy + int(fh * 0.58)
    r_cy: int = l_cy

    scale: float = float(fw) * 0.38 / float(_SRC_W)
    return (l_cx, l_cy), (r_cx, r_cy), scale


# ─────────────────────────────────────────────────────────────────────────────
def apply_blush(img: np.ndarray, opacity: float = 0.85) -> np.ndarray:
    """
    Blush overlay filter.

    Detects the largest face, anchors a scaled blush PNG to each cheek
    using eye positions (falls back to geometry when eyes aren't found),
    then alpha-composites both marks onto the frame.

    Args:
        img     : BGR uint8 image (H x W x 3)
        opacity : blush strength 0.0-1.0 (default 0.85)

    Returns:
        BGR uint8 image with blush on both cheeks,
        or a copy of the original if no face is detected.
    """
    face_cascade: cv2.CascadeClassifier = cv2.CascadeClassifier(_FACE_XML)

    gray: np.ndarray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # detectMultiScale returns () when nothing found
    raw_faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    if not isinstance(raw_faces, np.ndarray) or len(raw_faces) == 0:
        return img.copy()

    faces_list: list = raw_faces.tolist()
    face_box:   list = max(faces_list, key=lambda f: f[2] * f[3])

    fx: int = int(face_box[0])
    fy: int = int(face_box[1])
    fw: int = int(face_box[2])
    fh: int = int(face_box[3])

    # Pure geometry — no eye detection needed
    (l_cx, l_cy), (r_cx, r_cy), scale = _get_cheek_centres(fx, fy, fw, fh)

    # ── Load, scale, anchor ───────────────────────────────────────────────────
    blush_bgra:   np.ndarray = _load_blush_bgra()
    new_w:        int        = max(4, int(_SRC_W * scale))
    new_h:        int        = max(4, int(_SRC_H * scale))
    blush_scaled: np.ndarray = cv2.resize(
        blush_bgra, (new_w, new_h), interpolation=cv2.INTER_AREA
    )

    anchor_x: int = int(_SRC_ANCHOR_X / _SRC_W * new_w)
    anchor_y: int = int(_SRC_ANCHOR_Y / _SRC_H * new_h)

    # ── Left cheek ────────────────────────────────────────────────────────────
    result: np.ndarray = _overlay_bgra(
        img, blush_scaled,
        l_cx - anchor_x,
        l_cy - anchor_y,
        opacity,
    )

    # ── Right cheek (horizontally flipped) ───────────────────────────────────
    blush_flipped: np.ndarray = cv2.flip(blush_scaled, 1)
    result = _overlay_bgra(
        result, blush_flipped,
        r_cx - (new_w - anchor_x),
        r_cy - anchor_y,
        opacity,
    )

    return result