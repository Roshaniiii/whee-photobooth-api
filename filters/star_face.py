import os
import cv2
import numpy as np
from typing import Optional, Tuple, List

# ── Cascade ───────────────────────────────────────────────────────────────────
_FACE_XML: str = cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"

# ── Asset path ────────────────────────────────────────────────────────────────
_ASSET_PATH: str = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "assets", "star.png"
)

# ── Star PNG intrinsics (397×395, star centre at 198,200) ─────────────────────
_SRC_W:        int = 397
_SRC_H:        int = 395
_SRC_ANCHOR_X: int = 198
_SRC_ANCHOR_Y: int = 200

# ── Recolor targets (BGR) — purple + pink focus matching reference ────────────
_STAR_COLOURS: List[Tuple[int, int, int]] = [
    # HEX       RGB              BGR (swap R and B)
    (221, 169, 255),  # #ffa9dd  → BGR: (221, 169, 255)
    (  0, 183, 254),  # #feb700  → BGR: (  0, 183, 254)
    (111, 226, 255),  # #ffe26f  → BGR: (111, 226, 255)
    ( 47, 224, 196),  # #c4e02f  → BGR: ( 47, 224, 196)
    (217, 213, 124),  # #7cd5d9  → BGR: (217, 213, 124)
    (255, 173, 153),  # #99adff  → BGR: (255, 173, 153)
    (248, 190, 231),  # #e7bef8  → BGR: (248, 190, 231)
]

# ── Star placement config ─────────────────────────────────────────────────────
# 3 stars on each cheek in a *.* triangle pattern (two bottom, one top-centre)
# (pfx, pfy, size_frac, colour_idx, rotation_deg)
#   pfx: 0.0=left edge  0.5=centre  1.0=right edge
#   pfy: 0.0=top        0.5=mid     1.0=chin
_STAR_PLACEMENTS: List[Tuple[float, float, float, int, float]] = [
    # ── Left cheek  *.*  (top-centre, bottom-left, bottom-right) ─────────
    (0.20, 0.54, 0.08, 2,   0),   # left cheek top-centre
    (0.27, 0.62, 0.08, 6,   0),   # left cheek bottom-right

    # ── Nose bridge — unchanged size, colour updated ───────────────────────
    (0.42, 0.47, 0.05, 1,  12),   # nose left
    (0.50, 0.43, 0.05, 3,  -5),   # nose top centre
    (0.58, 0.47, 0.05, 5,  18),   # nose right

    # ── Right cheek  *.*  (top-centre, bottom-left, bottom-right) ────────
    (0.80, 0.54, 0.08, 4,   0),   # right cheek top-centre
    (0.73, 0.62, 0.08, 5,   0),   # right cheek bottom-left
]

# Cache
_star_colour_cache: Optional[List[np.ndarray]] = None


# ─────────────────────────────────────────────────────────────────────────────
def _build_star_mask(bgr: np.ndarray) -> np.ndarray:
    """
    Extract star shape as alpha mask from the blue star PNG.
    Background is light grey checkerboard (~175-195 BGR).
    Star is blue (HSV hue 85-130, sat > 60).
    """
    hsv: np.ndarray = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue: np.ndarray = hsv[:, :, 0].astype(np.float32)
    sat: np.ndarray = hsv[:, :, 1].astype(np.float32)

    hue_mask: np.ndarray = (
        np.clip((hue - 80.0) / 8.0, 0.0, 1.0) *
        np.clip((135.0 - hue) / 8.0, 0.0, 1.0)
    )
    sat_mask: np.ndarray = np.clip((sat - 50.0) / 40.0, 0.0, 1.0)

    alpha_f: np.ndarray = hue_mask * sat_mask * 255.0
    alpha: np.ndarray   = np.clip(alpha_f, 0, 255).astype(np.uint8)
    alpha = cv2.GaussianBlur(alpha, (5, 5), sigmaX=1.5)
    return alpha


def _make_coloured_star(bgr: np.ndarray, alpha: np.ndarray,
                        colour_bgr: Tuple[int, int, int]) -> np.ndarray:
    """
    Recolour the star to a solid colour while keeping its alpha shape.
    70% flat colour + 30% luminance texture from original.
    Returns a BGRA uint8 image.
    """
    h, w = bgr.shape[:2]
    coloured: np.ndarray = np.zeros((h, w, 3), dtype=np.uint8)
    coloured[:] = colour_bgr

    gray: np.ndarray  = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lum: np.ndarray   = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(np.float32)
    col: np.ndarray   = coloured.astype(np.float32)
    mixed: np.ndarray = np.clip(col * 1.0, 0, 255).astype(np.uint8)

    b, g, r = cv2.split(mixed)
    return cv2.merge([b, g, r, alpha])


def _load_star_variants() -> List[np.ndarray]:
    """
    Load star.png once, extract alpha, generate one BGRA variant per colour.
    Cached after first call.
    """
    global _star_colour_cache
    if _star_colour_cache is not None:
        return _star_colour_cache

    bgr: np.ndarray = cv2.imread(_ASSET_PATH, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(
            "star.png not found at: {}\n"
            "Place the file at  assets/star.png  relative to your project root."
            .format(_ASSET_PATH)
        )

    alpha: np.ndarray  = _build_star_mask(bgr)
    _star_colour_cache = [
        _make_coloured_star(bgr, alpha, c) for c in _STAR_COLOURS
    ]
    return _star_colour_cache


# ─────────────────────────────────────────────────────────────────────────────
def _overlay_bgra(
    base:    np.ndarray,
    overlay: np.ndarray,
    cx:      int,
    cy:      int,
    size:    int,
    angle:   float,
    opacity: float,
) -> np.ndarray:
    """
    Resize, rotate and alpha-composite a BGRA overlay centred at (cx, cy).
    """
    if size < 4:
        return base

    resized: np.ndarray = cv2.resize(
        overlay, (size, size), interpolation=cv2.INTER_AREA
    )

    if abs(angle) > 0.5:
        M: np.ndarray = cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1.0)
        resized = cv2.warpAffine(
            resized, M, (size, size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )

    fh: int = base.shape[0]
    fw: int = base.shape[1]
    x: int  = cx - size // 2
    y: int  = cy - size // 2

    dst_x1: int = max(x, 0)
    dst_y1: int = max(y, 0)
    dst_x2: int = min(x + size, fw)
    dst_y2: int = min(y + size, fh)
    if dst_x2 <= dst_x1 or dst_y2 <= dst_y1:
        return base

    src_x1: int = dst_x1 - x
    src_y1: int = dst_y1 - y
    src_x2: int = src_x1 + (dst_x2 - dst_x1)
    src_y2: int = src_y1 + (dst_y2 - dst_y1)

    result: np.ndarray    = base.copy()
    roi:    np.ndarray    = result[dst_y1:dst_y2, dst_x1:dst_x2].astype(np.float32)
    patch:  np.ndarray    = resized[src_y1:src_y2, src_x1:src_x2]
    patch_bgr: np.ndarray = patch[:, :, :3].astype(np.float32)
    patch_a:   np.ndarray = patch[:, :, 3:4].astype(np.float32) / 255.0 * float(opacity)

    blended: np.ndarray = roi * (1.0 - patch_a) + patch_bgr * patch_a
    result[dst_y1:dst_y2, dst_x1:dst_x2] = np.clip(blended, 0, 255).astype(np.uint8)
    return result


# ─────────────────────────────────────────────────────────────────────────────
def apply_star_face(img: np.ndarray, star_opacity: float = 0.92) -> np.ndarray:
    """
    Star Face filter — scatters coloured star stickers on cheeks and nose.

    Pipeline:
      1. Detect largest face with Haar cascade.
      2. Load star.png → generate 7 recoloured BGRA variants (cached).
      3. Place stars at fixed anatomical positions (no flicker).

    Args:
        img          : BGR uint8 image (H x W x 3)
        star_opacity : star sticker opacity 0.0-1.0 (default 0.92)

    Returns:
        BGR uint8 image with stars applied.
        Returns copy of original if no face detected.
    """
    face_cascade: cv2.CascadeClassifier = cv2.CascadeClassifier(_FACE_XML)
    gray: np.ndarray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    raw_faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    if not isinstance(raw_faces, np.ndarray) or len(raw_faces) == 0:
        return img.copy()

    face_box: list = max(raw_faces.tolist(), key=lambda f: f[2] * f[3])
    fx: int = int(face_box[0])
    fy: int = int(face_box[1])
    fw: int = int(face_box[2])
    fh: int = int(face_box[3])

    # ── Load star variants ────────────────────────────────────────────────────
    star_variants: List[np.ndarray] = _load_star_variants()

    # ── Place stars at fixed positions ────────────────────────────────────────
    result: np.ndarray = img.copy()
    for (pfx, pfy, size_frac, colour_idx, rotation) in _STAR_PLACEMENTS:
        cx:      int = fx + int(fw * pfx)
        cy:      int = fy + int(fh * pfy)
        size:    int = max(8, int(fw * size_frac))
        variant      = star_variants[colour_idx % len(star_variants)]
        result = _overlay_bgra(result, variant, cx, cy, size, rotation, star_opacity)

    return result