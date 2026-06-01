import os
import cv2
import numpy as np
from typing import Optional
from rembg import remove, new_session

# ── Asset path ────────────────────────────────────────────────────────────────
_ASSETS_DIR: str = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "assets", "backgrounds"
)

# ── rembg session — loaded once, reused every frame ──────────────────────────
# u2net_human_seg is optimised specifically for people — cleaner edges
# than the default u2net model, especially around hair and shoulders.
_session = None

def _get_session():
    global _session
    if _session is None:
        _session = new_session("u2net_human_seg")
    return _session


# ── Background cache ──────────────────────────────────────────────────────────
_bg_cache: dict = {}

def _load_bg(filename: str, target_w: int, target_h: int) -> np.ndarray:
    """
    Load and resize a background image to match frame dimensions.
    Cached per filename — resized fresh only when frame size changes.
    """
    cache_key = f"{filename}_{target_w}_{target_h}"
    if cache_key in _bg_cache:
        return _bg_cache[cache_key]

    path = os.path.join(_ASSETS_DIR, filename)
    bg = cv2.imread(path, cv2.IMREAD_COLOR)
    if bg is None:
        # Fallback: solid gradient if file not found
        bg = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        for y in range(target_h):
            t = y / target_h
            bg[y, :] = (
                int(255 * (1 - t)),
                int(180 * t),
                int(255 * t),
            )
    else:
        # Crop to same aspect ratio then resize — no stretching
        bh, bw = bg.shape[:2]
        target_aspect = target_w / target_h
        bg_aspect     = bw / bh

        if bg_aspect > target_aspect:
            # BG is wider — crop sides
            new_bw = int(bh * target_aspect)
            ox     = (bw - new_bw) // 2
            bg     = bg[:, ox: ox + new_bw]
        else:
            # BG is taller — crop top/bottom
            new_bh = int(bw / target_aspect)
            oy     = (bh - new_bh) // 2
            bg     = bg[oy: oy + new_bh, :]

        bg = cv2.resize(bg, (target_w, target_h), interpolation=cv2.INTER_AREA)

    _bg_cache[cache_key] = bg
    return bg


def apply_background_replace(
    img: np.ndarray,
    bg_filename: str = "bg1.jpg",
) -> np.ndarray:
    """
    Background Replacement filter.

    Uses rembg (U2Net human segmentation model) to extract the person
    from the frame, then composites them onto a custom background image.

    Pipeline:
      1. Convert BGR frame to RGB PNG bytes (rembg requires RGB input).
      2. rembg.remove() → returns RGBA PNG bytes with background removed.
      3. Decode RGBA → split into RGB person + alpha mask.
      4. Load + resize background image to match frame dimensions.
      5. Alpha-composite: person over background using the mask.
      6. Optional edge softening — slight blur on mask edges to reduce
         the hard cutout look, especially around fine hair.

    Args:
        img         : BGR uint8 image (H x W x 3)
        bg_filename : filename of background image in assets/backgrounds/
                      e.g. "bg1.jpg", "bg2.jpg", "bg3.jpg"

    Returns:
        BGR uint8 image with background replaced.
    """
    h: int = img.shape[0]
    w: int = img.shape[1]

    # ── 1. Convert BGR → RGB → PNG bytes for rembg ───────────────────────────
    rgb: np.ndarray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    _, png_buf = cv2.imencode('.png', rgb)
    png_bytes: bytes = png_buf.tobytes()

    # ── 2. Remove background → RGBA PNG bytes ────────────────────────────────
    result_bytes: bytes = remove(
        png_bytes,
        session=_get_session(),
        alpha_matting=True,              # better hair/edge quality
        alpha_matting_foreground_threshold=240,
        alpha_matting_background_threshold=10,
        alpha_matting_erode_size=10,
    )

    # ── 3. Decode RGBA ────────────────────────────────────────────────────────
    nparr: np.ndarray  = np.frombuffer(result_bytes, np.uint8)
    rgba:  np.ndarray  = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

    if rgba is None or rgba.shape[2] < 4:
        return img.copy()   # fallback — return original if decode fails

    person_rgb: np.ndarray = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)
    alpha_raw:  np.ndarray = rgba[:, :, 3].astype(np.float32) / 255.0

    # ── 4. Soften mask edges — reduces hard cutout look ───────────────────────
    alpha_soft: np.ndarray = cv2.GaussianBlur(
        alpha_raw, (5, 5), sigmaX=1.2
    )
    alpha3: np.ndarray = alpha_soft[:, :, np.newaxis]

    # ── 5. Load background ────────────────────────────────────────────────────
    background: np.ndarray = _load_bg(bg_filename, w, h)

    # ── 6. Composite: person over background ─────────────────────────────────
    person_f: np.ndarray = person_rgb.astype(np.float32)
    bg_f:     np.ndarray = background.astype(np.float32)

    composite: np.ndarray = person_f * alpha3 + bg_f * (1.0 - alpha3)
    return np.clip(composite, 0, 255).astype(np.uint8)