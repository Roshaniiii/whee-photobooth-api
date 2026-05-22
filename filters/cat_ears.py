import os
import cv2
import numpy as np

# ── Asset path — expects: assets/cat_ears.png next to filters/ folder ─────────
_ASSET_PATH = os.path.join(os.path.dirname(__file__), "..", "assets", "cat_ears.png")

# Cached processed asset (loaded once, reused every frame)
_ears_rgba_cache: np.ndarray | None = None


def _load_ears_rgba() -> np.ndarray:
    """
    Load cat_ears.png and convert black background → transparent alpha.

    The PNG has no alpha channel — background is pure/near-black (R,G,B < 40).
    We:
      1. Read as BGR
      2. Build an alpha mask: luminance drives transparency
             lum < 30  → fully transparent
             30–80     → linear ramp (soft edge pixels)
             > 80      → fully opaque
      3. Merge into BGRA and return.

    Result is cached so the file is read only once.
    """
    global _ears_rgba_cache
    if _ears_rgba_cache is not None:
        return _ears_rgba_cache

    bgr = cv2.imread(_ASSET_PATH, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(
            f"cat_ears.png not found at: {_ASSET_PATH}\n"
            "Place the asset at  assets/cat_ears.png  relative to your project root."
        )

    # Luminance (perceived brightness) — better than raw channel max for grey fur
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Soft ramp: [0,30] → 0 alpha,  [30,80] → linear,  [80,255] → 255 alpha
    alpha = np.clip((gray - 30) / (80 - 30), 0.0, 1.0) * 255
    alpha = alpha.astype(np.uint8)

    # Optional: gentle Gaussian to anti-alias the ear silhouette edges
    alpha = cv2.GaussianBlur(alpha, (3, 3), sigmaX=1.0)

    bgra = cv2.merge([bgr[:, :, 0],
                      bgr[:, :, 1],
                      bgr[:, :, 2],
                      alpha])

    _ears_rgba_cache = bgra
    return bgra


def _overlay_rgba(base: np.ndarray, overlay_bgra: np.ndarray,
                  x: int, y: int) -> np.ndarray:
    """
    Alpha-composite `overlay_bgra` (BGRA) onto `base` (BGR) at position (x, y).

    Handles partial out-of-bounds placement — only the visible portion is drawn.

    Args:
        base        : BGR uint8 frame
        overlay_bgra: BGRA uint8 overlay image
        x, y        : top-left corner of the overlay on `base` (can be negative)

    Returns:
        New BGR uint8 frame with overlay composited.
    """
    result = base.copy()
    oh, ow = overlay_bgra.shape[:2]
    fh, fw = base.shape[:2]

    # Compute the intersection rectangle
    x1_b = max(x, 0);          y1_b = max(y, 0)
    x2_b = min(x + ow, fw);    y2_b = min(y + oh, fh)

    if x2_b <= x1_b or y2_b <= y1_b:
        return result   # completely out of frame

    # Corresponding region in the overlay
    x1_o = x1_b - x;   y1_o = y1_b - y
    x2_o = x1_o + (x2_b - x1_b)
    y2_o = y1_o + (y2_b - y1_b)

    roi     = result[y1_b:y2_b, x1_b:x2_b].astype(np.float32)
    patch   = overlay_bgra[y1_o:y2_o, x1_o:x2_o]
    bgr_p   = patch[:, :, :3].astype(np.float32)
    alpha_p = (patch[:, :, 3].astype(np.float32) / 255.0)[:, :, np.newaxis]

    blended = roi * (1.0 - alpha_p) + bgr_p * alpha_p
    result[y1_b:y2_b, x1_b:x2_b] = np.clip(blended, 0, 255).astype(np.uint8)
    return result


def apply_cat_ears(img: np.ndarray) -> np.ndarray:
    """
    Cat Ears overlay filter.

    Pipeline:
      1. Detect the largest face with Haar cascade.
      2. Scale the ears PNG so its width = 130 % of the face width
         (ears spread naturally just beyond the face sides).
      3. Position: horizontally centred on the face, vertically so the
         bottom of the ears asset aligns with the top of the face box
         (with a 10 % downward nudge so ears sit just *on* the head).
      4. Alpha-composite onto the frame.

    If no face is detected the original frame is returned unchanged.

    Args:
        img: BGR uint8 image.

    Returns:
        BGR uint8 image with cat ears composited.
    """
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
    )

    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    if len(faces) == 0:
        return img.copy()

    # Largest detected face
    fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])

    # ── Load & scale ears ─────────────────────────────────────────────────────
    ears_bgra = _load_ears_rgba()
    oh, ow    = ears_bgra.shape[:2]

    # Target width = 130 % of face width so ears spread past the face edges
    target_w = int(fw * 1.30)
    scale    = target_w / ow
    target_h = int(oh * scale)

    ears_scaled = cv2.resize(ears_bgra, (target_w, target_h),
                             interpolation=cv2.INTER_AREA)

    # ── Position ──────────────────────────────────────────────────────────────
    # Horizontally: centre the ears sprite on the face centre
    face_cx = fx + fw // 2
    x = face_cx - target_w // 2

    # Vertically: bottom of the ears sprite sits at face top + 10 % of fh
    # The 10 % nudge down lets the ear bases overlap the hairline naturally.
    y = fy - target_h + int(fh * 0.22)

    # ── Composite ─────────────────────────────────────────────────────────────
    result = _overlay_rgba(img, ears_scaled, x, y)
    return result