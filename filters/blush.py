import cv2
import numpy as np

# ── Cascade paths (bundled with opencv-python, no extra install) ──────────────
_CASCADE_DATA = cv2.data.haarcascades
_face_cascade = cv2.CascadeClassifier(_CASCADE_DATA + "haarcascade_frontalface_alt2.xml")
_eye_cascade  = cv2.CascadeClassifier(_CASCADE_DATA + "haarcascade_eye.xml")


# ── Colour: Soft rose-pink in BGR ─────────────────────────────────────────────
# (219, 153, 189)  →  muted, warm rose — not hot-pink, not orange
_BLUSH_BGR = np.array([189, 153, 219], dtype=np.float32)


def _draw_cheek(canvas: np.ndarray, cx: int, cy: int, rx: int, ry: int,
                color_bgr: np.ndarray, alpha: float) -> np.ndarray:
    """
    Paint one soft elliptical blush patch onto `canvas`.

    Uses a float32 overlay so the Gaussian feathering is smooth,
    then blends back with addWeighted.

    Args:
        canvas   : BGR uint8 image to draw on (modified in-place copy)
        cx, cy   : ellipse centre (pixels)
        rx, ry   : semi-axes (pixels)  — rx is horizontal, ry vertical
        color_bgr: BGR float32 colour
        alpha    : peak opacity (0.0–1.0)
    Returns:
        Blended BGR uint8 image.
    """
    h, w = canvas.shape[:2]

    # --- soft ellipse mask ---------------------------------------------------
    # Build a float32 mask the same size as the image, draw a filled white
    # ellipse, then Gaussian-blur it heavily so edges dissolve naturally.
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(mask,
                center=(cx, cy),
                axes=(rx, ry),
                angle=0,
                startAngle=0,
                endAngle=360,
                color=1.0,
                thickness=-1)

    # Blur radius scales with the ellipse size so feathering always looks right.
    blur_k = int(max(rx, ry) * 1.6) | 1   # must be odd
    blur_k = max(blur_k, 21)
    mask = cv2.GaussianBlur(mask, (blur_k, blur_k), sigmaX=blur_k / 3)

    # Scale mask to peak alpha
    mask = np.clip(mask * alpha, 0.0, 1.0)

    # --- colour overlay -------------------------------------------------------
    overlay = np.zeros_like(canvas, dtype=np.float32)
    overlay[:] = color_bgr

    base    = canvas.astype(np.float32)
    mask3   = mask[:, :, np.newaxis]           # broadcast over BGR channels
    blended = base * (1.0 - mask3) + overlay * mask3
    return np.clip(blended, 0, 255).astype(np.uint8)


def _cheek_centres_from_eyes(face_box, eyes, img_w: int):
    """
    Derive left/right cheek centres from detected eye positions.

    Cheek is placed:
      - horizontally: aligned with the outer edge of each eye
      - vertically  : ~1.0× eye-height below the eye's bottom edge

    Falls back to a face-geometry estimate when eyes aren't detected.
    """
    fx, fy, fw, fh = face_box

    if len(eyes) >= 2:
        # Sort eyes left-to-right (relative to image, not face)
        eyes_sorted = sorted(eyes, key=lambda e: e[0])
        ex0, ey0, ew0, eh0 = eyes_sorted[0]   # left eye (image-left)
        ex1, ey1, ew1, eh1 = eyes_sorted[1]   # right eye

        # Absolute pixel coords (eyes are relative to face ROI)
        l_cx = fx + ex0 + ew0 // 2            # left eye centre-x
        l_cy = fy + ey0 + eh0 + int(eh0 * 1.0)  # below left eye

        r_cx = fx + ex1 + ew1 // 2
        r_cy = fy + ey1 + eh1 + int(eh1 * 1.0)

        # Spread cheeks outward a little from eye centres
        spread = int(fw * 0.10)
        l_cx -= spread
        r_cx += spread

        rx = int(fw * 0.18)   # horizontal radius  ≈ 18 % of face width
        ry = int(fh * 0.13)   # vertical radius    ≈ 13 % of face height

    else:
        # Fallback: pure geometry from face box
        # Cheeks sit at ~62 % down, 22 % inward from each side
        l_cx = fx + int(fw * 0.22)
        r_cx = fx + int(fw * 0.78)
        cy   = fy + int(fh * 0.62)
        l_cy = r_cy = cy
        rx = int(fw * 0.17)
        ry = int(fh * 0.12)

    return (l_cx, l_cy), (r_cx, r_cy), rx, ry


def apply_blush(img: np.ndarray, alpha: float = 0.38) -> np.ndarray:
    """
    Soft & Natural blush filter.

    Detects the face with Haar cascade, uses eye positions to anchor
    rose-pink ellipses precisely on each cheek, then feathers them
    with a heavy Gaussian blur so the result looks like real skin flush.

    Args:
        img   : BGR uint8 image
        alpha : peak blush opacity — 0.38 gives a natural, visible flush
                without looking painted.  Range 0.0–1.0.

    Returns:
        BGR uint8 image with blush applied.
        If no face is detected the original image is returned unchanged.
    """
    result = img.copy()
    gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ── Face detection ────────────────────────────────────────────────────────
    faces = _face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    if len(faces) == 0:
        return result   # no face found — return untouched

    # Use the largest detected face (most prominent in frame)
    face_box = max(faces, key=lambda f: f[2] * f[3])
    fx, fy, fw, fh = face_box

    # ── Eye detection inside face ROI ─────────────────────────────────────────
    # Restrict search to the upper 55 % of the face (eyes don't live lower).
    eye_roi_gray = gray[fy: fy + int(fh * 0.55), fx: fx + fw]
    eyes = _eye_cascade.detectMultiScale(
        eye_roi_gray,
        scaleFactor=1.1,
        minNeighbors=10,
        minSize=(20, 20)
    )

    # ── Compute cheek positions ───────────────────────────────────────────────
    (l_cx, l_cy), (r_cx, r_cy), rx, ry = _cheek_centres_from_eyes(
        face_box, eyes, img.shape[1]
    )

    # ── Paint blush on both cheeks ────────────────────────────────────────────
    result = _draw_cheek(result, l_cx, l_cy, rx, ry, _BLUSH_BGR, alpha)
    result = _draw_cheek(result, r_cx, r_cy, rx, ry, _BLUSH_BGR, alpha)

    # ── Subtle warmth boost in LAB to tie blush into skin tone ───────────────
    # Nudge the A channel (green↔red axis) very slightly toward red/warm.
    # This stops the pink from looking like a sticker on grey skin.
    lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float32)
    l, a, b = cv2.split(lab)
    a = np.clip(a + 2.5, 0, 255)          # +2.5 on A = barely perceptible warmth
    lab_out = cv2.merge([l, a, b]).astype(np.uint8)
    result  = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)

    return result