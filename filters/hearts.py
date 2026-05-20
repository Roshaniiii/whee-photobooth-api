import time
import math
import cv2
import numpy as np


# ── Heart colours (BGR) ───────────────────────────────────────────────────────
_HEART_COLOURS = [
    (157, 107, 255),   # Hot pink   #FF6B9D
    (87,  71,  255),   # Soft red   #FF4757
    (198, 184, 255),   # Rose white #FFB8C6
    (180, 105, 255),   # Deep pink  #FF69B4
]

# ── Per-heart configuration (4 hearts) ───────────────────────────────────────
# Each entry: (x_offset_ratio, speed, size_ratio, phase_offset, drift_freq)
#   x_offset_ratio : horizontal position as fraction of face width (-0.5 to 0.5)
#   speed          : rise speed multiplier
#   size_ratio     : heart size as fraction of face width
#   phase_offset   : time offset so hearts are staggered vertically (seconds)
#   drift_freq     : frequency of horizontal sine drift
_HEART_CONFIG = [
    (-0.30, 1.00, 0.10, 0.00, 1.1),   # left,       normal speed, small
    ( 0.30, 1.20, 0.12, 1.50, 0.9),   # right,      faster,       medium
    (-0.05, 0.85, 0.14, 0.75, 1.3),   # centre-left, slower,      large
    ( 0.18, 1.10, 0.09, 2.50, 1.0),   # centre-right, medium,     small
]

# How far above the head the hearts travel before resetting (as multiple of fh)
_RISE_HEIGHT_RATIO = 1.4


def _draw_heart(img: np.ndarray, cx: int, cy: int,
                size: int, colour_bgr: tuple, alpha: float) -> np.ndarray:
    """
    Draw a filled heart centred at (cx, cy) onto img using a
    parametric Bézier-based polygon approach.

    The heart shape is built from the classic parametric equations:
        x(t) = 16 sin³(t)
        y(t) = 13cos(t) − 5cos(2t) − 2cos(3t) − cos(4t)
    scaled to `size` pixels.

    Alpha blending is done via a float32 overlay so hearts fade
    naturally as they rise.

    Args:
        img        : BGR uint8 image
        cx, cy     : heart centre pixel position
        size       : radius scale in pixels
        colour_bgr : (B, G, R) tuple
        alpha      : opacity 0.0–1.0

    Returns:
        BGR uint8 image with heart composited.
    """
    if alpha <= 0.01 or size < 3:
        return img

    # Build heart polygon via parametric equations
    t_vals  = np.linspace(0, 2 * math.pi, 120)
    scale   = size / 17.0          # normalise so size≈px radius

    xs = 16 * np.sin(t_vals) ** 3
    ys = -(13 * np.cos(t_vals)
           - 5  * np.cos(2 * t_vals)
           - 2  * np.cos(3 * t_vals)
           -      np.cos(4 * t_vals))

    pts = np.column_stack([
        (xs * scale + cx).astype(np.int32),
        (ys * scale + cy).astype(np.int32),
    ]).reshape((-1, 1, 2))

    # Bounding box for the overlay region (clamped to image)
    fh, fw = img.shape[:2]
    x_min  = max(int(pts[:, 0, 0].min()) - 2, 0)
    x_max  = min(int(pts[:, 0, 0].max()) + 2, fw)
    y_min  = max(int(pts[:, 0, 1].min()) - 2, 0)
    y_max  = min(int(pts[:, 0, 1].max()) + 2, fh)

    if x_max <= x_min or y_max <= y_min:
        return img

    result  = img.copy()
    overlay = result.copy()

    # Filled heart
    cv2.fillPoly(overlay, [pts], colour_bgr)

    # Soft inner highlight — lighter ellipse in upper-left of heart
    hx = cx - size // 5
    hy = cy - size // 4
    highlight = tuple(min(int(c * 1.4), 255) for c in colour_bgr)
    cv2.ellipse(overlay, (hx, hy),
                (max(size // 6, 2), max(size // 8, 1)),
                -30, 0, 360, highlight, -1)

    # Alpha blend only the bounding-box region for efficiency
    roi_r = result[y_min:y_max, x_min:x_max].astype(np.float32)
    roi_o = overlay[y_min:y_max, x_min:x_max].astype(np.float32)
    blended = roi_r * (1.0 - alpha) + roi_o * alpha
    result[y_min:y_max, x_min:x_max] = np.clip(blended, 0, 255).astype(np.uint8)

    return result


def apply_hearts(img: np.ndarray) -> np.ndarray:
    """
    Hearts Floating Filter.

    Detects the largest face and animates 4 hearts rising above the head.
    Each heart has an independent phase offset, speed, size, horizontal
    position, and drift frequency — driven by time.time() so every frame
    from the live camera stream shows a slightly different position,
    producing smooth floating animation at the frontend.

    Heart lifecycle per frame
    ─────────────────────────
    • cycle_duration  : how long (seconds) one heart takes to fully rise
    • progress ∈ [0,1]: where in the cycle this heart currently is
        0   → just above the head
        1   → fully risen (_RISE_HEIGHT_RATIO × fh above head), then resets
    • y position      : linear rise  +  small sine horizontal drift
    • alpha           : 1.0 at bottom → 0.0 at top (fade out while rising)
    • size            : shrinks slightly as heart rises (perspective feel)

    Falls back gracefully — returns original frame if no face detected.

    Args:
        img: BGR uint8 image.

    Returns:
        BGR uint8 image with animated hearts composited above the face.
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
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    if len(faces) == 0:
        return img.copy()

    # Largest face
    fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])

    # Face centre-x and top-y (anchor for all hearts)
    face_cx  = fx + fw // 2
    face_top = fy                              # y=0 of the rise column

    # Total vertical travel distance in pixels
    rise_px  = int(fh * _RISE_HEIGHT_RATIO)

    # Cycle duration: one full rise takes ~2.5 s at normal speed
    base_cycle = 2.5
    now        = time.time()

    result = img.copy()

    for i, (x_ratio, speed, size_ratio, phase, drift_freq) in enumerate(_HEART_CONFIG):
        colour = _HEART_COLOURS[i % len(_HEART_COLOURS)]

        # ── Progress through rise cycle ───────────────────────────────────────
        cycle_duration = base_cycle / speed
        # Phase offset staggers each heart so they're never all at the same y
        t_offset  = (now + phase) % cycle_duration
        progress  = t_offset / cycle_duration   # 0.0 → 1.0

        # ── Pixel position ────────────────────────────────────────────────────
        # Base x: face centre + proportional offset
        base_x   = face_cx + int(x_ratio * fw)
        # Horizontal drift: gentle sine sway, amplitude = 8 % of face width
        drift_amp = int(fw * 0.08)
        drift_x   = int(drift_amp * math.sin(2 * math.pi * drift_freq * now + phase))
        cx        = base_x + drift_x

        # Y rises from face_top downward (lower y = higher on screen)
        cy = face_top - int(progress * rise_px)

        # ── Alpha: fully opaque at bottom, transparent at top ─────────────────
        # Ease-in: hearts start fading after 60 % of the rise
        alpha = max(0.0, 1.0 - max(0.0, (progress - 0.35) / 0.65))
        alpha = float(np.clip(alpha, 0.0, 1.0))

        # ── Size: starts at size_ratio × fw, shrinks 30 % by the top ─────────
        base_size = int(size_ratio * fw)
        size      = max(4, int(base_size * (1.0 - 0.30 * progress)))

        # ── Draw ──────────────────────────────────────────────────────────────
        result = _draw_heart(result, cx, cy, size, colour, alpha)

    return result