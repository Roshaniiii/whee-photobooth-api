import cv2
import numpy as np


def apply_heatmap(img: np.ndarray) -> np.ndarray:
    """
    Thermal Heatmap filter — simulates an infrared thermal camera.

    Matches the reference image:
      • Dark/cool areas  → deep blue / purple
      • Mid-areas        → cyan → green → yellow
      • Bright/warm areas → orange → red → white-yellow

    Pipeline:
      1. Convert to grayscale (luminance = heat proxy)
      2. Apply COLORMAP_JET  — OpenCV's built-in thermal palette
         (blue=cold → cyan → green → yellow → red=hot)
      3. Boost saturation so colours are vivid like the reference
      4. Slight blur before mapping to smooth thermal gradients

    Args:
        img : BGR uint8 image (H x W x 3)

    Returns:
        BGR uint8 heatmap image, same shape as input.
    """
    # ── 1. Smooth first — thermal cameras don't have sharp edges ─────────────
    smoothed: np.ndarray = cv2.GaussianBlur(img, (5, 5), sigmaX=1.5)

    # ── 2. Grayscale — luminance acts as heat proxy ───────────────────────────
    gray: np.ndarray = cv2.cvtColor(smoothed, cv2.COLOR_BGR2GRAY)

    # ── 3. Enhance contrast so full colour range is used ─────────────────────
    # CLAHE (Contrast Limited Adaptive Histogram Equalization) — spreads
    # brightness values across the full 0-255 range locally, so even
    # low-contrast areas get the full thermal colour spread.
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray_eq: np.ndarray = clahe.apply(gray)

    # ── 4. Apply thermal color map ───────────────────────────────────────────
    # COLORMAP_JET matches the reference exactly:
    #   0   (darkest) → blue/purple
    #   64            → cyan
    #   128           → green/yellow
    #   192           → orange
    #   255 (brightest)→ red/yellow-white
    heatmap: np.ndarray = cv2.applyColorMap(gray_eq, cv2.COLORMAP_JET)

    # ── 5. Saturation boost — make colours more vivid like reference ──────────
    hsv: np.ndarray = cv2.cvtColor(heatmap, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.25, 0, 255)   # +25% saturation
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.05, 0, 255)   # +5% brightness
    result: np.ndarray = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return result