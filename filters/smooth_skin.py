import cv2
import numpy as np


def apply_smooth_skin(img: np.ndarray, intensity: float = 0.6) -> np.ndarray:
    """
    Smooth Skin Filter — Medium strength by default (intensity=0.6).

    Pipeline:
      1. Convert to LAB → work only on the L (luminance) channel
         so skin tones / colors are never washed out.
      2. Multi-pass bilateral filter → blurs texture while keeping
         hard edges (eyes, lips, hair lines).
      3. High-frequency detail layer → subtract blurred from original
         to isolate fine details (pores, wrinkles).
      4. Blend: smoothed L + (1 - intensity) * detail layer
         so features are subtracted back in proportion to strength.
      5. Recombine with A/B channels → convert back to BGR.
      6. Final unsharp mask over the full image to restore global crispness.

    Args:
        img:       BGR image (numpy array, uint8)
        intensity: 0.0 = no effect, 1.0 = maximum smoothing
                   0.6 is the recommended medium default.

    Returns:
        Processed BGR image (same shape/dtype as input).
    """
    intensity = float(np.clip(intensity, 0.0, 1.0))

    # ── 1. LAB split ──────────────────────────────────────────────────────────
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    l_float = l_ch.astype(np.float32)

    # ── 2. Multi-pass bilateral filter on L channel ───────────────────────────
    # Two passes: first broad smoothing, second tight edge-preserving cleanup.
    # d=9  → neighbourhood diameter (pixel reach)
    # sigmaColor → how much color difference is tolerated (higher = more blur)
    # sigmaSpace → how much spatial distance matters
    smooth = cv2.bilateralFilter(l_ch, d=9,  sigmaColor=75, sigmaSpace=75)
    smooth = cv2.bilateralFilter(smooth, d=7, sigmaColor=55, sigmaSpace=55)
    smooth_float = smooth.astype(np.float32)

    # ── 3. High-frequency detail layer ────────────────────────────────────────
    # detail > 0 means the original had MORE texture than the smoothed version.
    # We'll add some of it back to avoid the plastic / over-processed look.
    detail = l_float - smooth_float  # range roughly [-128, +128]

    # ── 4. Blend ──────────────────────────────────────────────────────────────
    # At intensity=0.6:
    #   blended = smooth + 0.4 * detail  → 60 % of texture removed
    blended = smooth_float + (1.0 - intensity) * detail
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    # ── 5. Recombine LAB → BGR ────────────────────────────────────────────────
    lab_out = cv2.merge([blended, a_ch, b_ch])
    result = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)

    # ── 6. Global unsharp mask — brings back edge crispness ───────────────────
    # Strength is intentionally mild so it doesn't fight the smoothing.
    gaussian = cv2.GaussianBlur(result, (0, 0), sigmaX=2.0)
    result = cv2.addWeighted(result, 1.3, gaussian, -0.3, 0)
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result