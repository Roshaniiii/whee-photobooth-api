import cv2
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
_BLOCK_SIZE:      int   = 2      # pixel block size in px — change to 16/32 for bigger blocks
_SCANLINE_ALPHA:  float = 0.28   # scanline darkness 0.0=invisible 1.0=full black bars
_SCANLINE_GAP:    int   = 2     # draw a dark bar every N rows (2 = every other row)
_CONTRAST_BOOST:  float = 1.08   # slight contrast lift to compensate scanline darkening
_SATURATION_BOOST:float = 1.12   # slight colour pop — CRT screens were vivid


def apply_pixel(img: np.ndarray) -> np.ndarray:
    """
    Pixel Art + Scanlines filter (old CRT game screen aesthetic).

    Pipeline:
      1. Pixelate  — shrink to block grid, scale back with INTER_NEAREST
                     so every block is a flat colour with hard edges.
      2. Saturation boost — CRT screens had punchy vivid colours.
      3. Contrast boost   — compensates for scanline darkening.
      4. Scanlines — draw semi-transparent horizontal dark bars every
                     _SCANLINE_GAP rows to mimic a CRT phosphor screen.

    Args:
        img : BGR uint8 image (H x W x 3)

    Returns:
        BGR uint8 image with pixel + scanline effect applied.
    """
    h: int = img.shape[0]
    w: int = img.shape[1]

    # ── 1. Pixelate ───────────────────────────────────────────────────────────
    block: int = max(2, _BLOCK_SIZE)

    # Shrink — each block becomes one pixel
    small: np.ndarray = cv2.resize(
        img,
        (max(1, w // block), max(1, h // block)),
        interpolation=cv2.INTER_LINEAR,   # average colours within block
    )
    # Scale back — INTER_NEAREST copies pixel exactly, no blending → hard edges
    pixelated: np.ndarray = cv2.resize(
        small,
        (w, h),
        interpolation=cv2.INTER_NEAREST,
    )

    # ── 2. Saturation + contrast boost ───────────────────────────────────────
    hsv: np.ndarray = cv2.cvtColor(pixelated, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * _SATURATION_BOOST, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * _CONTRAST_BOOST,   0, 255)
    result: np.ndarray = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # ── 3. Scanlines ──────────────────────────────────────────────────────────
    # Build a scanline mask: 1.0 everywhere except dark bars on every Nth row
    scanline_mask: np.ndarray = np.ones((h, w), dtype=np.float32)

    # Every _SCANLINE_GAP rows, draw a bar that is _SCANLINE_ALPHA darker
    # Bar height = 1 px for subtle effect at 8px block size
    for y in range(0, h, _SCANLINE_GAP):
        scanline_mask[y, :] = 1.0 - _SCANLINE_ALPHA

    # Apply mask: multiply each channel by the scanline pattern
    scanline_mask3: np.ndarray = scanline_mask[:, :, np.newaxis]   # H×W×1
    result = np.clip(
        result.astype(np.float32) * scanline_mask3, 0, 255
    ).astype(np.uint8)

    # ── 4. Subtle green phosphor tint — optional CRT colour cast ─────────────
    # Adds a very faint green-blue warmth typical of old CRT monitors.
    # Comment out these 4 lines if you want a neutral (no tint) look.
    tint: np.ndarray = np.zeros_like(result, dtype=np.float32)
    tint[:, :] = (2, 4, 0)          # tiny BGR boost: +2 blue, +4 green, +0 red
    result = np.clip(
        result.astype(np.float32) + tint, 0, 255
    ).astype(np.uint8)

    return result