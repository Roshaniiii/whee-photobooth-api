import os

from dotenv import load_dotenv

load_dotenv()

ALLOWED_FILTERS: frozenset[str] = frozenset({
    "blush",
    "cat_ears",
    "hearts",
    "star_face",
    "pixel",
    "heatmap",
})

# Max decoded JPEG size (~5 MB default)
MAX_IMAGE_BYTES: int = int(os.getenv("MAX_IMAGE_BYTES", str(5 * 1024 * 1024)))

# Max width after decode (reject oversized dimensions before processing)
MAX_IMAGE_DIMENSION: int = int(os.getenv("MAX_IMAGE_DIMENSION", "4096"))

# Output resize caps
PREVIEW_MAX_WIDTH: int = int(os.getenv("PREVIEW_MAX_WIDTH", "640"))
FULL_MAX_WIDTH: int = int(os.getenv("FULL_MAX_WIDTH", "1280"))

JPEG_QUALITY_PREVIEW: int = int(os.getenv("JPEG_QUALITY_PREVIEW", "75"))
JPEG_QUALITY_FULL: int = int(os.getenv("JPEG_QUALITY_FULL", "90"))

# Comma-separated origins, e.g. https://whee.example.com,http://localhost:5173
_cors_raw: str = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:5173,http://localhost:4173,https://whee-photobooth.vercel.app",
)
CORS_ORIGINS: list[str] = [o.strip() for o in _cors_raw.split(",") if o.strip()]

# slowapi limit string, e.g. "300/minute"
RATE_LIMIT: str = os.getenv("RATE_LIMIT", "300/minute")

ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
