import base64
import binascii
import logging

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from config import (
    ALLOWED_FILTERS,
    CORS_ORIGINS,
    ENVIRONMENT,
    FULL_MAX_WIDTH,
    JPEG_QUALITY_FULL,
    JPEG_QUALITY_PREVIEW,
    MAX_IMAGE_BYTES,
    MAX_IMAGE_DIMENSION,
    PREVIEW_MAX_WIDTH,
    RATE_LIMIT,
)
from filters.blush import apply_blush
from filters.cat_ears import apply_cat_ears
from filters.hearts import apply_hearts
from filters.heatmap import apply_heatmap
from filters.pixel import apply_pixel
from filters.star_face import apply_star_face

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Whee Photobooth API")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://wheephotobooth.site",
        "https://www.wheephotobooth.site", 
        "https://whee-photobooth.vercel.app",
        "http://localhost:5173",
        "http://localhost:4173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

_MAX_B64_CHARS = (MAX_IMAGE_BYTES * 4 // 3) + 16


class FilterRequest(BaseModel):
    image: str
    filter: str
    preview: bool = False

    @field_validator("image")
    @classmethod
    def validate_image(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("image is required")
        cleaned = value.strip()
        if cleaned.startswith("data:"):
            _, _, cleaned = cleaned.partition(",")
        if len(cleaned) > _MAX_B64_CHARS:
            raise ValueError(f"image exceeds max size ({MAX_IMAGE_BYTES} bytes decoded)")
        try:
            decoded = base64.b64decode(cleaned, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("image must be valid base64") from exc
        if len(decoded) > MAX_IMAGE_BYTES:
            raise ValueError(f"image exceeds max size ({MAX_IMAGE_BYTES} bytes)")
        if len(decoded) < 100:
            raise ValueError("image payload is too small to be a valid JPEG")
        return cleaned

    @field_validator("filter")
    @classmethod
    def validate_filter(cls, value: str) -> str:
        if value not in ALLOWED_FILTERS:
            allowed = ", ".join(sorted(ALLOWED_FILTERS))
            raise ValueError(f"filter must be one of: {allowed}")
        return value


def decode_image(b64_string: str) -> np.ndarray:
    image_data = base64.b64decode(b64_string)
    image_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image as JPEG")
    return img


def resize_to_max_width(img: np.ndarray, max_width: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= max_width:
        return img
    scale = max_width / w
    return cv2.resize(img, (max_width, int(h * scale)))


def encode_image(img: np.ndarray, *, preview: bool) -> str:
    quality = JPEG_QUALITY_PREVIEW if preview else JPEG_QUALITY_FULL
    ok, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode image")
    return base64.b64encode(buffer).decode("utf-8")


def apply_filter_to_image(img: np.ndarray, request: FilterRequest) -> np.ndarray:
    f = request.filter
    if f == "blush":
        return apply_blush(img, opacity=0.70 if request.preview else 0.85)
    if f == "cat_ears":
        return apply_cat_ears(img)
    if f == "hearts":
        return apply_hearts(img)
    if f == "star_face":
        return apply_star_face(img)
    if f == "pixel":
        return apply_pixel(img)
    if f == "heatmap":
        return apply_heatmap(img)
    raise HTTPException(status_code=400, detail=f"Unknown filter: {f}")


@app.get("/")
def root():
    return {"status": "Whee Photobooth API is running!"}


@app.get("/health")
def health():
    return {"status": "ok", "environment": ENVIRONMENT}


@app.on_event("startup")
async def startup_event():
    logger.info("Whee Photobooth API started")
    logger.info("Environment: %s", ENVIRONMENT)
    logger.info("CORS origins: %s", CORS_ORIGINS)
    logger.info("Allowed filters: %s", ALLOWED_FILTERS)


@app.options("/apply-filter")
def apply_filter_preflight():
    return Response(status_code=200)


@app.post("/apply-filter")
@limiter.limit(RATE_LIMIT)
def apply_filter(request: Request, body: FilterRequest):
    img = decode_image(body.image)
    h, w = img.shape[:2]
    if w > MAX_IMAGE_DIMENSION or h > MAX_IMAGE_DIMENSION:
        raise HTTPException(
            status_code=400,
            detail=f"Image dimensions exceed max {MAX_IMAGE_DIMENSION}px",
        )
    max_width = PREVIEW_MAX_WIDTH if body.preview else FULL_MAX_WIDTH
    img = resize_to_max_width(img, max_width)
    try:
        img = apply_filter_to_image(img, body)
    except HTTPException:
        raise
    except Exception:
        logger.exception("Filter processing failed: filter=%s", body.filter)
        raise HTTPException(status_code=500, detail="Filter processing failed") from None
    return {"image": encode_image(img, preview=body.preview)}