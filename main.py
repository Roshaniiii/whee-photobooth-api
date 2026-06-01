from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import base64
import cv2
import numpy as np

from filters.blush    import apply_blush
from filters.cat_ears import apply_cat_ears
from filters.hearts   import apply_hearts
from filters.star_face import apply_star_face
from filters.pixel import apply_pixel
from filters.heatmap import apply_heatmap
from filters.background_replace import apply_background_replace

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class FilterRequest(BaseModel):
    image:   str
    filter:  str
    preview: bool = False
    bg_filename: Optional[str] = "bg1.jpg"

def decode_image(b64_string: str) -> np.ndarray:
    image_data  = base64.b64decode(b64_string)
    image_array = np.frombuffer(image_data, np.uint8)
    return cv2.imdecode(image_array, cv2.IMREAD_COLOR)

def encode_image(img: np.ndarray) -> str:
    _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer).decode('utf-8')

@app.post("/apply-filter")
async def apply_filter(request: FilterRequest):
    img = decode_image(request.image)

    # Resize for performance
    h, w = img.shape[:2]
    if w > 640:
        scale = 640 / w
        img   = cv2.resize(img, (640, int(h * scale)))

    f = request.filter

    if f == "blush":
        img = apply_blush(img, opacity=0.70 if request.preview else 0.85)
    elif f == "cat_ears":
        img = apply_cat_ears(img)
    elif f == "hearts":
        img = apply_hearts(img)
    elif f == "star_face":
        img = apply_star_face(img)
    elif f == "pixel":
        img = apply_pixel(img)
    elif f == "heatmap":
        img = apply_heatmap(img)
    elif f == "bg_replace":
        img = apply_background_replace(
            img,
            bg_filename=request.bg_filename or "bg1.jpg",
        )

    return {"image": encode_image(img)}

@app.get("/")
async def root():
    return {"status": "Y2K Photobooth API is running!"}