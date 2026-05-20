from filters.smooth_skin import apply_smooth_skin
from filters.blush import apply_blush
from filters.cat_ears import apply_cat_ears
from filters.hearts import apply_hearts
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import cv2
import numpy as np

from filters.vhs import apply_vhs
from filters.glitch import apply_glitch
from filters.y2k import apply_y2k
from filters.crt import apply_crt
from filters.grain import apply_grain
from filters.chroma import apply_chroma

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class FilterRequest(BaseModel):
    image: str
    filter: str

def decode_image(b64_string: str):
    image_data = base64.b64decode(b64_string)
    image_array = np.frombuffer(image_data, np.uint8)
    return cv2.imdecode(image_array, cv2.IMREAD_COLOR)

def encode_image(img) -> str:
    _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer).decode('utf-8')

@app.post("/apply-filter")
async def apply_filter(request: FilterRequest):
    img = decode_image(request.image)

    # Resize large frames before processing — speeds everything up
    h, w = img.shape[:2]
    if w > 640:
        scale = 640 / w
        img = cv2.resize(img, (640, int(h * scale)))

    if request.filter == "vhs":
        img = apply_vhs(img)
    elif request.filter == "glitch":
        img = apply_glitch(img)
    elif request.filter == "y2k":
        img = apply_y2k(img)
    elif request.filter == "crt":
        img = apply_crt(img)
    elif request.filter == "grain":
        img = apply_grain(img)
    elif request.filter == "chroma":
        img = apply_chroma(img)
    elif request.filter == "smooth_skin":
        img = apply_smooth_skin(img, intensity=0.6)
    elif request.filter == "blush":
        img = apply_blush(img, alpha=0.38)
    elif request.filter == "cat_ears":
        img = apply_cat_ears(img)
    elif request.filter == "hearts":
        img = apply_hearts(img)

    return {"image": encode_image(img)}

@app.get("/")
async def root():
    return {"status": "Y2K Photobooth API is running!"}