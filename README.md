# Whee Photobooth API 

🔗 **Live site:** [wheephotobooth.site](https://wheephotobooth.site)

FastAPI backend for Whee Photobooth. Applies image filters using OpenCV and returns processed frames to the frontend.

**Frontend repo:** https://github.com/Roshaniiii/whee-photobooth

## Filters
| Filter ID | Effect |
|---|---|
| blush | Pink blush marks on cheeks |
| cat_ears | Cat ears overlay |
| hearts | Floating hearts |
| star_face | Star decorations |
| pixel | Pixelate effect |
| heatmap | Heatmap colour effect |

## Local Development

### Requirements
- Python 3.11+
- pip

### Setup
```bash
git clone https://github.com/YOURUSERNAME/whee-photobooth-api
cd whee-photobooth-api
python -m venv .venv
.venv\Scripts\activate    # Windows
pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --reload --port 8000
```

### Environment Variables
| Variable | Default | Description |
|---|---|---|
| CORS_ORIGINS | http://localhost:5173 | Comma separated frontend URLs |
| RATE_LIMIT | 300/minute | Per IP rate limit |
| ENVIRONMENT | development | development or production |
| MAX_IMAGE_BYTES | 5242880 | Max image size (5MB) |
| MAX_IMAGE_DIMENSION | 4096 | Max image dimension in px |

### Example .env
ENVIRONMENT=development
CORS_ORIGINS=http://localhost:5173
RATE_LIMIT=300/minute

## API Endpoints

### GET /
Health check
{ "status": "Whee Photobooth API is running!" }

### GET /health
{ "status": "ok", "environment": "development" }

### POST /apply-filter
Apply a filter to a base64 image.

**Request:**
```json
{
  "image": "<base64 string>",
  "filter": "blush",
  "preview": true
}
```

**Response:**
```json
{
  "image": "<base64 string>"
}
```

## Production Deployment (Render)

1. Connect GitHub repo to Render
2. Set build command: `pip install -r requirements.txt`
3. Set start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Add environment variables:

ENVIRONMENT=production
CORS_ORIGINS=https://yourdomain.com

## Project Structure
whee-photobooth-api/
├── main.py          # FastAPI app
├── config.py        # Settings
├── requirements.txt
├── .env.example
├── filters/
│   ├── blush.py
│   ├── cat_ears.py
│   ├── hearts.py
│   ├── heatmap.py
│   ├── pixel.py
│   └── star_face.py
└── assets/
└── (filter asset PNGs)
