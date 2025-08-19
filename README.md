# Wealth Potential Estimator API

A FastAPI application that receives images via multipart/form-data and saves them locally.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python main.py
```

Or using uvicorn directly:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## API Endpoints

### GET `/`
Returns basic API information.

### GET `/health`
Health check endpoint.

### POST `/upload-image`
Upload an image file.

- **Content-Type**: `multipart/form-data`
- **Parameter**: `file` (image file)
- **Supported formats**: JPG, JPEG, PNG, GIF, BMP, WEBP

#### Example using curl:
```bash
curl -X POST "http://localhost:8000/upload-image" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@path/to/your/image.jpg"
```

#### Example using Python requests:
```python
import requests

url = "http://localhost:8000/upload-image"
files = {"file": open("path/to/your/image.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

## Features

- ✅ Image file validation
- ✅ Unique filename generation (UUID)
- ✅ File size tracking
- ✅ Proper error handling
- ✅ Automatic uploads directory creation
- ✅ Support for multiple image formats

## API Documentation

Once the server is running, you can access:
- Interactive API docs: http://localhost:8000/docs
- Alternative docs: http://localhost:8000/redoc
