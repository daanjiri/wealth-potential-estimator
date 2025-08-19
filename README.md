# Wealth Potential Estimator API

A FastAPI application that predicts wealth potential from facial images using machine learning. The system uses FaceNet for facial embedding extraction and K-Nearest Neighbors (KNN) regression for wealth estimation based on facial features.

## How the Solution was Built

The wealth prediction system was developed using a combination of deep learning and traditional machine learning techniques to analyze facial features and estimate wealth potential. The solution leverages a pre-trained FaceNet model to extract high-dimensional facial embeddings that capture distinctive facial characteristics, which are then stored in a vector database for efficient similarity searching. A K-Nearest Neighbors regression algorithm was implemented to predict wealth by finding the most similar faces in the training dataset and computing weighted averages based on cosine similarity scores, enabling accurate wealth estimation from facial images alone.

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

### POST `/predict-wealth`
Predict wealth potential from facial images using machine learning.

This endpoint uses the **FaceNet model** for extracting facial embeddings and **K-Nearest Neighbors (KNN) regression** for wealth estimation. The system:

1. **Face Detection & Embedding**: Uses FaceNet to extract 512-dimensional facial embeddings that capture facial features
2. **Similarity Search**: Finds the most similar faces in the training dataset using cosine similarity
3. **Wealth Prediction**: Applies KNN regression with weighted averaging based on similarity scores

- **Content-Type**: `multipart/form-data`
- **Parameters**: 
  - `file` (required): Image file containing a face
  - `k` (optional): Number of neighbors for KNN (default: 3)
- **Supported formats**: JPG, JPEG, PNG, GIF, BMP, WEBP

#### Example using curl:
```bash
curl -X POST "http://localhost:8000/predict-wealth" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@/path/to/your/selfie.jpg" \
     -F "k=5"
```

#### Example using Python requests:
```python
import requests

url = "http://localhost:8000/predict-wealth"
files = {"file": open("path/to/your/selfie.jpg", "rb")}
data = {"k": 5}  # Optional: number of neighbors
response = requests.post(url, files=files, data=data)
print(response.json())
```

#### Response format:
```json
{
    "message": "Wealth prediction successful",
    "original_filename": "selfie.jpg",
    "saved_filename": "uuid-generated-name.jpg",
    "file_size_bytes": 245760,
    "prediction": {
        "predicted_wealth": 75000.50,
        "confidence": 0.85,
        "similar_faces": [
            {
                "distance": 0.23,
                "wealth": 80000.0,
                "image_path": "dataset/selfie5.png"
            }
        ]
    },
    "timestamp": "2024-01-15T10:30:45.123456"
}
```

### GET `/model-status`
Check the status of the FaceNet model and system readiness.

## Features

- ✅ **Wealth Prediction**: ML-based wealth estimation from facial images
- ✅ **FaceNet Integration**: State-of-the-art facial embedding extraction
- ✅ **KNN Regression**: Weighted K-Nearest Neighbors for accurate predictions
- ✅ **Vector Database**: Efficient similarity search and storage
- ✅ Image file validation and processing
- ✅ Unique filename generation (UUID)
- ✅ File size tracking and metadata
- ✅ Comprehensive error handling
- ✅ Automatic uploads directory creation
- ✅ Support for multiple image formats (JPG, PNG, GIF, BMP, WEBP)
- ✅ Model status monitoring and health checks
