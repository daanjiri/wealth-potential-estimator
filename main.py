from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import os
import shutil
from pathlib import Path
import uuid
from datetime import datetime
import logging
from PIL import Image
import numpy as np
from contextlib import asynccontextmanager
from typing import Optional

from model_manager import get_model_manager, initialize_model
from vector_db import get_vector_db, initialize_database

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle.
    Load model on startup, cleanup on shutdown.
    """
    # Startup
    logger.info("Starting application...")
    logger.info("Loading FaceNet model...")
    
    success = initialize_model()
    if success:
        logger.info("Model loaded successfully!")
        
        # Initialize vector database and load dataset
        logger.info("Initializing vector database...")
        db_success = initialize_database()
        if db_success:
            logger.info("Database initialized and dataset loaded successfully!")
            db = get_vector_db()
            stats = db.get_stats()
            logger.info(f"Database stats: {stats}")
        else:
            logger.warning("Failed to initialize database or load dataset")
    else:
        logger.warning("Failed to load model. Embeddings extraction will not be available.")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")


app = FastAPI(
    title="Wealth Potential Estimator API",
    description="API for uploading, processing images, and extracting facial embeddings",
    version="2.0.0",
    lifespan=lifespan
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

def is_valid_image(filename: str) -> bool:
    """Check if the uploaded file has a valid image extension."""
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/model-status")
async def model_status():
    """
    Check the status of the FaceNet model.
    
    Returns:
        JSON response with model status information
    """
    model_manager = get_model_manager()
    
    return JSONResponse(
        status_code=200,
        content={
            "model_loaded": model_manager.is_loaded,
            "device": model_manager.device,
            "timestamp": datetime.now().isoformat()
        }
    )


@app.post("/predict-wealth")
async def predict_wealth(file: UploadFile = File(...), k: int = 3):
    """
    Predict wealth using KNN on facial embeddings.
    
    Args:
        file: The uploaded image file (multipart/form-data)
        k: Number of neighbors for KNN (default: 3)
    
    Returns:
        JSON response with wealth prediction and similar faces
    """
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")
        
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    if not is_valid_image(file.filename):
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    model_manager = get_model_manager()
    vector_db = get_vector_db()
    
    if not model_manager.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please try again later."
        )
    
    try:
        file_extension = Path(file.filename).suffix.lower()
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = UPLOAD_DIR / unique_filename
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        image = Image.open(file_path)
        
        if image.mode == 'RGBA':
            rgb_image = Image.new('RGB', image.size, (255, 255, 255))
            rgb_image.paste(image, mask=image.split()[3])
            image = rgb_image
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        embeddings = model_manager.extract_embeddings(image)
        
        if embeddings is None:
            raise HTTPException(
                status_code=500,
                detail="Failed to extract embeddings from image"
            )
        
        prediction = vector_db.predict_wealth_knn(embeddings, k=k, weighted=True)
        
        file_size = os.path.getsize(file_path)
        
        return JSONResponse(
            status_code=200,
            content={
                "message": "Wealth prediction successful",
                "original_filename": file.filename,
                "saved_filename": unique_filename,
                "file_size_bytes": file_size,
                "prediction": prediction,
                "timestamp": datetime.now().isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"Error predicting wealth: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )





if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# source venv/bin/activate
# uvicorn main:app --reload --host 0.0.0.0 --port 8000