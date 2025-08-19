"""
Vector database manager using Chroma for facial embeddings and wealth prediction.
Handles KNN search for wealth estimation based on facial similarity.
"""

import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import chromadb
from chromadb.config import Settings
from PIL import Image

from model_manager import get_model_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_PATH = Path(os.getenv("CHROMA_DB_PATH", "./chroma_db"))
COLLECTION_NAME = "wealth_faces"


class WealthFaceDB:
    """
    Vector database for facial embeddings and wealth labels.
    Supports KNN search for wealth prediction.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the vector database.
        
        Args:
            db_path: Path to store the Chroma database
        """
        self.db_path = Path(db_path) if db_path else DB_PATH
        self.db_path.mkdir(exist_ok=True, parents=True)
        
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"description": "Facial embeddings for wealth prediction"}
        )
        
        logger.info(f"Initialized vector DB at: {self.db_path}")
        logger.info(f"Collection '{COLLECTION_NAME}' has {self.collection.count()} entries")
    
    def add_face_embedding(
        self, 
        embedding: np.ndarray, 
        file_name: str, 
        label: float,
        image_path: Optional[str] = None
    ) -> bool:
        """
        Add a facial embedding with metadata to the database.
        
        Args:
            embedding: 512-dimensional facial embedding
            file_name: Name of the image file
            label: Wealth value/label
            image_path: Full path to the image
            
        Returns:
            bool: True if successful
        """
        try:
            if embedding.shape != (512,):
                raise ValueError(f"Expected 512D embedding, got {embedding.shape}")
            
            metadata = {
                "file_name": file_name,
                "label": float(label),
                "image_path": image_path or f"dataset/{file_name}"
            }
            
            self.collection.add(
                embeddings=[embedding.tolist()],
                metadatas=[metadata],
                ids=[file_name.replace('.png', '').replace('.jpg', '')]
            )
            
            logger.info(f"Added {file_name} with label {label}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add embedding for {file_name}: {e}")
            return False
    
    def search_similar_faces(
        self, 
        query_embedding: np.ndarray, 
        k: int = 3
    ) -> Dict:
        """
        Find K most similar faces using cosine similarity.
        
        Args:
            query_embedding: 512D facial embedding to search for
            k: Number of nearest neighbors to return
            
        Returns:
            Dictionary with similar faces and their metadata
        """
        try:
            if query_embedding.shape != (512,):
                raise ValueError(f"Expected 512D embedding, got {query_embedding.shape}")
            
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=k,
                include=["metadatas", "distances", "embeddings"]
            )
            
            similar_faces = []
            if results['metadatas'][0]:
                for i in range(len(results['metadatas'][0])):
                    similar_faces.append({
                        "file_name": results['metadatas'][0][i]['file_name'],
                        "label": results['metadatas'][0][i]['label'],
                        "image_path": results['metadatas'][0][i]['image_path'],
                        "distance": results['distances'][0][i],
                        "similarity": 1 - results['distances'][0][i]
                    })
            
            return {
                "query_results": similar_faces,
                "count": len(similar_faces)
            }
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return {"query_results": [], "count": 0}
    
    def predict_wealth_knn(
        self, 
        query_embedding: np.ndarray, 
        k: int = 3,
        weighted: bool = True
    ) -> Dict:
        """
        Predict wealth using KNN regression on facial similarity.
        
        Args:
            query_embedding: 512D facial embedding
            k: Number of neighbors for prediction
            weighted: Whether to weight by similarity
            
        Returns:
            Prediction results with details
        """
        try:
            similar_faces = self.search_similar_faces(query_embedding, k)
            
            if similar_faces['count'] == 0:
                return {
                    "predicted_wealth": 0,
                    "method": f"KNN-{k}",
                    "neighbors": [],
                    "error": "No similar faces found"
                }
            
            neighbors = similar_faces['query_results']
            
            if weighted:
                # Weighted average based on similarity
                total_weight = sum(face['similarity'] for face in neighbors)
                if total_weight > 0:
                    predicted_wealth = sum(
                        face['label'] * face['similarity'] for face in neighbors
                    ) / total_weight
                else:
                    predicted_wealth = np.mean([face['label'] for face in neighbors])
            else:
                # Simple average
                predicted_wealth = np.mean([face['label'] for face in neighbors])
            
            return {
                "predicted_wealth": float(predicted_wealth),
                "method": f"KNN-{k}" + (" (weighted)" if weighted else ""),
                "neighbors": neighbors,
                "neighbor_count": len(neighbors)
            }
            
        except Exception as e:
            logger.error(f"Wealth prediction failed: {e}")
            return {
                "predicted_wealth": 0,
                "method": f"KNN-{k}",
                "neighbors": [],
                "error": str(e)
            }
    
    def load_dataset_from_csv(self, csv_path: str = "dataset/dataset.csv") -> bool:
        """
        Load the dataset from CSV and extract embeddings for all images.
        
        Args:
            csv_path: Path to the CSV file with file_name,label columns
            
        Returns:
            bool: True if successful
        """
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"Loading {len(df)} images from {csv_path}")
            
            model_manager = get_model_manager()
            if not model_manager.is_loaded:
                logger.error("Model not loaded. Please load the FaceNet model first.")
                return False
            
            dataset_dir = Path(csv_path).parent
            successful_loads = 0
            
            for _, row in df.iterrows():
                file_name = row['file_name']
                label = float(row['label'])
                image_path = dataset_dir / file_name
                
                if not image_path.exists():
                    logger.warning(f"Image not found: {image_path}")
                    continue
                
                try:
                    image = Image.open(image_path)
                    
                    if image.mode == 'RGBA':
                        rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                        rgb_image.paste(image, mask=image.split()[3])
                        image = rgb_image
                    elif image.mode != 'RGB':
                        image = image.convert('RGB')
                    
                    embedding = model_manager.extract_embeddings(image)
                    
                    if embedding is not None:
                        if self.add_face_embedding(embedding, file_name, label, str(image_path)):
                            successful_loads += 1
                    else:
                        logger.warning(f"Failed to extract embeddings for {file_name}")
                        
                except Exception as e:
                    logger.error(f"Error processing {file_name}: {e}")
                    continue
            
            logger.info(f"Successfully loaded {successful_loads}/{len(df)} images")
            return successful_loads > 0
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return False
    
    def get_stats(self) -> Dict:
        """Get database statistics."""
        try:
            count = self.collection.count()
            
            if count == 0:
                return {
                    "total_faces": 0,
                    "labels": {"min": 0, "max": 0, "mean": 0, "std": 0}
                }
            
            all_data = self.collection.get(include=["metadatas"])
            labels = [metadata['label'] for metadata in all_data['metadatas']]
            
            return {
                "total_faces": count,
                "labels": {
                    "min": float(np.min(labels)),
                    "max": float(np.max(labels)),
                    "mean": float(np.mean(labels)),
                    "std": float(np.std(labels)),
                    "count": len(labels)
                },
                "db_path": str(self.db_path)
            }
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}
    
    def reset_database(self) -> bool:
        """Reset/clear the database."""
        try:
            self.client.delete_collection(COLLECTION_NAME)
            self.collection = self.client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"description": "Facial embeddings for wealth prediction"}
            )
            logger.info("Database reset successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to reset database: {e}")
            return False


_vector_db: Optional[WealthFaceDB] = None


def get_vector_db() -> WealthFaceDB:
    """Get or create the global vector database instance."""
    global _vector_db
    if _vector_db is None:
        _vector_db = WealthFaceDB()
    return _vector_db


def initialize_database() -> bool:
    """Initialize and load the database with dataset."""
    try:
        db = get_vector_db()
        
        if db.collection.count() > 0:
            logger.info("Database already contains data")
            return True
        
        return db.load_dataset_from_csv()
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        return False
