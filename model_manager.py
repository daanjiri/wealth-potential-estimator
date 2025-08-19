"""
Model manager for FaceNet identity detector.
Handles model downloading, caching, and inference.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Union
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_CACHE_DIR = Path(os.getenv("MODEL_CACHE_DIR", "./models"))
MODEL_CACHE_DIR.mkdir(exist_ok=True, parents=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelManager:
    """Manages FaceNet model loading and inference."""
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize the model manager.
        
        Args:
            device: Device to run model on ('cpu' or 'cuda'). If None, auto-detects.
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.transform = None
        self._initialize_transform()
        
    def _initialize_transform(self):
        """Initialize image transformation pipeline for FaceNet input."""
        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),  # FaceNet expects 160x160 input
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def download_and_load_model(self) -> bool:
        """
        Download and load the FaceNet model.
        
        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            from facenet_pytorch import InceptionResnetV1
            
            logger.info("Initializing FaceNet model...")
            
            try:
                logger.info("Loading pretrained FaceNet model (vggface2)...")
                self.model = InceptionResnetV1(
                    pretrained='vggface2',
                    device=self.device
                )
                logger.info("Model loaded from facenet-pytorch pretrained weights")
            except Exception as e:
                # Option 2: Try to load from HuggingFace if facenet-pytorch fails
                logger.warning(f"Failed to load from facenet-pytorch: {e}")
                logger.info("Attempting to load from HuggingFace Hub...")
                
                from huggingface_hub import hf_hub_download
                
                model_path = MODEL_CACHE_DIR / "facenet_20180402_114759_vggface2.pth"
                
                if not model_path.exists():
                    logger.info("Downloading FaceNet model from HuggingFace Hub...")
                    model_file = hf_hub_download(
                        repo_id='py-feat/facenet',
                        filename="facenet_20180402_114759_vggface2.pth",
                        cache_dir=str(MODEL_CACHE_DIR),
                        local_dir=str(MODEL_CACHE_DIR)
                    )
                    logger.info(f"Model downloaded to: {model_file}")
                else:
                    logger.info(f"Using cached model from: {model_path}")
                    model_file = str(model_path)
                
                self.model = InceptionResnetV1(
                    pretrained=None,
                    device=self.device
                )
                
                logger.info("Loading model weights...")
                state_dict = torch.load(model_file, map_location=self.device)
                self.model.load_state_dict(state_dict)
            
            self.model.eval()
            self.model.to(self.device)
            
            logger.info(f"Model successfully loaded on {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return False
    
    def extract_embeddings(self, image: Union[Image.Image, np.ndarray, torch.Tensor]) -> Optional[np.ndarray]:
        """
        Extract facial embeddings from an image.
        
        Args:
            image: Input image as PIL Image, numpy array, or torch tensor.
                   Should be a face crop, ideally 160x160 pixels.
        
        Returns:
            numpy array: 512-dimensional facial embeddings, or None if failed.
        """
        if self.model is None:
            logger.error("Model not loaded. Call download_and_load_model() first.")
            return None
        
        try:
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            elif isinstance(image, torch.Tensor):
                image = transforms.ToPILImage()(image)
            
            input_tensor = self.transform(image)
            
            input_tensor = input_tensor.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                embeddings = self.model(input_tensor)
            
            return embeddings.cpu().numpy().squeeze()
            
        except Exception as e:
            logger.error(f"Failed to extract embeddings: {str(e)}")
            return None
    
    def batch_extract_embeddings(self, images: list) -> Optional[np.ndarray]:
        """
        Extract facial embeddings from multiple images.
        
        Args:
            images: List of images (PIL Images, numpy arrays, or torch tensors).
        
        Returns:
            numpy array: Batch of 512-dimensional facial embeddings, or None if failed.
        """
        if self.model is None:
            logger.error("Model not loaded. Call download_and_load_model() first.")
            return None
        
        try:
            batch_tensors = []
            
            for image in images:
                if isinstance(image, np.ndarray):
                    image = Image.fromarray(image)
                elif isinstance(image, torch.Tensor):
                    image = transforms.ToPILImage()(image)
                
                input_tensor = self.transform(image)
                batch_tensors.append(input_tensor)
            
            batch = torch.stack(batch_tensors).to(self.device)
            
            with torch.no_grad():
                embeddings = self.model(batch)
            
            return embeddings.cpu().numpy()
            
        except Exception as e:
            logger.error(f"Failed to extract batch embeddings: {str(e)}")
            return None
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None


_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """
    Get or create the global model manager instance.
    
    Returns:
        ModelManager: The global model manager instance.
    """
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager


def initialize_model() -> bool:
    """
    Initialize and load the model at startup.
    
    Returns:
        bool: True if successful, False otherwise.
    """
    manager = get_model_manager()
    return manager.download_and_load_model()
