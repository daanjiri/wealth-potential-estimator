#!/usr/bin/env python3
"""
Script to download the FaceNet model during Docker build.
This ensures the model is cached in the image and doesn't need to be downloaded on every startup.
"""

import os
import sys
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_facenet_model():
    """
    Download the FaceNet model using facenet-pytorch.
    
    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        model_dir = Path("./models")
        model_dir.mkdir(exist_ok=True, parents=True)
        
        logger.info("Starting FaceNet model download...")
        logger.info(f"Model cache directory: {model_dir.absolute()}")
        
        logger.info("Importing facenet-pytorch and downloading pretrained model...")
        from facenet_pytorch import InceptionResnetV1
        
        model = InceptionResnetV1(pretrained='vggface2')
        logger.info("Model successfully downloaded and loaded")
        
        try:
            from huggingface_hub import hf_hub_download
            logger.info("Also downloading from HuggingFace Hub as backup...")
            
            model_file = hf_hub_download(
                repo_id='py-feat/facenet',
                filename="facenet_20180402_114759_vggface2.pth",
                cache_dir=str(model_dir),
                local_dir=str(model_dir)
            )
            logger.info(f"HuggingFace model downloaded to: {model_file}")
        except Exception as e:
            logger.warning(f"HuggingFace download failed (not critical): {e}")
        
        return True
            
    except Exception as e:
        logger.error(f"Failed to download model: {str(e)}")
        return False


def main():
    """Main function to run the download script."""
    logger.info("=" * 50)
    logger.info("FaceNet Model Download Script")
    logger.info("=" * 50)
    
    success = download_facenet_model()
    
    if success:
        logger.info("✅ Model download completed successfully!")
        sys.exit(0)
    else:
        logger.error("❌ Model download failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
