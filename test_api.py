#!/usr/bin/env python3
"""
Test script for the Wealth Potential Estimator API.
Tests health, model status, and wealth prediction endpoints.
"""

import requests
import json
import sys
from pathlib import Path


BASE_URL = "http://localhost:8000"


def test_health_check():
    """Test the health check endpoint."""
    print("Testing health check endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    
    if response.status_code == 200:
        print("✅ Health check passed")
        print(f"   Response: {response.json()}")
    else:
        print(f"❌ Health check failed: {response.status_code}")
    
    return response.status_code == 200


def test_model_status():
    """Test the model status endpoint."""
    print("\nTesting model status endpoint...")
    response = requests.get(f"{BASE_URL}/model-status")
    
    if response.status_code == 200:
        data = response.json()
        print("✅ Model status check passed")
        print(f"   Model loaded: {data['model_loaded']}")
        print(f"   Device: {data['device']}")
    else:
        print(f"❌ Model status check failed: {response.status_code}")
    
    return response.status_code == 200


def test_predict_wealth(image_path: str, k: int = 3):
    """Test the wealth prediction endpoint."""
    print(f"\nTesting wealth prediction with: {image_path} (k={k})")
    
    if not Path(image_path).exists():
        print(f"❌ Image file not found: {image_path}")
        return False
    
    with open(image_path, "rb") as f:
        files = {"file": (Path(image_path).name, f, "image/png")}
        params = {"k": k}
        response = requests.post(f"{BASE_URL}/predict-wealth", files=files, params=params)
    
    if response.status_code == 200:
        data = response.json()
        print("✅ Wealth prediction successful")
        print(f"   Original filename: {data['original_filename']}")
        print(f"   Predicted wealth: {data['prediction']['predicted_wealth']:.2f}")
        print(f"   Method: {data['prediction']['method']}")
        print(f"   Number of neighbors: {data['prediction']['neighbor_count']}")
        
        if 'neighbors' in data['prediction'] and data['prediction']['neighbors']:
            print("   Neighbors:")
            for i, neighbor in enumerate(data['prediction']['neighbors'], 1):
                print(f"     {i}. {neighbor['file_name']} - Label: {neighbor['label']:.2f}, Similarity: {neighbor['similarity']:.4f}")
    elif response.status_code == 503:
        print("⚠️  Model not yet loaded - this is expected on first startup")
        print("   The model will be downloaded and loaded in the background")
    else:
        print(f"❌ Wealth prediction failed: {response.status_code}")
        print(f"   Error: {response.text}")
    
    return response.status_code in [200, 503]


def main():
    """Run all tests."""
    print("=" * 60)
    print("Wealth Potential Estimator API Tests")
    print("=" * 60)
    
    try:
        requests.get(f"{BASE_URL}/health", timeout=2)
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API at http://localhost:8000")
        print("   Make sure the Docker container is running:")
        print("   docker-compose up --build")
        sys.exit(1)
    
    tests_passed = []
    
    tests_passed.append(test_health_check())
    tests_passed.append(test_model_status())
    
    sample_images = ["dataset/selfie1.png", "dataset/selfie2.png", "dataset/selfie3.png"]
    image_tested = False
    
    for sample_image in sample_images:
        if Path(sample_image).exists():
            print(f"\n📸 Testing with {sample_image}")
            tests_passed.append(test_predict_wealth(sample_image, k=3))
            image_tested = True
            break
    
    if not image_tested:
        print(f"\n⚠️  No sample images found in dataset/")
        print("   Skipping wealth prediction test")
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    passed = sum(tests_passed)
    total = len(tests_passed)
    
    if passed == total:
        print(f"✅ All tests passed ({passed}/{total})")
    else:
        print(f"⚠️  Some tests failed ({passed}/{total} passed)")
    
    print("\n📝 API Documentation available at:")
    print(f"   {BASE_URL}/docs")
    print(f"   {BASE_URL}/redoc")


if __name__ == "__main__":
    main()
