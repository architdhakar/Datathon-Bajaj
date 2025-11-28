# app/utils/image_processor.py
import cv2
import numpy as np
from PIL import Image
import requests
from io import BytesIO

def download_image(url: str) -> np.ndarray:
    """Download image from URL"""
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    return np.array(image)

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Preprocess image for better OCR results"""
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray)
    
    # Thresholding
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Deskew (simple version)
    deskewed = deskew_image(binary)
    
    return deskewed

def deskew_image(image: np.ndarray) -> np.ndarray:
    """Simple deskewing implementation"""
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
        
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, 
                            borderMode=cv2.BORDER_REPLICATE)
    return rotated