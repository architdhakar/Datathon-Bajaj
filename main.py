import os
import re
import json
import logging
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, validator
import requests
from io import BytesIO
from PIL import Image
import numpy as np
import easyocr
from pdf2image import convert_from_bytes
from difflib import SequenceMatcher
import cv2

# Gemini client
try:
    import google.generativeai as genai
    HAS_GENAI = True
except Exception:
    HAS_GENAI = False

# ---------------------------
# LOGGING CONFIGURATION
# ---------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------------------------
# CONSTANTS AND LIMITS
# ---------------------------
class DocumentFormat(str, Enum):
    PDF = "pdf"
    PNG = "png"
    JPG = "jpg"
    JPEG = "jpeg"

SUPPORTED_FORMATS = ['pdf', 'png', 'jpg', 'jpeg']

# Resource limits
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
MAX_PAGES = 10
MAX_IMAGE_PIXELS = 100000000  # ~100MP
REQUEST_TIMEOUT = 30
DOWNLOAD_TIMEOUT = 45

# OCR Configuration
OCR_LANGUAGES = ['en']
OCR_GPU = False

# Thresholds
Y_TOLERANCE = 10  # Reduced for better row detection
CONFIDENCE_THRESHOLD = 0.3  # Lowered to capture more text
SIMILARITY_THRESHOLD = 0.85  # Increased for better deduplication

# ---------------------------
# DATA MODELS
# ---------------------------
class BillItem(BaseModel):
    item_name: str = Field(..., description="Exactly as mentioned in the bill")
    item_amount: float = Field(..., description="Net Amount of the item post discounts as mentioned in the bill")
    item_rate: float = Field(..., description="Exactly as mentioned in the bill")
    item_quantity: float = Field(..., description="Exactly as mentioned in the bill")

class PageData(BaseModel):
    page_no: str
    page_type: str = Field(..., description="Bill Detail | Final Bill | Pharmacy")
    bill_items: List[BillItem]

class TokenUsage(BaseModel):
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

class ExtractionResponse(BaseModel):
    is_success: bool
    token_usage: TokenUsage
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class ExtractRequest(BaseModel):
    document: str = Field(..., description="URL to bill image/PDF")
    
    @validator('document')
    def validate_url(cls, v):
        """Validate URL format"""
        if not v.startswith(('http://', 'https://')):
            raise ValueError('URL must start with http:// or https://')
        
        try:
            result = urlparse(v)
            if not all([result.scheme, result.netloc]):
                raise ValueError('Invalid URL format')
        except Exception:
            raise ValueError('Invalid URL format')
            
        return v

# ---------------------------
# ENHANCED DOCUMENT PROCESSOR
# ---------------------------
class DocumentProcessor:
    """Enhanced document processing with image preprocessing"""
    
    @staticmethod
    def extract_file_extension(url: str) -> str:
        """Extract file extension from URL"""
        try:
            parsed_url = urlparse(url)
            path = parsed_url.path
            filename = path.split('/')[-1]
            extension = filename.split('.')[-1].lower()
            extension = extension.split('?')[0].split('&')[0]
            
            if extension in SUPPORTED_FORMATS:
                return extension
            else:
                return DocumentProcessor._detect_extension_from_url(url)
                
        except Exception as e:
            logger.warning(f"Failed to extract extension from URL: {e}")
            return "unknown"
    
    @staticmethod
    def _detect_extension_from_url(url: str) -> str:
        """Detect file extension from URL content type"""
        try:
            response = requests.head(url, timeout=10, allow_redirects=True)
            content_type = response.headers.get('content-type', '').lower()
            
            if 'pdf' in content_type:
                return 'pdf'
            elif 'png' in content_type:
                return 'png'
            elif 'jpeg' in content_type or 'jpg' in content_type:
                return 'jpg'
            elif 'image' in content_type:
                return 'png'
            
            path = urlparse(url).path
            if '.pdf' in path.lower():
                return 'pdf'
            elif '.png' in path.lower():
                return 'png'
            elif '.jpg' in path.lower() or '.jpeg' in path.lower():
                return 'jpg'
                
            return 'unknown'
        except Exception as e:
            logger.warning(f"Content type detection failed: {e}")
            return 'unknown'
    
    @staticmethod
    def preprocess_image(img: Image.Image) -> Image.Image:
        """Enhance image quality for better OCR"""
        try:
            # Convert to OpenCV format
            img_cv = np.array(img)
            
            # Convert to grayscale if needed
            if len(img_cv.shape) == 3:
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
            
            # Apply denoising
            img_cv = cv2.fastNlMeansDenoising(img_cv)
            
            # Enhance contrast using CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            img_cv = clahe.apply(img_cv)
            
            # Convert back to PIL Image
            img_processed = Image.fromarray(img_cv)
            return img_processed
            
        except Exception as e:
            logger.warning(f"Image preprocessing failed: {e}, using original image")
            return img
    
    @staticmethod
    def load_document(document_url: str) -> Tuple[List[Image.Image], str]:
        """Load document with image preprocessing"""
        try:
            logger.info(f"Loading document from URL: {document_url}")
            
            file_extension = DocumentProcessor.extract_file_extension(document_url)
            logger.info(f"Detected file extension: {file_extension}")
            
            if file_extension not in SUPPORTED_FORMATS:
                raise ValueError(f"Unsupported format: {file_extension}. Supported: {SUPPORTED_FORMATS}")
            
            # Download document
            logger.info("Starting document download...")
            response = requests.get(document_url, timeout=DOWNLOAD_TIMEOUT, stream=True)
            response.raise_for_status()
            
            content_length = response.headers.get('content-length')
            if content_length and int(content_length) > MAX_FILE_SIZE:
                raise ValueError(f"Document too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB")
            
            content = BytesIO()
            total_size = 0
            for chunk in response.iter_content(chunk_size=8192):
                total_size += len(chunk)
                if total_size > MAX_FILE_SIZE:
                    raise ValueError(f"Document exceeds size limit of {MAX_FILE_SIZE // (1024*1024)}MB")
                content.write(chunk)
            
            content.seek(0)
            logger.info(f"Download completed. Size: {total_size} bytes")
            
            images = []
            
            if file_extension == 'pdf':
                try:
                    logger.info("Converting PDF to images...")
                    images = convert_from_bytes(
                        content.getvalue(), 
                        dpi=300,  # Increased DPI for better quality
                        first_page=1, 
                        last_page=MAX_PAGES,
                        size=(None, 4000)
                    )
                    logger.info(f"Extracted {len(images)} pages from PDF")
                    
                except Exception as pdf_error:
                    logger.error(f"PDF conversion failed: {pdf_error}")
                    raise ValueError(f"PDF processing failed: {str(pdf_error)}")
            else:
                try:
                    logger.info("Processing image...")
                    img = Image.open(content)
                    
                    width, height = img.size
                    total_pixels = width * height
                    if total_pixels > MAX_IMAGE_PIXELS:
                        raise ValueError(f"Image too large: {width}x{height} = {total_pixels} pixels")
                    
                    logger.info(f"Image size: {width}x{height}")
                    img = img.convert("RGB")
                    images = [img]
                    logger.info("Loaded image document successfully")
                    
                except Exception as img_error:
                    logger.error(f"Failed to open image: {img_error}")
                    raise ValueError(f"Invalid image format: {file_extension}")
            
            if not images:
                raise ValueError("No content could be extracted from the document")
            
            # Apply image preprocessing
            processed_images = []
            for img in images:
                processed_img = DocumentProcessor.preprocess_image(img)
                processed_images.append(processed_img)
            
            logger.info(f"Document processing completed. {len(processed_images)} images ready for OCR")
            return processed_images, file_extension
            
        except requests.RequestException as e:
            logger.error(f"Network error loading document: {str(e)}")
            raise ValueError(f"Failed to download document: {str(e)}")
        except Exception as e:
            logger.error(f"Document loading failed: {str(e)}")
            raise

# ---------------------------
# ENHANCED OCR ENGINE
# ---------------------------
class OCREngine:
    """Enhanced OCR with better configuration"""
    
    def __init__(self):
        try:
            self.reader = easyocr.Reader(
                OCR_LANGUAGES, 
                gpu=OCR_GPU,
                model_storage_directory='./model_storage',
                download_enabled=True
            )
            logger.info("OCR engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OCR engine: {e}")
            raise
    
    def extract_text(self, img: Image.Image) -> List[Tuple[List, str, float]]:
        """Enhanced text extraction with better parameters"""
        try:
            width, height = img.size
            total_pixels = width * height
            
            # Optimize image size for OCR
            if total_pixels > 4000000:  # 4MP
                max_dimension = 2000
                ratio = min(max_dimension / width, max_dimension / height)
                new_size = (int(width * ratio), int(height * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                logger.info(f"Resized image to: {new_size}")
            
            img_array = np.array(img)
            
            logger.info("Starting OCR processing...")
            # Use better OCR parameters
            results = self.reader.readtext(
                img_array, 
                paragraph=False, 
                detail=1,
                batch_size=4,
                decoder='beamsearch',
                beamWidth=10,
                contrast_ths=0.1,
                adjust_contrast=0.5,
                width_ths=0.7,
                ycenter_ths=0.5
            )
            logger.info(f"OCR completed. Found {len(results)} text elements")
            
            filtered_results = []
            for bbox, text, confidence in results:
                if confidence >= CONFIDENCE_THRESHOLD:
                    # Better text cleaning
                    cleaned_text = re.sub(r'[^\w\s\.\,\-\+\@\$\₹\&\/\\]', '', text.strip())
                    if cleaned_text and len(cleaned_text) >= 1:  # Reduced minimum length
                        filtered_results.append((bbox, cleaned_text, confidence))
            
            logger.info(f"Filtered to {len(filtered_results)} confident text elements")
            return filtered_results
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {str(e)}", exc_info=True)
            return []

# ---------------------------
# ENHANCED TEXT PROCESSOR
# ---------------------------
class TextProcessor:
    """Advanced text processing with better row detection"""
    
    @staticmethod
    def calculate_bbox_center(bbox: List) -> Tuple[float, float]:
        """Calculate center point of bounding box"""
        x_coords = [point[0] for point in bbox]
        y_coords = [point[1] for point in bbox]
        return np.mean(x_coords), np.mean(y_coords)
    
    @staticmethod
    def cluster_text_rows(ocr_results: List) -> List[Dict]:
        """Enhanced row clustering with dynamic Y-tolerance"""
        if not ocr_results:
            return []
        
        items_with_coords = []
        for bbox, text, confidence in ocr_results:
            center_x, center_y = TextProcessor.calculate_bbox_center(bbox)
            items_with_coords.append({
                'text': text,
                'confidence': confidence,
                'center_y': center_y,
                'center_x': center_x,
                'bbox': bbox
            })
        
        # Sort by Y coordinate
        items_with_coords.sort(key=lambda x: x['center_y'])
        
        # Calculate dynamic Y-tolerance based on text height distribution
        if len(items_with_coords) > 1:
            y_coords = [item['center_y'] for item in items_with_coords]
            y_diffs = [abs(y_coords[i] - y_coords[i-1]) for i in range(1, len(y_coords))]
            dynamic_tolerance = np.percentile(y_diffs, 25) * 1.5
            tolerance = max(Y_TOLERANCE, dynamic_tolerance)
        else:
            tolerance = Y_TOLERANCE
        
        logger.info(f"Using dynamic Y-tolerance: {tolerance:.2f}")
        
        rows = []
        current_row = []
        previous_y = None
        
        for item in items_with_coords:
            current_y = item['center_y']
            
            if previous_y is None or abs(current_y - previous_y) <= tolerance:
                current_row.append(item)
            else:
                if current_row:
                    rows.append(TextProcessor.merge_row_items(current_row))
                current_row = [item]
            
            previous_y = current_y
        
        if current_row:
            rows.append(TextProcessor.merge_row_items(current_row))
        
        logger.info(f"Organized into {len(rows)} text rows")
        return rows
    
    @staticmethod
    def merge_row_items(row_items: List) -> Dict:
        """Merge items in the same row with proper spacing"""
        row_items.sort(key=lambda x: x['center_x'])
        
        # Reconstruct text with proper spacing based on X coordinates
        full_text = row_items[0]['text']
        for i in range(1, len(row_items)):
            prev_item = row_items[i-1]
            curr_item = row_items[i]
            
            x_gap = curr_item['center_x'] - prev_item['center_x'] - 10  # Approximate character width
            
            if x_gap > 30:  # Large gap indicates separate columns
                full_text += "    " + curr_item['text']  # Extra spaces for column separation
            elif x_gap > 15:  # Medium gap
                full_text += "  " + curr_item['text']  # Two spaces
            else:  # Small gap
                full_text += " " + curr_item['text']  # Single space
        
        avg_confidence = np.mean([item['confidence'] for item in row_items])
        
        return {
            'text': full_text.strip(),
            'confidence': avg_confidence,
            'item_count': len(row_items),
            'raw_items': row_items
        }
    
    @staticmethod
    def is_line_item_candidate(text: str) -> bool:
        """Enhanced line item detection"""
        text_lower = text.lower().strip()
        
        if len(text_lower) < 3:
            return False
        
        # Enhanced exclusion patterns
        exclude_patterns = [
            r'^invoice\s*#?\s*\d*', r'^bill\s*#?\s*\d*', r'^date\s*:?\s*\d', 
            r'^time\s*:?\s*\d', r'^phone\s*:?', r'^address\s*:?', r'^email\s*:?',
            r'^customer\s*:?', r'^sub\s*total', r'^total\s+[\d,]', r'^grand\s+total',
            r'^amount\s+due', r'^gst\s*', r'^tax\s*', r'^discount\s*', r'^thank\s+you',
            r'^page\s+\d+', r'^category\s+total', r'^final\s+total', r'^net\s+total',
            r'^balance\s+due', r'^item\s+total', r'^bill\s+total', r'^final\s+amount',
            r'^invoice\s+total', r'^round\s+off', r'^payable\s+amount'
        ]
        
        for pattern in exclude_patterns:
            if re.search(pattern, text_lower):
                return False
        
        # Must contain at least one letter and one number/digit character
        has_letter = bool(re.search(r'[a-zA-Z]', text))
        has_digit_or_currency = bool(re.search(r'[\d\$\₹\€\£]', text))
        
        if not has_letter or not has_digit_or_currency:
            return False
        
        # Check for common bill patterns
        bill_patterns = [
            r'.+\s+\d+\.?\d*\s+[\d,]+\.?\d{2}',  # Name Qty Amount
            r'.+\s+[\d,]+\.?\d{2}\s+\d+\.?\d*',  # Name Amount Qty
            r'.+\s+@\s*[\d,]+\.?\d{2}',          # Name @ Rate
            r'.+\s+[\d,]+\.?\d{2}$',             # Name Amount
            r'^[\d,]+\.?\d{2}\s+.+',             # Amount Name
            r'.+\s+x\s*\d+\.?\d*',               # Name x Qty
        ]
        
        for pattern in bill_patterns:
            if re.search(pattern, text):
                return True
        
        return False

# ---------------------------
# IMPROVED LINE ITEM PARSER
# ---------------------------
class LineItemParser:
    """Greatly improved line item parsing with multiple strategies"""
    
    # Comprehensive patterns for different bill formats
    ITEM_PATTERNS = [
        # Standard format: "Item Name Qty Rate Amount"
        r'^(.+?)\s+(\d+\.?\d*)\s+([\d,]+\.?\d{0,2})\s+([\d,]+\.?\d{0,2})$',
        
        # Format with @: "Item Name Qty @ Rate Amount"  
        r'^(.+?)\s+(\d+\.?\d*)\s*@\s*([\d,]+\.?\d{0,2})\s+([\d,]+\.?\d{0,2})$',
        
        # Format with x: "Item Name Qty x Rate Amount"
        r'^(.+?)\s+(\d+\.?\d*)\s*x\s*([\d,]+\.?\d{0,2})\s+([\d,]+\.?\d{0,2})$',
        
        # Simple format: "Item Name Amount"
        r'^(.+?)\s+([\d,]+\.?\d{0,2})$',
        
        # Amount first: "Amount Item Name"
        r'^([\d,]+\.?\d{0,2})\s+(.+)$',
        
        # Reordered: "Item Name Rate Qty Amount"
        r'^(.+?)\s+([\d,]+\.?\d{0,2})\s+(\d+\.?\d*)\s+([\d,]+\.?\d{0,2})$',
        
        # With dashes: "Item Name - Amount"
        r'^(.+?)\s*[-–—]\s*([\d,]+\.?\d{0,2})$',
        
        # Multi-column with tabs: "Item Name    Qty    Rate    Amount"
        r'^(.+?)\s{2,}(\d+\.?\d*)\s{2,}([\d,]+\.?\d{0,2})\s{2,}([\d,]+\.?\d{0,2})$',
        
        # Pharmacy format: "Medicine Name Pack Size Qty Amount"
        r'^(.+?)\s+(\d+[a-zA-Z]*)\s+(\d+\.?\d*)\s+([\d,]+\.?\d{0,2})$',
        # Medical bill pattern: "Service Name Qty Rate Amount"
        r'^([A-Za-z][^0-9]+?)\s+(\d+\.?\d*)\s+([\d,]+\.?\d{0,2})\s+([\d,]+\.?\d{0,2})$',
        
        # Pattern with description and codes: "Description Code Date Qty Rate Amount"
        r'^([A-Za-z].+?)\s+\d*\s*\d{2}/\d{2}/\d{4}\s+(\d+\.?\d*)\s+([\d,]+\.?\d{0,2})\s+([\d,]+\.?\d{0,2})',
        
        # Medical service pattern: "Service Name - Amount" 
        r'^([A-Za-z][^0-9]+?)\s*[-–—]\s*([\d,]+\.?\d{0,2})$',
        
        # Pattern with doctor names: "Service DR NAME- Qty Rate Discount Amount"
        r'^([A-Za-z\s]+)\s+DR\s+[A-Z]+\s+[A-Z\s-]+\s+(\d+\.?\d*)\s+([\d,]+\.?\d{0,2})\s+[\d,]+\.?\d{0,2}\s+([\d,]+\.?\d{0,2})$',
        
        # Simple medical service: "SERVICE NAME Amount"
        r'^([A-Z][A-Za-z\s\(\)]+)\s+([\d,]+\.?\d{0,2})$',
    ]
    
    @staticmethod
    def parse_numeric_value(text: str) -> Optional[float]:
        """Robust numeric value extraction"""
        if not text:
            return None
            
        # Handle currency symbols and commas
        cleaned = re.sub(r'[^\d\.]', '', text.replace(',', ''))
        
        try:
            value = float(cleaned)
            return value if value >= 0 else None
        except (ValueError, TypeError):
            return None
    
    @staticmethod
    def extract_line_item(row_text: str) -> Optional[Dict]:
        """Advanced line item extraction with multiple fallbacks"""
        cleaned_text = re.sub(r'\s+', ' ', row_text.strip())
        
        # Skip obvious non-line items
        if LineItemParser._is_non_item_row(cleaned_text):
            return None
        
        # Try all patterns
        for pattern_idx, pattern in enumerate(LineItemParser.ITEM_PATTERNS):
            match = re.match(pattern, cleaned_text)
            if match:
                try:
                    groups = match.groups()
                    parsed_item = LineItemParser._parse_pattern_groups(groups, pattern_idx, cleaned_text)
                    if parsed_item and LineItemParser._validate_parsed_item(parsed_item):
                        return parsed_item
                except Exception as e:
                    logger.debug(f"Pattern {pattern_idx} failed: {e}")
                    continue
        
        # Fallback: Try to extract using heuristic approach
        return LineItemParser._heuristic_extraction(cleaned_text)
    
    @staticmethod
    def _parse_pattern_groups(groups: Tuple, pattern_idx: int, original_text: str) -> Optional[Dict]:
        """Parse matched groups based on pattern type"""
        if len(groups) == 4:
            # 4-field patterns
            if pattern_idx in [0, 1, 2, 5, 7, 8]:  # Various 4-field formats
                name = groups[0].strip()
                
                if pattern_idx in [0, 7]:  # "Name Qty Rate Amount" or tab-separated
                    qty_str, rate_str, amount_str = groups[1], groups[2], groups[3]
                elif pattern_idx in [1, 2]:  # "Name Qty @ Rate Amount" or "Name Qty x Rate Amount"
                    qty_str, rate_str, amount_str = groups[1], groups[2], groups[3]
                elif pattern_idx == 5:  # "Name Rate Qty Amount"
                    rate_str, qty_str, amount_str = groups[1], groups[2], groups[3]
                elif pattern_idx == 8:  # Pharmacy format
                    # Skip pack size for now, use name + pack size as item name
                    name = f"{groups[0]} {groups[1]}".strip()
                    qty_str, amount_str = groups[2], groups[3]
                    rate_str = None
                else:
                    return None
                
                return LineItemParser._build_item_dict(name, qty_str, rate_str, amount_str, original_text)
        
        elif len(groups) == 2:
            # 2-field patterns
            if pattern_idx == 3:  # "Name Amount"
                name, amount_str = groups[0].strip(), groups[1]
                return LineItemParser._build_item_dict(name, None, None, amount_str, original_text)
            elif pattern_idx == 4:  # "Amount Name"
                amount_str, name = groups[0], groups[1].strip()
                return LineItemParser._build_item_dict(name, None, None, amount_str, original_text)
            elif pattern_idx == 6:  # "Name - Amount"
                name, amount_str = groups[0].strip(), groups[1]
                return LineItemParser._build_item_dict(name, None, None, amount_str, original_text)
        
        return None
    
    @staticmethod
    def _build_item_dict(name: str, qty_str: str, rate_str: str, amount_str: str, original_text: str) -> Dict:
        """Build item dictionary with calculated fields"""
        qty = LineItemParser.parse_numeric_value(qty_str) if qty_str else 1.0
        rate = LineItemParser.parse_numeric_value(rate_str) if rate_str else 0.0
        amount = LineItemParser.parse_numeric_value(amount_str) if amount_str else None
        
        # Calculate missing fields logically
        if amount is None:
            if qty and rate:
                amount = qty * rate
            else:
                return None
        elif rate == 0 and qty and qty != 1:
            rate = amount / qty
        elif qty == 1 and rate == 0 and amount:
            rate = amount
        
        # Clean item name
        name = LineItemParser._clean_item_name(name, original_text)
        
        if name and amount and amount > 0:
            return {
                'item_name': name,
                'item_quantity': qty or 1.0,
                'item_rate': rate or 0.0,
                'item_amount': amount,
                'source': 'regex'
            }
        
        return None
    
    @staticmethod
    def _clean_item_name(name: str, original_text: str) -> str:
        """Clean item name by removing prices and irrelevant text"""
        # Remove numeric patterns that look like prices
        cleaned = re.sub(r'[\$\₹\€\£]?\s*[\d,]+\s*\.?\s*\d{0,2}', '', name)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Remove common suffixes
        suffixes = ['-', '--', '---', '----']
        for suffix in suffixes:
            if cleaned.endswith(suffix):
                cleaned = cleaned[:-len(suffix)].strip()
        
        return cleaned if cleaned else name
    
    @staticmethod
    def _heuristic_extraction(text: str) -> Optional[Dict]:
        """Fallback heuristic extraction for difficult cases"""
        # Split by multiple spaces (column-based)
        parts = re.split(r'\s{2,}', text)
        if len(parts) >= 2:
            # Assume last part is amount
            amount = LineItemParser.parse_numeric_value(parts[-1])
            if amount and amount > 0:
                name = ' '.join(parts[:-1]).strip()
                name = LineItemParser._clean_item_name(name, text)
                
                if name:
                    return {
                        'item_name': name,
                        'item_quantity': 1.0,
                        'item_rate': 0.0,
                        'item_amount': amount,
                        'source': 'heuristic'
                    }
        
        return None
    
    @staticmethod
    def _validate_parsed_item(item: Dict) -> bool:
        """Validate parsed item for basic sanity"""
        if not item or not item.get('item_name') or item.get('item_amount') is None:
            return False
        
        if item['item_amount'] <= 0 or item['item_amount'] > 1000000:
            return False
        
        if item.get('item_quantity', 1) <= 0:
            return False
        
        if item.get('item_rate', 0) < 0:
            return False
        
        return True
    
    @staticmethod
    def _is_non_item_row(text: str) -> bool:
        """Enhanced non-item row detection"""
        text_lower = text.lower()
        
        non_item_indicators = [
            'total', 'subtotal', 'category total', 'final total', 'grand total',
            'amount due', 'balance', 'tax', 'gst', 'discount', 'thank you',
            'invoice', 'bill', 'date', 'time', 'phone', 'address', 'round off',
            'payable', 'net payable', 'item total', 'bill total', 'invoice total'
        ]
        
        # Check for exact matches at start of string
        for indicator in non_item_indicators:
            if text_lower.startswith(indicator) or f" {indicator} " in f" {text_lower} ":
                return True
        
        return False

# ---------------------------
# IMPROVED LLM PARSER
# ---------------------------
class LLMAssistedParser:
    """Enhanced LLM parser with better prompting"""
    
    def __init__(self, api_key: str = None, model: str = "gemini-1.5-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model
        self.token_usage = {"total_tokens": 0, "input_tokens": 0, "output_tokens": 0}
        self.enabled = False
        
        if self.api_key and HAS_GENAI:
            try:
                genai.configure(api_key=self.api_key)
                self.model_instance = genai.GenerativeModel(self.model)
                self.enabled = True
                logger.info("Gemini client configured for LLM parsing")
            except Exception as e:
                logger.warning(f"Failed to configure Gemini client: {e}")
                self.enabled = False
        else:
            logger.info("LLM parsing disabled - no API key or library")
    
    def parse_with_llm(self, ambiguous_rows: List[str]) -> Tuple[List[Dict], Dict]:
        """Use LLM for ambiguous rows with improved context"""
        if not ambiguous_rows or not self.enabled:
            return [], self.token_usage
        
        try:
            logger.info(f"Sending {len(ambiguous_rows)} ambiguous rows to LLM")
            prompt = self._build_extraction_prompt(ambiguous_rows)
            response_text, usage = self._call_gemini(prompt)
            items = self._parse_llm_response(response_text)
            
            # Update token usage
            if usage:
                self.token_usage["total_tokens"] += usage.get("total_tokens", 0)
                self.token_usage["input_tokens"] += usage.get("input_tokens", 0)
                self.token_usage["output_tokens"] += usage.get("output_tokens", 0)
            
            logger.info(f"LLM parsing completed. Items extracted: {len(items)}")
            return items, self.token_usage
        except Exception as e:
            logger.error(f"LLM parsing failed: {e}")
            return [], self.token_usage
    
    def _build_extraction_prompt(self, rows: List[str]) -> str:
        """Build comprehensive extraction prompt with perfect examples"""
        rows_text = "\n".join([f"{i+1}. {row}" for i, row in enumerate(rows)])
        
        return f"""You are an expert at extracting line items from medical bills, pharmacy bills, and hospital invoices. Extract ONLY product/service line items.

    CRITICAL RULES:
    1. Extract ONLY line items with amounts - skip totals, taxes, headers, footers, category totals
    2. Each item MUST have: item_name (string), item_amount (number)
    3. item_amount is the FINAL amount for that line item (Net Amount)
    4. Include quantity and rate ONLY if clearly present in the text
    5. If quantity not clear, set to 1.0
    6. If rate not clear, set to 0.0
    7. DO NOT calculate rate from amount/quantity unless explicitly shown
    8. Clean item names - remove doctor names, dates, codes, amounts from the name
    9. Skip any rows that are clearly not line items (totals, headers, category totals, etc.)
    10. Focus on the main item description, not supplementary details

    PERFECT EXAMPLES FROM REAL BILLS:

    HOSPITAL BILL EXAMPLES:
    - "Consultation Charge DR PREETH MARY JOSEPH- 1.00 300.00 0.00 300.00" → name: "Consultation Charge", quantity: 1.0, rate: 300.00, amount: 300.00
    - "RR -2-Room Rent Step Down Icu 1.00 1,000.00 0.00 1,000.00" → name: "Room Rent Step Down Icu", quantity: 1.0, rate: 1000.00, amount: 1000.00
    - "SG204-Room Rent Single Non Ac Room A 3.00 900.00 0.00 2,700.00" → name: "Room Rent Single Non Ac Room A", quantity: 3.0, rate: 900.00, amount: 2700.00
    - "BT 1.00 50.00 0.00 50.00" → name: "BT", quantity: 1.0, rate: 50.00, amount: 50.00
    - "Liver Function Test(LFT) 1.00 400.00 0.00 400.00" → name: "Liver Function Test", quantity: 1.0, rate: 400.00, amount: 400.00
    - "Chest PA 1.00 250.00 0.00 250.00" → name: "Chest PA", quantity: 1.0, rate: 250.00, amount: 250.00
    - "SURGEON FEE 1.00 3,300.00 0.00 3,300.00" → name: "Surgeon Fee", quantity: 1.0, rate: 3300.00, amount: 3300.00

    PHARMACY BILL EXAMPLES:
    - "Livi 300mg Tab 20/11/2025 14 32.00 448.00 0.00" → name: "Livi 300mg Tab", quantity: 14.0, rate: 32.00, amount: 448.00
    - "Metnuro 20/11/2025 7 17.72 124.03 0.00" → name: "Metnuro", quantity: 7.0, rate: 17.72, amount: 124.03
    - "Pizat 4.5 20/11/2025 2 419.06 838.12 0.00" → name: "Pizat 4.5", quantity: 2.0, rate: 419.06, amount: 838.12
    - "Supralite Os Syp 20/11/2025 1 289.69 289.69 0.00" → name: "Supralite Os Syp", quantity: 1.0, rate: 289.69, amount: 289.69

    TABLE FORMAT EXAMPLES:
    - "RENAL FUNCTION TEST (RFT) 1 450.00 450.00" → name: "Renal Function Test", quantity: 1.0, rate: 450.00, amount: 450.00
    - "ELECTROLYTES 1 250.00 250.00" → name: "Electrolytes", quantity: 1.0, rate: 250.00, amount: 250.00
    - "GLUCOSE FASTING (FBS) 1 50.00 50.00" → name: "Glucose Fasting", quantity: 1.0, rate: 50.00, amount: 50.00
    - "X-Ray 1 500.00 500.00" → name: "X-Ray", quantity: 1.0, rate: 500.00, amount: 500.00
    - "Registration 1 300.00 300.00" → name: "Registration", quantity: 1.0, rate: 300.00, amount: 300.00

    MEDICAL SERVICE EXAMPLES:
    - "Consultation (Dr. Neo Church Tharsis(Diabetologist, General Medicine))" → name: "Consultation", quantity: 1.0, rate: 0.0, amount: [extract from context]
    - "RENAL FUNCTION TEST (RFT)" → name: "Renal Function Test", quantity: 1.0, rate: 0.0, amount: [extract from context]
    - "URINE COMPLETE ANALYSIS" → name: "Urine Complete Analysis", quantity: 1.0, rate: 0.0, amount: [extract from context]
    - "Nebulization" → name: "Nebulization", quantity: 1.0, rate: 0.0, amount: [extract from context]

    NAME CLEANING RULES:
    - Remove doctor names: "DR PREETH MARY JOSEPH-" → remove
    - Remove dates: "20/11/2025" → remove  
    - Remove codes: "Cpt Code", "Sl#" values → remove
    - Remove amounts from names: keep only the item description
    - Keep medical terms, medicine names, service names
    - Simplify but keep essential descriptors: "300mg", "Tab", "Syp", "Scan", "Test"

    SKIP THESE (NON-ITEMS):
    - "Category Total", "Total:", "Subtotal", "Final Amount"
    - "Particulars", "Description", "Qty", "Rate", "Amount" headers
    - "Page 1 of 4", "Printed On : 19/11/2025 12:09 PM"
    - Section headers: "CONSULTATION", "INVESTIGATION CHARGES", "LABORATORY SERVICES"

    ROWS TO PROCESS:
    {rows_text}

    Return ONLY a valid JSON array with objects containing: item_name, item_amount, item_quantity, item_rate"""
    
    def _call_gemini(self, prompt: str) -> Tuple[str, Dict]:
        """Call Gemini API"""
        try:
            response = self.model_instance.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.0,
                    max_output_tokens=2048,
                )
            )
            
            response_text = response.text
            usage = self._extract_token_usage(response)
            
            return response_text, usage
            
        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            return "", {"total_tokens": 0, "input_tokens": 0, "output_tokens": 0}
    
    def _extract_token_usage(self, response) -> Dict:
        """Extract token usage"""
        usage = {"total_tokens": 0, "input_tokens": 0, "output_tokens": 0}
        
        try:
            if hasattr(response, 'usage_metadata'):
                usage["total_tokens"] = response.usage_metadata.total_token_count
                usage["input_tokens"] = response.usage_metadata.prompt_token_count
                usage["output_tokens"] = response.usage_metadata.candidates_token_count
            else:
                # Estimate tokens
                input_estimate = len(response.text.split()) * 1.3 if hasattr(response, 'text') else 0
                output_estimate = len(response.text.split()) * 1.3 if hasattr(response, 'text') else 0
                usage = {
                    "total_tokens": int(input_estimate + output_estimate),
                    "input_tokens": int(input_estimate),
                    "output_tokens": int(output_estimate)
                }
        except Exception as e:
            logger.warning(f"Could not extract token usage: {e}")
        
        return usage
    
    def _parse_llm_response(self, response: str) -> List[Dict]:
        """Parse LLM response with validation"""
        try:
            if not response:
                return []
                
            # Extract JSON from response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if not json_match:
                return []
            
            json_data = json.loads(json_match.group())
            
            parsed_items = []
            for item in json_data:
                try:
                    name = self._clean_item_name(item.get('item_name') or item.get('name', ''))
                    amount = self._safe_float(item.get('item_amount') or item.get('amount'))
                    quantity = self._safe_float(item.get('item_quantity') or item.get('quantity'), 1.0)
                    rate = self._safe_float(item.get('item_rate') or item.get('rate'), 0.0)
                    
                    if name and amount and amount > 0:
                        parsed_items.append({
                            'item_name': name,
                            'item_amount': round(float(amount), 2),
                            'item_quantity': round(float(quantity), 2),
                            'item_rate': round(float(rate), 2),
                            'source': 'llm'
                        })
                        
                except Exception as e:
                    logger.debug(f"Skipping invalid LLM item: {e}")
                    continue
                    
            return parsed_items
            
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {str(e)}")
            return []
    
    def _clean_item_name(self, name: str) -> str:
        """Clean item name"""
        if not name:
            return ""
        
        cleaned = re.sub(r'[\$\₹\€\£]?\s*[\d,]+\s*\.?\s*\d{0,2}', '', str(name))
        cleaned = re.sub(r'[^\w\s\-\.]', '', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned
    
    def _safe_float(self, value: Any, default: float = None) -> Optional[float]:
        """Safe float conversion"""
        if value is None:
            return default
        
        try:
            if isinstance(value, (int, float)):
                return float(value)
            
            cleaned = re.sub(r'[^\d\.]', '', str(value))
            return float(cleaned) if cleaned else default
        except (ValueError, TypeError):
            return default

# ---------------------------
# PAGE CLASSIFIER (UNCHANGED)
# ---------------------------
class PageClassifier:
    """Classify page types"""
    
    @staticmethod
    def classify_page(text_rows: List[str]) -> str:
        """Classify page as per required types"""
        full_text = ' '.join(text_rows).lower()
        
        pharmacy_terms = ['pharmacy', 'medical', 'medicine', 'drug', 'tablet', 'capsule', 'syrup']
        if any(term in full_text for term in pharmacy_terms):
            return "Pharmacy"
        
        final_terms = ['final total', 'grand total', 'amount due', 'net payable', 'balance due']
        total_patterns = [r'total\s+[\d,]+\.[\d]{2}', r'amount\s+[\d,]+\.[\d]{2}']
        
        if any(term in full_text for term in final_terms) or \
           any(re.search(pattern, full_text) for pattern in total_patterns):
            return "Final Bill"
        
        return "Bill Detail"

# ---------------------------
# DEDUPLICATION ENGINE (UNCHANGED)
# ---------------------------
class DeduplicationEngine:
    """Remove duplicate line items"""
    
    @staticmethod
    def calculate_similarity(text1: str, text2: str) -> float:
        """Calculate text similarity ratio"""
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    @staticmethod
    def remove_duplicates(items: List[Dict]) -> List[Dict]:
        """Remove duplicate line items"""
        if len(items) <= 1:
            return items
        
        unique_items = []
        used_indices = set()
        
        for i, item in enumerate(items):
            if i in used_indices:
                continue
                
            current_item = item
            
            for j in range(i + 1, len(items)):
                if j in used_indices:
                    continue
                    
                other_item = items[j]
                similarity = DeduplicationEngine.calculate_similarity(
                    item['item_name'], 
                    other_item['item_name']
                )
                
                if similarity >= SIMILARITY_THRESHOLD:
                    # Merge missing fields
                    if not current_item.get('item_quantity') and other_item.get('item_quantity'):
                        current_item['item_quantity'] = other_item['item_quantity']
                    if not current_item.get('item_rate') and other_item.get('item_rate'):
                        current_item['item_rate'] = other_item['item_rate']
                    
                    used_indices.add(j)
            
            unique_items.append(current_item)
        
        logger.info(f"Reduced {len(items)} to {len(unique_items)} items after deduplication")
        return unique_items

# ---------------------------
# IMPROVED BILL EXTRACTOR
# ---------------------------
class BillExtractor:
    """Improved bill extractor with enhanced accuracy"""
    
    def __init__(self, gemini_api_key: str = None):
        logger.info("Initializing Enhanced BillExtractor...")
        self.doc_processor = DocumentProcessor()
        self.ocr_engine = OCREngine()
        self.text_processor = TextProcessor()
        self.item_parser = LineItemParser()
        self.llm_parser = LLMAssistedParser(api_key=gemini_api_key or os.getenv("GEMINI_API_KEY"))
        self.page_classifier = PageClassifier()
        self.dedup_engine = DeduplicationEngine()
        logger.info("Enhanced BillExtractor initialized successfully")
    
    def extract_from_document(self, document_url: str) -> Dict:
        """Main extraction pipeline"""
        total_token_usage = {"total_tokens": 0, "input_tokens": 0, "output_tokens": 0}
        
        try:
            logger.info(f"Starting extraction pipeline for URL: {document_url}")
            
            images, doc_format = self.doc_processor.load_document(document_url)
            logger.info(f"Successfully loaded {len(images)} images, format: {doc_format}")
            
            pagewise_data = []
            all_line_items = []
            
            for page_num, image in enumerate(images, 1):
                logger.info(f"Processing page {page_num}/{len(images)}")
                
                page_result = self._process_single_page(image, page_num, total_token_usage)
                pagewise_data.append(page_result)
                all_line_items.extend(page_result['bill_items'])
                logger.info(f"Page {page_num} completed - {len(page_result['bill_items'])} items found")
            
            response_data = {
                "pagewise_line_items": pagewise_data,
                "total_item_count": len(all_line_items),
                "document_format": doc_format,
                "pages_processed": len(images)
            }
            
            logger.info(f"Extraction completed successfully. Total items: {len(all_line_items)}")
            return {
                "is_success": True,
                "token_usage": total_token_usage,
                "data": response_data
            }
            
        except Exception as e:
            logger.error(f"Extraction pipeline failed: {str(e)}", exc_info=True)
            return {
                "is_success": False,
                "token_usage": total_token_usage,
                "data": None,
                "error": str(e)
            }
    
    def _process_single_page(self, image: Image.Image, page_num: int, token_usage: Dict) -> Dict:
        """Process single page with enhanced item extraction"""
        try:
            logger.info(f"Starting OCR for page {page_num}...")
            ocr_results = self.ocr_engine.extract_text(image)
            logger.info(f"OCR completed for page {page_num}. Found {len(ocr_results)} text elements")
            
            text_rows = self.text_processor.cluster_text_rows(ocr_results)
            row_texts = [row['text'] for row in text_rows]
            logger.info(f"Text clustering completed. {len(row_texts)} rows identified")
            
            page_type = self.page_classifier.classify_page(row_texts)
            logger.info(f"Page classified as: {page_type}")
            
            line_items = []
            ambiguous_rows = []
            
            # First pass: Try regex parsing
            for row in text_rows:
                if not self.text_processor.is_line_item_candidate(row['text']):
                    continue
                
                parsed_item = self.item_parser.extract_line_item(row['text'])
                if parsed_item:
                    line_items.append(parsed_item)
                else:
                    ambiguous_rows.append(row['text'])
            
            logger.info(f"First pass: {len(line_items)} regex items, {len(ambiguous_rows)} ambiguous rows")
            
            # Second pass: LLM for ambiguous rows
            if ambiguous_rows and self.llm_parser.enabled:
                llm_items, llm_tokens = self.llm_parser.parse_with_llm(ambiguous_rows)
                line_items.extend(llm_items)
                
                # Update token usage
                for key in token_usage:
                    token_usage[key] += llm_tokens.get(key, 0)
                
                logger.info(f"LLM added {len(llm_items)} items")
            
            # Third pass: Additional heuristic extraction for remaining rows
            remaining_rows = [row for row in ambiguous_rows if not any(row in item.get('original_text', '') for item in line_items)]
            if remaining_rows:
                heuristic_items = self._heuristic_fallback(remaining_rows)
                line_items.extend(heuristic_items)
                logger.info(f"Heuristic fallback added {len(heuristic_items)} items")
            
            # Apply deduplication
            line_items = self.dedup_engine.remove_duplicates(line_items)
            
            # Convert to BillItem format
            bill_items = []
            for item in line_items:
                try:
                    bill_item = BillItem(
                        item_name=item['item_name'],
                        item_amount=round(float(item['item_amount']), 2),
                        item_quantity=round(float(item.get('item_quantity', 1.0)), 2),
                        item_rate=round(float(item.get('item_rate', 0.0)), 2)
                    )
                    bill_items.append(bill_item.dict())
                except Exception as e:
                    logger.warning(f"Failed to create BillItem: {e}")
                    continue
            
            logger.info(f"Page {page_num} processing completed. Final items: {len(bill_items)}")
            
            return {
                "page_no": str(page_num),
                "page_type": page_type,
                "bill_items": bill_items,
                "text_rows_processed": len(row_texts),
                "ambiguous_rows_sent_to_llm": len(ambiguous_rows) if self.llm_parser.enabled else 0
            }
            
        except Exception as e:
            logger.error(f"Page {page_num} processing failed: {str(e)}", exc_info=True)
            return {
                "page_no": str(page_num),
                "page_type": "Error",
                "bill_items": [],
                "text_rows_processed": 0,
                "ambiguous_rows_sent_to_llm": 0,
                "error": str(e)
            }
    
    def _heuristic_fallback(self, rows: List[str]) -> List[Dict]:
        """Final fallback for difficult rows"""
        items = []
        
        for row in rows:
            # Try to extract amount from end of string
            amount_match = re.search(r'([\d,]+\.\d{2})$', row.strip())
            if amount_match:
                amount = self.item_parser.parse_numeric_value(amount_match.group(1))
                if amount and amount > 0:
                    # Use everything before the amount as name
                    name = row[:amount_match.start()].strip()
                    name = self.item_parser._clean_item_name(name, row)
                    
                    if name and len(name) > 2:
                        items.append({
                            'item_name': name,
                            'item_amount': amount,
                            'item_quantity': 1.0,
                            'item_rate': 0.0,
                            'source': 'heuristic_fallback'
                        })
        
        return items

# ---------------------------
# FASTAPI APPLICATION (UNCHANGED)
# ---------------------------
app = FastAPI(
    title="Enhanced Bill Extractor API",
    description="Advanced bill line item extraction system with improved accuracy",
    version="2.0.0"
)

@app.middleware("http")
async def skip_ngrok_warning(request: Request, call_next):
    response = await call_next(request)
    response.headers["ngrok-skip-browser-warning"] = "true"
    return response

GEMINI_API_KEY = "AIzaSyAWvvhnvuivy-BcLhf026K9MIoLvqNdhlc"
extractor = BillExtractor(GEMINI_API_KEY)

@app.post("/extract-bill-data", response_model=ExtractionResponse)
async def extract_bill_data(request: ExtractRequest):
    """Main extraction endpoint"""
    logger.info(f"Received bill extraction request for URL: {request.document}")
    
    start_time = datetime.now()
    try:
        result = extractor.extract_from_document(request.document)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Request completed in {processing_time:.2f}s. Success: {result['is_success']}")
        return result
        
    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"API endpoint error after {processing_time:.2f}s: {str(e)}", exc_info=True)
        return {
            "is_success": False,
            "token_usage": {"total_tokens": 0, "input_tokens": 0, "output_tokens": 0},
            "data": None,
            "error": f"Internal server error: {str(e)}"
        }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "Enhanced Bill Extractor API",
        "version": "2.0.0"
    }
    
    dependencies = {
        "easyocr": "healthy",
        "pdf2image": "healthy", 
        "pillow": "healthy",
        "numpy": "healthy",
        "requests": "healthy",
        "opencv": "healthy"
    }
    
    try:
        test_reader = easyocr.Reader(['en'])
        dependencies["easyocr"] = "healthy"
    except Exception as e:
        dependencies["easyocr"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    health_status["dependencies"] = dependencies
    health_status["llm_status"] = "enabled" if extractor.llm_parser.enabled else "disabled"
    
    return health_status

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Enhanced Bill Extractor",
        "version": "2.0.0",
        "status": "operational",
        "endpoints": {
            "extract": "/extract-bill-data",
            "health": "/health"
        },
        "supported_formats": SUPPORTED_FORMATS
    }

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    debug = os.getenv("DEBUG", "false").lower() == "true"
    
    logger.info(f"Starting server on port {port}, debug: {debug}")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=debug,
        log_level="info",
        timeout_keep_alive=5
    )
