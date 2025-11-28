# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from typing import List, Optional
# import requests
# from PIL import Image
# from io import BytesIO
# import base64
# import json
# import re
# import os

# app = FastAPI(title="Bill Data Extraction API")

# class BillRequest(BaseModel):
#     document: str

# class BillItem(BaseModel):
#     item_name: str
#     item_amount: float
#     item_rate: float
#     item_quantity: int

# class PageWiseLineItems(BaseModel):
#     page_no: str
#     bill_items: List[BillItem]

# class BillData(BaseModel):
#     pagewise_line_items: List[PageWiseLineItems]
#     total_item_count: int
#     reconciled_amount: float

# class BillResponse(BaseModel):
#     is_success: bool
#     data: Optional[BillData] = None
#     error: Optional[str] = None

# def download_image(url: str) -> Image.Image:
#     """Download image from URL"""
#     try:
#         response = requests.get(url, timeout=30)
#         response.raise_for_status()
#         return Image.open(BytesIO(response.content))
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Failed to download image: {str(e)}")

# def image_to_base64(image: Image.Image) -> str:
#     """Convert PIL Image to base64 string"""
#     buffered = BytesIO()
#     # Resize if too large
#     if image.width > 2000 or image.height > 2000:
#         image.thumbnail((2000, 2000), Image.Resampling.LANCZOS)
#     image.save(buffered, format="PNG")
#     return base64.b64encode(buffered.getvalue()).decode()

# def extract_with_huggingface(image_base64: str) -> dict:
#     """Extract bill data using Hugging Face Inference API (Free)"""
    
#     # Using BLIP-2 for image captioning and then GPT-2 for structure
#     # Or use Salesforce BLIP for OCR + understanding
    
#     try:
#         # Method 1: Try Hugging Face Inference API with vision model
#         HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")  # Optional, works without too
        
#         headers = {}
#         if HF_API_TOKEN:
#             headers["Authorization"] = f"Bearer {HF_API_TOKEN}"
        
#         # Using Microsoft's Florence-2 (free inference API)
#         api_url = "https://api-inference.huggingface.co/models/microsoft/Florence-2-large"
        
#         image_bytes = base64.b64decode(image_base64)
        
#         response = requests.post(
#             api_url,
#             headers=headers,
#             data=image_bytes,
#             timeout=30
#         )
        
#         if response.status_code == 200:
#             result = response.json()
#             # Process OCR result
#             return parse_ocr_to_structure(result)
            
#     except Exception as e:
#         print(f"HF API error: {e}")
    
#     # Fallback: Use Google's Gemini (free tier)
#     try:
#         return extract_with_gemini(image_base64)
#     except:
#         pass
    
#     # Last resort: Pattern-based extraction
#     return extract_with_ocr_space(image_base64)

# def extract_with_gemini(image_base64: str) -> dict:
#     """Extract using Google Gemini Flash (Free tier)"""
    
#     GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    
#     if not GEMINI_API_KEY:
#         raise Exception("GEMINI_API_KEY not set")
    
#     prompt = """Extract all line items from this medical bill/invoice.

# For each item, extract:
# - item_name: The medicine/item name
# - item_quantity: Number of units
# - item_rate: Price per unit
# - item_amount: Total amount for that item

# Return ONLY valid JSON in this format (no markdown, no explanation):
# {
#   "page_no": "1",
#   "bill_items": [
#     {"item_name": "Medicine Name", "item_quantity": 10, "item_rate": 50.0, "item_amount": 500.0}
#   ],
#   "total_amount": 500.0
# }

# IMPORTANT: 
# - Extract ALL items, don't miss any
# - Don't double count
# - If quantity not visible, use 1
# - Sum individual amounts for total"""

#     try:
#         response = requests.post(
#             f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}",
#             json={
#                 "contents": [{
#                     "parts": [
#                         {"text": prompt},
#                         {
#                             "inline_data": {
#                                 "mime_type": "image/png",
#                                 "data": image_base64
#                             }
#                         }
#                     ]
#                 }],
#                 "generationConfig": {
#                     "temperature": 0.1,
#                     "topK": 32,
#                     "topP": 1,
#                     "maxOutputTokens": 2048
#                 }
#             },
#             timeout=30
#         )
        
#         if response.status_code == 200:
#             result = response.json()
#             text = result["candidates"][0]["content"]["parts"][0]["text"]
            
#             # Clean JSON
#             text = text.strip()
#             text = re.sub(r'^```json\s*', '', text)
#             text = re.sub(r'\s*```$', '', text)
#             text = text.strip()
            
#             return json.loads(text)
#     except Exception as e:
#         print(f"Gemini error: {e}")
#         raise

# def extract_with_ocr_space(image_base64: str) -> dict:
#     """Fallback: OCR.space free API"""
    
#     try:
#         image_bytes = base64.b64decode(image_base64)
        
#         response = requests.post(
#             'https://api.ocr.space/parse/image',
#             files={'file': ('bill.png', image_bytes, 'image/png')},
#             data={
#                 'apikey': 'helloworld',  # Free tier key
#                 'language': 'eng',
#                 'isTable': True,
#                 'OCREngine': 2
#             },
#             timeout=30
#         )
        
#         if response.status_code == 200:
#             result = response.json()
#             text = result.get('ParsedResults', [{}])[0].get('ParsedText', '')
#             return parse_text_to_structure(text)
#     except Exception as e:
#         print(f"OCR.space error: {e}")
    
#     # Return empty structure
#     return {
#         "page_no": "1",
#         "bill_items": [],
#         "total_amount": 0.0
#     }

# def parse_ocr_to_structure(ocr_result) -> dict:
#     """Parse OCR result to bill structure"""
#     # Implement parsing logic based on OCR output
#     return {
#         "page_no": "1",
#         "bill_items": [],
#         "total_amount": 0.0
#     }

# def parse_text_to_structure(text: str) -> dict:
#     """Parse plain text to bill structure using pattern matching"""
    
#     bill_items = []
#     lines = text.split('\n')
    
#     # Common patterns for bill items
#     # Pattern: Item name | quantity | rate | amount
#     item_pattern = r'([A-Za-z0-9\s\-\.]+)\s+(\d+)\s+(\d+\.?\d*)\s+(\d+\.?\d*)'
    
#     for line in lines:
#         match = re.search(item_pattern, line)
#         if match:
#             item_name = match.group(1).strip()
#             quantity = int(match.group(2))
#             rate = float(match.group(3))
#             amount = float(match.group(4))
            
#             if amount > 0:  # Valid item
#                 bill_items.append({
#                     "item_name": item_name,
#                     "item_quantity": quantity,
#                     "item_rate": rate,
#                     "item_amount": amount
#                 })
    
#     total = sum(item["item_amount"] for item in bill_items)
    
#     return {
#         "page_no": "1",
#         "bill_items": bill_items,
#         "total_amount": total
#     }

# def parse_extracted_data(extracted_data: dict) -> BillData:
#     """Parse extracted data into response format"""
    
#     bill_items = []
#     for item in extracted_data.get("bill_items", []):
#         bill_items.append(BillItem(
#             item_name=str(item.get("item_name", "Unknown")).strip(),
#             item_amount=float(item.get("item_amount", 0)),
#             item_rate=float(item.get("item_rate", 0)),
#             item_quantity=int(item.get("item_quantity", 1))
#         ))
    
#     # Calculate reconciled amount (sum of all item amounts)
#     reconciled_amount = sum(item.item_amount for item in bill_items)
    
#     pagewise_data = PageWiseLineItems(
#         page_no=str(extracted_data.get("page_no", "1")),
#         bill_items=bill_items
#     )
    
#     return BillData(
#         pagewise_line_items=[pagewise_data],
#         total_item_count=len(bill_items),
#         reconciled_amount=round(reconciled_amount, 2)
#     )

# @app.post("/extract-bill-data", response_model=BillResponse)
# async def extract_bill_data(request: BillRequest):
#     """
#     Extract line items and amounts from bill/invoice images
#     """
#     try:
#         # Download image
#         image = download_image(request.document)
        
#         # Convert to base64
#         image_base64 = image_to_base64(image)
        
#         # Try extraction methods in order of preference
#         extracted_data = None
        
#         # Method 1: Gemini (most accurate for structured data)
#         try:
#             extracted_data = extract_with_gemini(image_base64)
#         except Exception as e:
#             print(f"Gemini failed: {e}")
            
#             # Method 2: Hugging Face
#             try:
#                 extracted_data = extract_with_huggingface(image_base64)
#             except Exception as e2:
#                 print(f"HuggingFace failed: {e2}")
                
#                 # Method 3: OCR.space fallback
#                 extracted_data = extract_with_ocr_space(image_base64)
        
#         # Parse into response format
#         bill_data = parse_extracted_data(extracted_data)
        
#         return BillResponse(
#             is_success=True,
#             data=bill_data
#         )
        
#     except Exception as e:
#         return BillResponse(
#             is_success=False,
#             error=str(e)
#         )

# @app.get("/")
# async def root():
#     return {
#         "message": "Bill Data Extraction API",
#         "endpoint": "/extract-bill-data",
#         "method": "POST",
#         "status": "ready"
#     }

# @app.get("/health")
# async def health_check():
#     return {"status": "healthy"}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)


# version 2-------------------------
# import os
# import re
# import json
# from typing import List, Dict, Tuple, Optional
# from fastapi import FastAPI
# from pydantic import BaseModel
# import requests
# from io import BytesIO
# from PIL import Image
# from collections import defaultdict
# from difflib import SequenceMatcher

# from paddleocr import PaddleOCR


# app = FastAPI(title="Bajaj Bill Extractor")

# # ---------------------------
# # Config
# # ---------------------------
# LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")

# GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
# GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
# GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

# ocr = PaddleOCR(use_angle_cls=True, lang="en")

# # ---------------------------
# # Input model
# # ---------------------------
# class ExtractRequest(BaseModel):
#     document: str   # URL to the bill image


# # -----------------------------------------------------------
# # UTILITIES
# # -----------------------------------------------------------

# def download_image(url: str) -> Image.Image:
#     resp = requests.get(url, timeout=30)
#     resp.raise_for_status()
#     return Image.open(BytesIO(resp.content)).convert("RGB")


# def box_center(box):
#     xs = [p[0] for p in box]
#     ys = [p[1] for p in box]
#     return sum(xs)/4, sum(ys)/4


# def ocr_page(img: Image.Image):
#     import numpy as np
#     arr = np.array(img)
#     out = ocr.ocr(arr)
#     flat = []

#     if not out:
#         return []

#     for block in out:
#         if not block: 
#             continue
#         for line in block:
#             if len(line) < 2:
#                 continue

#             box = line[0]
#             text, conf = None, None

#             if isinstance(line[1], tuple):
#                 text, conf = line[1]
#             elif isinstance(line[1], list):
#                 text, conf = line[1][0], line[1][1]
#             else:
#                 continue

#             flat.append((box, text, conf))

#     return flat


# def cluster_rows(ocr_items, y_tol=12):
#     if not ocr_items:
#         return []

#     items = []
#     for box, text, score in ocr_items:
#         cx, cy = box_center(box)
#         items.append((cy, cx, text, box, score))

#     items.sort()

#     rows, cur_row = [], []
#     last_y = None

#     for cy, cx, text, box, score in items:
#         if last_y is None or abs(cy - last_y) <= y_tol:
#             cur_row.append((cx, text, box, score))
#         else:
#             cur_row.sort()
#             rows.append(cur_row)
#             cur_row = [(cx, text, box, score)]
#         last_y = cy

#     if cur_row:
#         cur_row.sort()
#         rows.append(cur_row)

#     final = []
#     for row in rows:
#         texts = [t[1] for t in row]
#         final.append({"text": " ".join(texts)})

#     return final


# # ----------------------------------------------
# # Regex parsing
# # ----------------------------------------------
# LINE_PATTERNS = [
#     r"^(.+?)\s+(\d+\.?\d*)\s+([\d\.]+)\s+([\d\.]+)$",
#     r"^(.+?)\s+([\d\.]+)$",
# ]


# def try_parse_row(text):
#     t = text.strip().replace(",", "")

#     for pat in LINE_PATTERNS:
#         m = re.match(pat, t)
#         if m:
#             g = m.groups()
#             if len(g) == 4:
#                 return {
#                     "item_name": g[0].strip(),
#                     "item_quantity": float(g[1]),
#                     "item_rate": float(g[2]),
#                     "item_amount": float(g[3]),
#                     "src": "regex"
#                 }
#             if len(g) == 2:
#                 return {
#                     "item_name": g[0].strip(),
#                     "item_quantity": None,
#                     "item_rate": None,
#                     "item_amount": float(g[1]),
#                     "src": "regex_partial"
#                 }
#     return None


# # ----------------------------------------------
# # LLM FALLBACK (GROQ)
# # ----------------------------------------------
# def llm_parse_rows(rows: List[str]) -> List[Dict]:
#     if not rows:
#         return []

#     prompt = f"""
# You are an expert parser for pharmacy bills.
# Extract only actual bill line items from the following OCR lines.

# Return ONLY a JSON array.
# Each object MUST contain:
# - item_name (string)
# - item_quantity (number or null)
# - item_rate (number or null)
# - item_amount (number)

# Ignore totals, GST, headings.

# LINES:
# {json.dumps(rows, indent=2)}

# Return ONLY JSON array:
# """

#     headers = {
#         "Authorization": f"Bearer {GROQ_API_KEY}",
#         "Content-Type": "application/json"
#     }

#     payload = {
#         "model": GROQ_MODEL,
#         "messages": [{"role": "user", "content": prompt}],
#         "temperature": 0
#     }

#     r = requests.post(GROQ_ENDPOINT, json=payload, headers=headers, timeout=60)
#     r.raise_for_status()

#     text = r.json()["choices"][0]["message"]["content"]

#     # Extract JSON safely
#     try:
#         return json.loads(text)
#     except:
#         m = re.search(r"\[.*\]", text, re.S)
#         if m:
#             return json.loads(m.group())
#         return []


# # ----------------------------------------------
# # Deduplication
# # ----------------------------------------------
# def string_sim(a, b):
#     return SequenceMatcher(None, a, b).ratio()


# def dedupe(items):
#     final = []
#     for it in items:
#         nm = it["item_name"].lower()
#         amt = it["item_amount"]

#         merged = False
#         for f in final:
#             if string_sim(nm, f["item_name"].lower()) > 0.90:
#                 merged = True
#                 if len(it["item_name"]) > len(f["item_name"]):
#                     f["item_name"] = it["item_name"]
#                 if not f["item_quantity"]:
#                     f["item_quantity"] = it["item_quantity"]
#                 if not f["item_rate"]:
#                     f["item_rate"] = it["item_rate"]
#                 f["item_amount"] = (f["item_amount"] + amt) / 2
#                 break

#         if not merged:
#             final.append(it)

#     return final


# # ==================================================================
# # MAIN API
# # ==================================================================
# @app.post("/extract-bill-data")
# def extract_bill_data(req: ExtractRequest):
#     try:
#         img = download_image(req.document)
#         ocr_raw = ocr_page(img)
#         rows = cluster_rows(ocr_raw)

#         items = []
#         ambiguous = []

#         for r in rows:
#             line = r["text"]
#             if re.search(r"total|gst|invoice|bill|summary", line, re.I):
#                 continue

#             parsed = try_parse_row(line)
#             if parsed:
#                 items.append(parsed)
#             else:
#                 ambiguous.append(line)

#         # LLM fallback
#         if ambiguous:
#             llm_items = llm_parse_rows(ambiguous)
#             items.extend(llm_items)

#         items = dedupe(items)

#         total_amt = round(sum(i["item_amount"] for i in items), 2)

#         return {
#             "is_success": True,
#             "data": {
#                 "items": items,
#                 "total_item_count": len(items),
#                 "reconciled_amount": total_amt
#             }
#         }

#     except Exception as e:
#         return {
#             "is_success": False,
#             "error": str(e)
#         }


# @app.get("/")
# def home():
#     return {"message": "Bajaj Bill Extractor Running"}


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("main:app", port=8000, host="0.0.0.0")


# version 3

import os
import re
import json
from typing import List, Dict, Tuple, Optional
from fastapi import FastAPI
from pydantic import BaseModel
import requests
from io import BytesIO
from PIL import Image
from difflib import SequenceMatcher
import numpy as np
import easyocr

# ---------------------------
# CONFIG
# ---------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

# OCR init
ocr = easyocr.Reader(['en'], gpu=False)

app = FastAPI(title="Bajaj Bill Extractor")

# ---------------------------
# Input model
# ---------------------------
class ExtractRequest(BaseModel):
    document: str  # URL to the bill image

# ---------------------------
# UTILITIES
# ---------------------------
def download_image(url: str) -> Image.Image:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content)).convert("RGB")


def box_center(box):
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    return sum(xs)/4, sum(ys)/4


def ocr_page(img: Image.Image):
    arr = np.array(img)
    results = ocr.readtext(arr, detail=1)
    flat = []
    for box, text, conf in results:
        flat.append((box, text, conf))
    return flat


def cluster_rows(ocr_items, y_tol=12):
    if not ocr_items:
        return []
    items = []
    for box, text, score in ocr_items:
        cx, cy = box_center(box)
        items.append((cy, cx, text, box, score))
    items.sort()
    rows, cur_row = [], []
    last_y = None
    for cy, cx, text, box, score in items:
        if last_y is None or abs(cy - last_y) <= y_tol:
            cur_row.append((cx, text, box, score))
        else:
            cur_row.sort()
            rows.append(cur_row)
            cur_row = [(cx, text, box, score)]
        last_y = cy
    if cur_row:
        cur_row.sort()
        rows.append(cur_row)
    final = []
    for row in rows:
        texts = [t[1] for t in row]
        final.append({"text": " ".join(texts)})
    return final


# ---------------------------
# Regex parsing
# ---------------------------
LINE_PATTERNS = [
    r"^(.+?)\s+(\d+\.?\d*)\s+([\d\.]+)\s+([\d\.]+)$",
    r"^(.+?)\s+([\d\.]+)$",
]

def try_parse_row(text):
    t = text.strip().replace(",", "")
    for pat in LINE_PATTERNS:
        m = re.match(pat, t)
        if m:
            g = m.groups()
            if len(g) == 4:
                return {
                    "item_name": g[0].strip(),
                    "item_quantity": float(g[1]),
                    "item_rate": float(g[2]),
                    "item_amount": float(g[3]),
                    "src": "regex"
                }
            if len(g) == 2:
                return {
                    "item_name": g[0].strip(),
                    "item_quantity": None,
                    "item_rate": None,
                    "item_amount": float(g[1]),
                    "src": "regex_partial"
                }
    return None


# ---------------------------
# LLM FALLBACK (GROQ)
# ---------------------------
def llm_parse_rows(rows: List[str]) -> List[Dict]:
    if not rows or not GROQ_API_KEY:
        return []
    prompt = f"""
You are an expert parser for pharmacy/bill line items.
Extract actual bill line items ONLY. Return JSON array ONLY.
Each object must have:
- item_name (string)
- item_quantity (number or null)
- item_rate (number or null)
- item_amount (number)

Ignore totals, GST, headers.

LINES:
{json.dumps(rows, indent=2)}

Return only JSON array:
"""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0
    }
    try:
        r = requests.post(GROQ_ENDPOINT, json=payload, headers=headers, timeout=60)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]
        try:
            return json.loads(text)
        except:
            m = re.search(r"\[.*\]", text, re.S)
            if m:
                return json.loads(m.group())
            return []
    except:
        return []


# ---------------------------
# Deduplication
# ---------------------------
def string_sim(a, b):
    return SequenceMatcher(None, a, b).ratio()

def dedupe(items):
    final = []
    for it in items:
        nm = it["item_name"].lower()
        amt = it["item_amount"]
        merged = False
        for f in final:
            if string_sim(nm, f["item_name"].lower()) > 0.90:
                merged = True
                if len(it["item_name"]) > len(f["item_name"]):
                    f["item_name"] = it["item_name"]
                if not f["item_quantity"]:
                    f["item_quantity"] = it["item_quantity"]
                if not f["item_rate"]:
                    f["item_rate"] = it["item_rate"]
                f["item_amount"] = round((f["item_amount"] + amt)/2,2)
                break
        if not merged:
            final.append(it)
    return final


# ---------------------------
# Page type classifier
# ---------------------------
def classify_page_type(rows_text: List[str]) -> str:
    text = " ".join(rows_text).lower()
    if "pharmacy" in text or "medicine" in text:
        return "Pharmacy"
    if "total" in text or "grand total" in text or "net payable" in text:
        return "Final Bill"
    return "Bill Detail"


# ---------------------------
# MAIN ENDPOINT
# ---------------------------
@app.post("/extract-bill-data")
def extract_bill_data(req: ExtractRequest):
    try:
        img = download_image(req.document)
        ocr_raw = ocr_page(img)
        rows = cluster_rows(ocr_raw)
        rows_text = [r["text"] for r in rows]

        page_type = classify_page_type(rows_text)

        items, ambiguous = [], []

        for r in rows:
            line = r["text"]
            if re.search(r"total|gst|invoice|bill|summary", line, re.I):
                continue
            parsed = try_parse_row(line)
            if parsed:
                items.append(parsed)
            else:
                ambiguous.append(line)

        # LLM fallback
        if ambiguous:
            items.extend(llm_parse_rows(ambiguous))

        items = dedupe(items)
        total_amt = round(sum(i["item_amount"] for i in items), 2)

        return {
            "is_success": True,
            "token_usage": {"total_tokens": None, "input_tokens": None, "output_tokens": None},
            "data": {
                "pagewise_line_items": [
                    {
                        "page_no": "1",
                        "page_type": page_type,
                        "bill_items": items
                    }
                ],
                "total_item_count": len(items),
                "reconciled_amount": total_amt
            }
        }
    except Exception as e:
        import traceback
        return {"is_success": False, "error": str(e), "traceback": traceback.format_exc()}


@app.get("/")
def home():
    return {"message": "Bajaj Bill Extractor Running"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
