# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import cv2
import numpy as np
from typing import List, Dict, Optional
import json

app = FastAPI(title="Bajaj Health Bill Extraction API")

class BillItem(BaseModel):
    item_name: str
    item_amount: float
    item_rate: float
    item_quantity: float

class PageData(BaseModel):
    page_no: str
    bill_items: List[BillItem]

class ExtractionResponse(BaseModel):
    is_success: bool
    data: Optional[Dict]
    error: Optional[str]

class DocumentRequest(BaseModel):
    document: str

@app.post("/extract-bill-data", response_model=ExtractionResponse)
async def extract_bill_data(request: DocumentRequest):
    try:
        # Download and process document
        image = download_image(request.document)
        
        # Process document through pipeline
        result = await process_document_pipeline(image)
        
        return ExtractionResponse(
            is_success=True,
            data=result
        )
        
    except Exception as e:
        return ExtractionResponse(
            is_success=False,
            error=str(e)
        )

async def process_document_pipeline(image):
    """
    Main processing pipeline
    """
    # 1. Pre-process image
    processed_image = preprocess_image(image)
    
    # 2. Extract text and layout
    text_blocks = await extract_text_and_layout(processed_image)
    
    # 3. Identify line items
    line_items = await extract_line_items(text_blocks)
    
    # 4. Reconcile totals
    reconciled_data = reconcile_totals(line_items)
    
    return reconciled_data