# app/models/layout_analyzer.py
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
import torch
from PIL import Image
import numpy as np

class LayoutAnalyzer:
    def __init__(self):
        self.processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
        self.model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base")
        
    async def analyze_document_layout(self, image):
        """Extract text blocks with coordinates"""
        try:
            # Convert to PIL Image if needed
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            # Process through LayoutLMv3
            encoding = self.processor(image, return_tensors="pt")
            outputs = self.model(**encoding)
            predictions = torch.argmax(outputs.logits, dim=-1)[0].tolist()
            
            # Extract text blocks with bounding boxes
            text_blocks = self._extract_text_blocks(encoding, predictions)
            return text_blocks
            
        except Exception as e:
            print(f"Layout analysis error: {e}")
            return await self.fallback_ocr(image)
    
    def _extract_text_blocks(self, encoding, predictions):
        """Extract structured text blocks from model output"""
        tokens = encoding.tokens()
        bboxes = encoding.bboxes[0]
        
        text_blocks = []
        current_block = []
        
        for i, (token, bbox, pred) in enumerate(zip(tokens, bboxes, predictions)):
            if token in ["[CLS]", "[SEP]", "[PAD]"]:
                continue
                
            # Group tokens into blocks based on proximity
            if current_block and not self._is_same_line(bbox, current_block[-1]['bbox']):
                text_blocks.append(self._merge_block(current_block))
                current_block = []
                
            current_block.append({
                'text': token,
                'bbox': bbox,
                'label': pred
            })
        
        if current_block:
            text_blocks.append(self._merge_block(current_block))
            
        return text_blocks
    
    async def fallback_ocr(self, image):
        """Fallback to traditional OCR"""
        import pytesseract
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        
        text_blocks = []
        for i in range(len(data['text'])):
            if data['text'][i].strip():
                text_blocks.append({
                    'text': data['text'][i],
                    'bbox': [data['left'][i], data['top'][i], 
                            data['left'][i] + data['width'][i], 
                            data['top'][i] + data['height'][i]],
                    'confidence': data['conf'][i]
                })
        return text_blocks