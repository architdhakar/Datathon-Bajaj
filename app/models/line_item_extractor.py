# app/models/line_item_extractor.py
import re
from typing import List, Dict
import spacy

class LineItemExtractor:
    def __init__(self):
        # Load spaCy model for NER
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            self.nlp = None
        
        # Regex patterns for medical items and amounts
        self.medical_patterns = [
            r'[A-Z][a-z]+\s*\d+\.?\d*\s*(mg|mgm|tab|cap|syp|inj|ml|g)',
            r'[A-Z][a-zA-Z]+\s*\d*\.?\d*\s*(Tablet|Capsule|Syrup|Injection)',
            r'[A-Z]{2,}\s*\d*'
        ]
        
        self.amount_pattern = r'₹?\s*(\d+[.,]?\d*\.?\d{0,2})'
        self.quantity_pattern = r'(\d+)\s*(x|X|\*|@|nos?|pcs?)'
    
    async def extract_line_items(self, text_blocks: List[Dict]) -> List[Dict]:
        """Extract line items from text blocks"""
        line_items = []
        current_item = {}
        
        for block in text_blocks:
            text = block['text'].strip()
            
            # Skip headers, footers, totals
            if self._is_header_footer(text) or self._is_total_line(text):
                continue
                
            # Check if this looks like a medical item
            if self._is_medical_item(text):
                # If we have a current item, save it
                if current_item:
                    if self._validate_item(current_item):
                        line_items.append(current_item)
                    current_item = {}
                
                current_item = self._parse_medical_item(text, block)
            
            # If we're building an item, add additional information
            elif current_item and self._is_amount_or_quantity(text):
                self._enhance_item(current_item, text)
        
        # Don't forget the last item
        if current_item and self._validate_item(current_item):
            line_items.append(current_item)
            
        return self._deduplicate_items(line_items)
    
    def _is_medical_item(self, text: str) -> bool:
        """Check if text resembles a medical line item"""
        text_lower = text.lower()
        
        # Common medical keywords
        medical_keywords = ['tab', 'cap', 'syp', 'inj', 'mg', 'gm', 'ml', 
                           'tablet', 'capsule', 'syrup', 'injection', 'cream']
        
        if any(keyword in text_lower for keyword in medical_keywords):
            return True
            
        # Check regex patterns
        for pattern in self.medical_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
                
        return False
    
    def _parse_medical_item(self, text: str, block: Dict) -> Dict:
        """Parse medical item details"""
        item = {
            'item_name': self._extract_item_name(text),
            'item_amount': self._extract_amount(text),
            'item_rate': self._extract_rate(text),
            'item_quantity': self._extract_quantity(text),
            'raw_text': text,
            'bbox': block['bbox']
        }
        return item
    
    def _extract_item_name(self, text: str) -> str:
        """Extract cleaned item name"""
        # Remove amounts, quantities, rates
        cleaned = re.sub(self.amount_pattern, '', text)
        cleaned = re.sub(self.quantity_pattern, '', cleaned)
        cleaned = re.sub(r'₹|Rs?\.?\s*', '', cleaned)
        return cleaned.strip()
    
    def _extract_amount(self, text: str) -> float:
        """Extract item amount"""
        amounts = re.findall(self.amount_pattern, text)
        if amounts:
            # Typically the last amount is the total for the line
            return float(amounts[-1].replace(',', ''))
        return 0.0
    
    def _extract_quantity(self, text: str) -> float:
        """Extract quantity"""
        quantities = re.findall(r'(\d+)\s*(x|X|\*)?', text)
        if quantities:
            return float(quantities[0][0])
        return 1.0  # Default quantity
    
    def _extract_rate(self, text: str) -> float:
        """Extract rate/price per unit"""
        amounts = re.findall(self.amount_pattern, text)
        if len(amounts) >= 2:
            # Second amount might be the rate
            return float(amounts[-2].replace(',', ''))
        return 0.0