# app/models/reconciliation_engine.py
from typing import List, Dict
import numpy as np

class ReconciliationEngine:
    def __init__(self):
        self.tolerance = 0.01  # 1% tolerance for amount matching
    
    def reconcile_totals(self, line_items: List[Dict], extracted_total: float = None) -> Dict:
        """Reconcile extracted amounts and prevent double-counting"""
        
        # Remove duplicates and invalid items
        cleaned_items = self._remove_duplicates(line_items)
        
        # Calculate AI total
        ai_total = sum(item.get('item_amount', 0) for item in cleaned_items)
        
        # If we have an extracted total, check reconciliation
        if extracted_total and extracted_total > 0:
            reconciliation_status = self._check_reconciliation(ai_total, extracted_total)
        else:
            reconciliation_status = "no_total_found"
        
        return {
            "pagewise_line_items": [{
                "page_no": "1",  # You'll need to handle multiple pages
                "bill_items": cleaned_items
            }],
            "total_item_count": len(cleaned_items),
            "reconciled_amount": round(ai_total, 2),
            "reconciliation_status": reconciliation_status,
            "extracted_total": extracted_total,
            "ai_calculated_total": round(ai_total, 2)
        }
    
    def _remove_duplicates(self, items: List[Dict]) -> List[Dict]:
        """Remove duplicate line items using multiple strategies"""
        unique_items = []
        seen_patterns = set()
        
        for item in items:
            # Create a signature for this item
            signature = self._create_item_signature(item)
            
            if signature not in seen_patterns:
                seen_patterns.add(signature)
                unique_items.append(item)
            else:
                print(f"Removing duplicate: {item.get('item_name', 'Unknown')}")
                
        return unique_items
    
    def _create_item_signature(self, item: Dict) -> str:
        """Create unique signature for item deduplication"""
        name = item.get('item_name', '').lower().strip()
        amount = round(item.get('item_amount', 0), 2)
        quantity = item.get('item_quantity', 1)
        
        # Normalize name
        name = re.sub(r'\s+', ' ', name)
        
        return f"{name}_{quantity}_{amount}"
    
    def _check_reconciliation(self, ai_total: float, extracted_total: float) -> str:
        """Check if AI total matches extracted total"""
        difference = abs(ai_total - extracted_total)
        relative_diff = difference / extracted_total if extracted_total > 0 else 1
        
        if relative_diff <= self.tolerance:
            return "perfect_match"
        elif relative_diff <= 0.05:  # 5% tolerance
            return "close_match"
        else:
            return "mismatch"