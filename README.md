# Datathon-Bajaj
## Medical Bill Line Item Extractor

An advanced AI-powered pipeline for automatically extracting line items from medical bills, pharmacy invoices, and hospital documents with high accuracy.

##  Features

- **Multi-Format Support**: PDF, PNG, JPG, JPEG
- **Advanced OCR**: EasyOCR with image preprocessing
- **Intelligent Parsing**: Regex + LLM (Gemini) hybrid approach
- **Medical Domain Specialized**: Optimized for hospital & pharmacy bills
- **Duplicate Detection**: Smart similarity-based deduplication
- **REST API**: FastAPI with comprehensive endpoints

##  Installation

```bash
# Clone repository
git clone https://github.com/yourusername/medical-bill-extractor.git
cd medical-bill-extractor

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export GEMINI_API_KEY="your_gemini_api_key"
export LOG_LEVEL="INFO"

uvicorn main:app --host 0.0.0.0 --port 8000

curl -X POST "http://localhost:8000/extract-bill-data" \
  -H "Content-Type: application/json" \
  -d '{"document": "https://example.com/medical_bill.pdf"}'


## Core Components

### 1. **Document Processing & Preprocessing**
- **Format Detection**: Automatic PDF/Image recognition from URLs
- **Image Enhancement**: OpenCV-based denoising & contrast optimization (CLAHE)
- **PDF Conversion**: High-resolution (300 DPI) PDF to image conversion

### 2. **Advanced OCR Engine**
- **Technology**: EasyOCR with custom configuration
- **Optimizations**: Dynamic image resizing, beam search decoding
- **Output**: Bounding boxes with text and confidence scores

### 3. **Intelligent Text Reconstruction**
- **Row Clustering**: Dynamic Y-tolerance algorithm for line detection
- **Spatial Analysis**: X-coordinate based word spacing and column detection
- **Text Cleaning**: Specialized medical term preservation

### 4. **Multi-Stage Parsing Strategy**

#### **First Pass: Regex Patterns**
- 12+ specialized patterns for medical billing formats
- Handles: `Item Qty Rate Amount`, `Item @ Rate`, pharmacy formats
- Domain-specific: Hospital charges, lab tests, medicine items

#### **Second Pass: LLM Assistance (Gemini)**
- Processes ambiguous rows failed by regex
- Medical-domain fine-tuned prompts
- Extracts items from complex, unstructured text

#### **Third Pass: Heuristic Fallback**
- Amount extraction from string endings
- Position-based column identification
- Final attempt for difficult cases

### 5. **Post-Processing & Validation**
- **Deduplication**: Text similarity analysis (85% threshold)
- **Sanity Checks**: Amount validation, negative value filtering
- **Structure Enforcement**: Pydantic schema validation

## Key Innovations

### **Medical Domain Specialization**
- Hospital bill patterns: `"Consultation Charge DR NAME- Qty Rate Amount"`
- Pharmacy formats: `"Medicine Name Pack Size Qty Amount"`
- Lab test recognition: `"Liver Function Test(LFT) 1 400.00 400.00"`

### **Adaptive Processing**
- Dynamic Y-tolerance based on text height distribution
- Confidence thresholding (30% minimum)
- Multi-format bill type detection

### **Hybrid AI Approach**
```python
# Cascade parsing strategy
def parse_line_item(text):
    if regex_patterns(text): return regex_result     # Fast & accurate
    elif llm_available: return llm_parse(text)       # Intelligent fallback  
    else: return heuristic_extract(text)             # Final attempt



##Architecture
Input → Document Processing → OCR → Text Processing → Multi-Stage Parsing → Output
          ↓           ↓           ↓           ↓           ↓
        PDF/Image  EasyOCR    Row        Regex+LLM   Structured
        Loading  + Preprocess Clustering + Heuristic   JSON

##Acknowledgments
EasyOCR for text detection

Google Gemini for advanced parsing

PDF2Image for document conversion

FastAPI for robust API framework
