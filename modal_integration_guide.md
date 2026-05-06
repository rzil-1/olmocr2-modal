# Integration Guide: Modal Serverless OCR Engine to Evaluation Pipeline

Welcome to the OCR module! This document outlines how to safely integrate the Modal serverless OCR script (`modal_diagram_extractor.py`) into your MVP backend, and how to pipe its data directly into your Evaluation Engine.

## 1. Connecting the Frontend to the OCR Engine
When the user uploads a PDF via the frontend, your backend must trigger the Modal cloud GPU to process it. Because this is serverless, your backend can be a lightweight, cheap CPU server.

**Option A (Direct Modal API Import - Recommended):**
First, deploy the app permanently to the cloud by running: `modal deploy modal_diagram_extractor.py`. 
Then, inside your backend code:
```python
import modal

# 1. Connect to the deployed serverless function
ocr_engine = modal.Function.lookup("olmocr-diagram-extractor", "OLMOCREngine.process_document")

# 2. Read the user's uploaded PDF as raw bytes
with open("temp_uploaded_file.pdf", "rb") as pdf:
    pdf_bytes = pdf.read()
    
# 3. Call the remote GPU function!
# This beams the bytes up to the A10G and returns the structured data and image bytes directly into python memory!
results, cropped_images = ocr_engine.remote(pdf_bytes, "temp_uploaded_file.pdf")
```

**Option B (Subprocess Execution):**
If you prefer running via the command line interface:
`modal run modal_diagram_extractor.py --pdf-path temp_uploaded_file.pdf`

*(Note: The Modal container takes ~30-40 seconds to process a document. You should run this asynchronously via a task queue like Celery or BullMQ so it doesn't block the frontend API response).*

## 2. Connecting the OCR Engine to the Evaluation Engine
Whether you use Option A or B, the output is a highly structured dictionary. 

Your Evaluation Engine should read this schema:
```json
{
  "temp_uploaded_file.pdf": {
    "raw_markdown": "...full text...",
    "equations": ["y=x^2", "..."],
    "diagrams": [{"alt_text": "...", "coordinates": [...], "image_path": "..."}]
  }
}
```
- **Text Grading:** Pass the `raw_markdown` field to your LLM evaluator (e.g., DeepEval) to score the text against the ground truth.
- **Math Grading:** The `equations` array isolates all LaTeX formulas. You can use this to quickly verify mathematical correctness without parsing the full text.
- **Diagrams:** Send the image bytes (or the saved local paths) back to the frontend to display the neatly cropped charts to the user.

---

## 3. Safe Customization (What you CAN change)
You are free to modify the following parts of `modal_diagram_extractor.py` to fit your backend architecture:
- **Cloud Volumes & Object Storage:** You can change how `cropped_images` are saved. Even better, you can mount an AWS S3 bucket directly into the Modal environment so the cloud GPU saves images directly to your bucket!
- **GPU Tier:** You can change `@app.cls(gpu="A10G")` to `gpu="L4"` or `gpu="A100"` depending on your budget and speed requirements.
- **Local Entrypoint:** You can completely delete the `@app.local_entrypoint()` block at the bottom of the script if you are only importing the `OLMOCREngine` class directly via your backend.

## 4. Danger Zone (What you SHOULD NOT change)
To ensure the OCR remains highly accurate and can successfully detect and crop diagrams, **do not touch** the following logic:
- **The Prompt:** The prompt string is fine-tuned to trigger the Qwen2.5-VL model to output the precise markdown image syntax needed for automatic cropping.
- **Model Loading (`hf_transfer`):** Leave `os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"` intact. This official Rust downloader prevents Hugging Face rate limits and deadlocks when building the container.
- **Image Processing (`fitz.Matrix(2, 2)`):** PyMuPDF is specifically tuned to give the vision model enough resolution to read tiny math subscripts without crashing the GPU memory.
- **The Regex Pattern:** `pattern = r'!\[(.*?)\]\((page|\d+)_(\d+)_(\d+)_(\d+)_?(\d+)?\.png\)'`. This is strictly required to parse the diagram bounding-box coordinates from the AI's raw output.
