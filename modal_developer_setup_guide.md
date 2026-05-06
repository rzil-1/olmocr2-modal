# Serverless OCR & Diagram Extractor: Developer Setup Guide

This guide is for the integration team. It walks through setting up, authenticating, and running the **Modal-powered Serverless OCR Engine** (`modal_diagram_extractor.py`). 

By using Modal, you do **not** need a local GPU, nor do you need to install heavy machine learning libraries (like PyTorch or CUDA) on your local Windows/Mac machine. All heavy lifting is securely offloaded to a cloud NVIDIA A10G GPU.

---

## Step 1: Install the Modal Client
While the script uses complex ML libraries (`transformers`, `torch`, etc.), those are installed automatically *in the cloud container*. 

To trigger the cloud execution from your laptop, you only need to install the Modal Python client.

Open your terminal or command prompt and run:
```bash
pip install -r modal_requirements.txt
```

---

## Step 2: Create a Free Modal Account
To rent the cloud GPU for the 30 seconds it takes to process a PDF, you need a Modal account. Modal provides $30 in free credits per month (enough to process ~10,000 PDF pages).

1. Go to [modal.com](https://modal.com/) in your browser.
2. Click **Sign Up** (you can use your GitHub or Google account).

---

## Step 3: Authenticate Your Terminal
Now you need to link your local terminal to your new Modal account so it has permission to spin up the cloud GPU.

In your terminal, run:
```bash
modal token new
```
*This will open a browser window. Click "Accept" or "Authorize". Your terminal is now authenticated!*

---

## Step 4: Run the Engine
Ensure you have the `modal_diagram_extractor.py` script and a test PDF (e.g., `test.pdf`) in the same folder. 

Run the script by typing:
```bash
modal run modal_diagram_extractor.py --pdf-path "test.pdf"
```

### What happens when you run this?
1. **The Image Build (First Run Only):** The very first time you run this, Modal will read the script, provision a Linux container, install PyTorch/CUDA, and mount a persistent cloud drive. This takes about ~1 minute.
2. **The Model Download (First Run Only):** The GPU boots up and downloads the 14GB `Qwen2.5-VL` weights from HuggingFace to your persistent cloud drive. This takes ~60-90 seconds. 
3. **Processing:** The PDF is read locally, beamed to the GPU as bytes, processed completely in memory, and the results are beamed back down to your laptop.

*(Note: On all future runs, Steps 1 and 2 are skipped. The script will execute in ~30 seconds).*

---

## Step 5: Where to Find the Output
When the terminal says `App completed`, look in your local folder. The cloud GPU has beamed the results directly back to your hard drive:

1. **`modal_extracted_results.json`:** 
   This contains the highly structured OCR data. It is separated into:
   - `raw_markdown`: The continuous text of the document.
   - `equations`: A clean list of all LaTeX formulas detected (ready to be fed into your Evaluation Engine).
   - `diagrams`: Metadata linking descriptions to coordinates and local image paths.

2. **`detected_diagrams/` (Folder):** 
   Contains perfectly cropped `.png` files of any charts, diagrams, or figures found in the PDF.

---

## Integration Architecture Notes (For Backend Devs)
When integrating this into the final MVP backend (e.g., FastAPI, Express, Node):

1. **Do not use `modal run` in production:** Instead, deploy the Modal app using `modal deploy modal_diagram_extractor.py`. This keeps the cloud function permanently available via API.
2. **Triggering it programmatically:** Once deployed, your backend server can trigger the OCR simply by importing it:
   ```python
   import modal
   f = modal.Function.lookup("olmocr-diagram-extractor", "OLMOCREngine.process_document")
   
   # Read uploaded PDF
   with open("user_upload.pdf", "rb") as pdf:
       pdf_bytes = pdf.read()
       
   # Call the remote GPU function
   results, cropped_images = f.remote(pdf_bytes, "user_upload.pdf")
   ```
3. **No Heavy Dependencies:** Your production backend server does *not* need to be a $500/mo GPU instance. Your backend can be a cheap $5/mo CPU server because the OCR runs entirely on Modal's serverless infrastructure.
