import os
import sys
import json
import base64
import modal

# 1. Define the Modal Environment
# This tells Modal exactly what libraries the remote GPU container needs.
image = modal.Image.debian_slim().pip_install(
    "torch", 
    "transformers", 
    "accelerate", 
    "qwen-vl-utils", 
    "pymupdf", 
    "pillow",
    "torchvision",
    "huggingface_hub",
    "hf_transfer"
)

app = modal.App("olmocr-diagram-extractor")

# Create a persistent volume to store the 14GB model so it never downloads twice
hf_cache_vol = modal.Volume.from_name("hf-model-cache", create_if_missing=True)

# 2. Define the Remote GPU Class
@app.cls(
    image=image, 
    gpu="A10G", 
    timeout=1200,
    volumes={"/root/.cache/huggingface": hf_cache_vol}
)
class OLMOCREngine:
    @modal.enter()
    def load_model(self):
        """This runs once when the container boots up to load the 14GB model into VRAM."""
        import os
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        
        # Enable insanely fast Rust-based downloads to bypass any hanging issues
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        
        print("Downloading & Loading Model into A10G GPU VRAM...")
        self.MODEL_NAME = "allenai/olmOCR-2-7B-1025"
        self.device = torch.device("cuda")
        
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.MODEL_NAME, 
            torch_dtype=torch.float16, 
            device_map=self.device
        ).eval()
        self.processor = AutoProcessor.from_pretrained(self.MODEL_NAME)
        print("Model loaded successfully!")

    @modal.method()
    def process_document(self, pdf_bytes: bytes, filename: str):
        """This runs for every PDF passed to the engine. It returns the JSON and raw image bytes."""
        import fitz
        import re
        from io import BytesIO
        from PIL import Image
        import torch

        # Read the PDF directly from the byte stream in memory
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        total_pages = len(doc)
        
        results = {
            "raw_markdown": "",
            "equations": [],
            "diagrams": []
        }
        
        # We will hold cropped images in memory and send them back to the user's local machine
        cropped_images_data = []
        found_diagrams_total = 0

        # The exact same prompt ensuring OCR logic remains identical
        prompt = (
            "Act as an academic paper corrector. Transcribe all text and mathematical expressions from this document exactly as written. "
            "Use Markdown for text structure, LaTeX for all formulas, and ensure the transcription follows the natural reading flow of the page. "
            "Do not omit any handwritten annotations or symbols. "
            "If there are any figures or charts, label them with the following markdown syntax: "
            "![Alt text describing the contents of the figure](page_startx_starty_width_height.png)"
        )

        for page_idx in range(total_pages):
            print(f"Processing Page {page_idx+1}/{total_pages} on Cloud GPU...")
            page = doc[page_idx]
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            image_bytes_png = pix.tobytes("png")
            image_base64 = base64.b64encode(image_bytes_png).decode("utf-8")
            main_image = Image.open(BytesIO(image_bytes_png))

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            text_input = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            inputs = self.processor(
                text=[text_input],
                images=[main_image],
                padding=True,
                return_tensors="pt",
            )
            inputs = {key: value.to(self.device) for (key, value) in inputs.items()}

            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    temperature=0.1,
                    max_new_tokens=3000,
                    do_sample=True,
                )

            prompt_length = inputs["input_ids"].shape[1]
            new_tokens = output[:, prompt_length:]
            text_output = self.processor.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]
            
            # Text accumulation
            results["raw_markdown"] += text_output + "\n\n"
            
            # Equation extraction
            inline_math = re.findall(r'\\\((.*?)\\\)', text_output, re.DOTALL)
            block_math = re.findall(r'\\\[(.*?)\\\]', text_output, re.DOTALL)
            results["equations"].extend([eq.strip() for eq in inline_math])
            results["equations"].extend([eq.strip() for eq in block_math])
            
            # Diagram extraction and in-memory cropping
            pattern = r'!\[(.*?)\]\((page|\d+)_(\d+)_(\d+)_(\d+)_?(\d+)?\.png\)'
            for match_obj in re.finditer(pattern, text_output):
                alt_text = match_obj.group(1)
                g = match_obj.groups()
                
                if g[5] is not None:
                    x, y, w, h = g[2], g[3], g[4], g[5]
                else:
                    x, y, w, h = g[2], g[3], g[4], g[5]
                    
                if h is None: continue
                
                try:
                    rect = page.rect
                    pw, ph = rect.width, rect.height
                    x0 = (float(x) / 1000.0) * pw
                    y0 = (float(y) / 1000.0) * ph
                    w0 = (float(w) / 1000.0) * pw
                    h0 = (float(h) / 1000.0) * ph
                    
                    crop_rect = fitz.Rect(x0, y0, x0 + w0, y0 + h0)
                    crop_pix = page.get_pixmap(matrix=fitz.Matrix(3, 3), clip=crop_rect)
                    crop_bytes = crop_pix.tobytes("png")
                    
                    safe_alt = re.sub(r'[^a-zA-Z0-9]', '_', alt_text)[:30]
                    base_name = os.path.splitext(filename)[0]
                    img_filename = f"{base_name}_pg{page_idx}_{found_diagrams_total}_{safe_alt}.png"
                    
                    cropped_images_data.append({
                        "filename": img_filename,
                        "data": crop_bytes
                    })
                    
                    results["diagrams"].append({
                        "alt_text": alt_text,
                        "page": page_idx,
                        "coordinates": [int(x), int(y), int(w), int(h)],
                        "image_path": f"detected_diagrams/{img_filename}"
                    })
                    found_diagrams_total += 1
                except Exception as e:
                    print(f"Failed to crop diagram on page {page_idx}: {e}")
                    
        doc.close()
        return results, cropped_images_data

# 3. Local Execution Point (This runs on the Developer's Windows Laptop)
@app.local_entrypoint()
def main(pdf_path: str):
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        return
        
    filename = os.path.basename(pdf_path)
    print(f"Reading local file: {filename}")
    
    # Read PDF as bytes locally
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
        
    print(f"Sending {filename} up to the Modal A10G Cloud GPU for processing...")
    
    # Initialize the remote class and trigger the method
    engine = OLMOCREngine()
    results, cropped_images = engine.process_document.remote(pdf_bytes, filename)
    
    # --- Everything below this line happens back on the local machine ---
    
    # 1. Save JSON Results
    json_path = "modal_extracted_results.json"
    all_results = {}
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            try:
                all_results = json.load(f)
            except:
                pass
                
    all_results[filename] = results
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)
        
    print(f"Successfully saved structured text to {json_path}")
    
    # 2. Save Diagram Images
    if cropped_images:
        diagram_folder = "detected_diagrams"
        os.makedirs(diagram_folder, exist_ok=True)
        
        for img in cropped_images:
            img_path = os.path.join(diagram_folder, img["filename"])
            with open(img_path, "wb") as f:
                f.write(img["data"])  # Write the bytes sent back from the cloud
        print(f"Saved {len(cropped_images)} diagram images to '{diagram_folder}/'")
    else:
        print("No diagrams detected.")
