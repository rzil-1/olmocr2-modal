import os
import sys
import json
import base64
import modal
import re
from io import BytesIO
from PIL import Image

# 1. Define the Modal Environment
image = modal.Image.debian_slim().pip_install(
    "torch", 
    "transformers", 
    "accelerate", 
    "qwen-vl-utils", 
    "pymupdf==1.25.3", 
    "pillow==11.1.0",
    "json_repair",
    "torchvision",
    "huggingface_hub",
    "hf_transfer"
)

app = modal.App("olmocr-math-extractor")

# Create a persistent volume to store the 14GB model
hf_cache_vol = modal.Volume.from_name("hf-model-cache", create_if_missing=True)

@app.cls(
    image=image, 
    gpu="A10G", 
    timeout=1200,
    volumes={"/root/.cache/huggingface": hf_cache_vol}
)
class OLMOCREngine:
    @modal.enter()
    def load_model(self):
        import os
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        
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
        import fitz
        import re
        import torch
        
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        total_pages = len(doc)
        
        results = {
            "raw_markdown": "",
            "equations": [],
            "diagrams": []
        }
        
        cropped_images_data = []
        found_diagrams_total = 0
        
        print("Extracting text and diagrams (Math Optimized OCR)...")
        pass1_prompt = (
            "Act as an academic paper corrector. Transcribe all text and mathematical expressions from this document exactly as written. "
            "Use Markdown for text structure, LaTeX for all formulas, and ensure the transcription follows the natural reading flow of the page. "
            "Do not omit any handwritten annotations or symbols. "
            "If there are any figures or charts, label them with the following markdown syntax: "
            "![Alt text describing the contents of the figure](page_x1_y1_x2_y2.png) "
            "CRITICAL: The coordinates (x1, y1, x2, y2) MUST be normalized integers between 0 and 1000. (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner. For example, a full page image is page_0_0_1000_1000.png."
        )
        
        full_markdown = ""
        
        for page_idx in range(total_pages):
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
                        {"type": "text", "text": pass1_prompt},
                    ],
                }
            ]

            text_input = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text_input], images=[main_image], padding=True, return_tensors="pt")
            inputs = {key: value.to(self.device) for (key, value) in inputs.items()}

            with torch.no_grad():
                output = self.model.generate(**inputs, temperature=0.1, max_new_tokens=3000, do_sample=True)

            prompt_length = inputs["input_ids"].shape[1]
            new_tokens = output[:, prompt_length:]
            text_output = self.processor.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]
            
            # Extract diagrams and crop
            pattern = r'!\[(.*?)\]\((page|\d+)_(\d+)_(\d+)_(\d+)_?(\d+)?\.png\)'
            matches = list(re.finditer(pattern, text_output))
            
            for match_obj in matches:
                original_markdown = match_obj.group(0)
                alt_text = match_obj.group(1)
                g = match_obj.groups()
                
                # In this schema, we keep the original markdown in raw_markdown
                # but we still want to crop
                
                if g[5] is not None:
                    ax1, ay1, ax2, ay2 = float(g[2]), float(g[3]), float(g[4]), float(g[5])
                else:
                    ax1, ay1, ax2, ay2 = float(g[2]), float(g[3]), float(g[4]), float(g[5])
                    
                # Robust min/max logic to handle flipped coordinates from AI
                x1 = max(0, min(1000, min(ax1, ax2)))
                y1 = max(0, min(1000, min(ay1, ay2)))
                x2 = max(0, min(1000, max(ax1, ax2)))
                y2 = max(0, min(1000, max(ay1, ay2)))
                
                if x2 - x1 < 5 or y2 - y1 < 5:
                    print(f"Skipping tiny/invalid crop: {x1}, {y1}, {x2}, {y2}")
                    results["diagrams"].append({
                        "alt_text": alt_text,
                        "image_path": None,
                        "page": page_idx + 1,
                        "coordinates": [int(x1), int(y1), int(x2), int(y2)]
                    })
                    continue
                
                try:
                    rect = page.rect
                    pw, ph = rect.width, rect.height
                    x0_pdf = (x1 / 1000.0) * pw
                    y0_pdf = (y1 / 1000.0) * ph
                    x1_pdf = (x2 / 1000.0) * pw
                    y1_pdf = (y2 / 1000.0) * ph
                    
                    crop_rect = fitz.Rect(x0_pdf, y0_pdf, x1_pdf, y1_pdf)
                    crop_pix = page.get_pixmap(matrix=fitz.Matrix(3, 3), clip=crop_rect)
                    crop_bytes = crop_pix.tobytes("png")
                    
                    safe_alt = re.sub(r'[^a-zA-Z0-9]', '_', alt_text)[:30]
                    base_name = os.path.splitext(filename)[0]
                    img_filename = f"{base_name}_pg{page_idx+1}_{found_diagrams_total}_{safe_alt}.png"
                    
                    cropped_images_data.append({
                        "filename": img_filename,
                        "data": crop_bytes
                    })
                    
                    results["diagrams"].append({
                        "alt_text": alt_text,
                        "image_path": img_filename,
                        "page": page_idx + 1,
                        "coordinates": [int(x1), int(y1), int(x2), int(y2)]
                    })
                    found_diagrams_total += 1
                except Exception as e:
                    print(f"Failed to crop diagram on page {page_idx+1}: {e}")
                    results["diagrams"].append({
                        "alt_text": alt_text,
                        "image_path": None,
                        "page": page_idx + 1,
                        "coordinates": [int(x1), int(y1), int(x2), int(y2)]
                    })
                    found_diagrams_total += 1
            
            full_markdown += f"\n\n--- PAGE {page_idx+1} ---\n\n" + text_output

        results["raw_markdown"] = full_markdown
        
        # Extract equations from the full markdown
        inline_math = re.findall(r'\\\((.*?)\\\)', full_markdown, re.DOTALL)
        block_math = re.findall(r'\\\[(.*?)\\\]', full_markdown, re.DOTALL)
        results["equations"] = [eq.strip() for eq in inline_math + block_math]
        
        doc.close()
        return results, cropped_images_data

@app.local_entrypoint()
def main(pdf_path: str):
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        return
        
    filename = os.path.basename(pdf_path)
    print(f"Reading local file: {filename}")
    
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
        
    print(f"Sending {filename} up to the Modal A10G Cloud GPU for Math-Optimized processing...")
    
    engine = OLMOCREngine()
    results, cropped_images = engine.process_document.remote(pdf_bytes, filename)
    
    # Save JSON Results
    json_path = "modal_math_results.json"
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
        
    print(f"Successfully saved math-optimized results to {json_path}")
    
    # Save Diagram Images
    if cropped_images:
        diagram_folder = "math_detected_diagrams"
        os.makedirs(diagram_folder, exist_ok=True)
        
        for img in cropped_images:
            img_path = os.path.join(diagram_folder, img["filename"])
            with open(img_path, "wb") as f:
                f.write(img["data"])
        print(f"Saved {len(cropped_images)} diagram images to '{diagram_folder}/'")
    else:
        print("No diagrams detected.")
