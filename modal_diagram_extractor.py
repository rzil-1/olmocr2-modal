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
    "pymupdf==1.25.3", 
    "pillow==11.1.0",
    "json_repair",
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
            "filename": filename,
            "answers": {},
            "equations": [],
            "diagrams": [],
            "metadata": {
                "student_id": "UNKNOWN",
                "subject": "UNKNOWN",
                "total_pages": total_pages
            }
        }
        
        # We will hold cropped images in memory and send them back to the user's local machine
        cropped_images_data = []
        found_diagrams_total = 0
        
        # --- PASS 1: Spatial OCR (Page by Page) ---
        print("Pass 1: Extracting text and diagrams (Spatial OCR)...")
        pass1_prompt = (
            "Act as an academic paper corrector. Transcribe all text and mathematical expressions from this document exactly as written. "
            "Use Markdown for text structure, LaTeX for all formulas, and ensure the transcription follows the natural reading flow of the page. "
            "Do not omit any handwritten annotations or symbols. "
            "If there are any figures or charts, label them with the following markdown syntax: "
            "![Alt text describing the contents of the figure](page_startx_starty_endx_endy.png) "
            "CRITICAL: The coordinates (startx, starty, endx, endy) MUST be normalized integers between 0 and 1000. For example, a full page image is page_0_0_1000_1000.png."
        )
        
        full_markdown = ""
        diagram_mapping = {} # Maps original markdown string to filename
        
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
            
            # Crop diagrams immediately while we are on the page
            pattern = r'!\[(.*?)\]\((page|\d+)_(\d+)_(\d+)_(\d+)_?(\d+)?\.png\)'
            matches = list(re.finditer(pattern, text_output))
            
            for match_obj in matches:
                original_markdown = match_obj.group(0)
                alt_text = match_obj.group(1)
                g = match_obj.groups()
                
                marker_id = f"[STUDENT_DRAWING_DETECTED_{found_diagrams_total}]"
                text_output = text_output.replace(original_markdown, marker_id)
                
                if g[5] is not None:
                    x1, y1, x2, y2 = float(g[2]), float(g[3]), float(g[4]), float(g[5])
                else:
                    x1, y1, x2, y2 = float(g[2]), float(g[3]), float(g[4]), float(g[5])
                    
                # Clamp coordinates between 0 and 1000 to prevent PyMuPDF crashes
                x1 = max(0, min(1000, x1))
                y1 = max(0, min(1000, y1))
                x2 = max(0, min(1000, x2))
                y2 = max(0, min(1000, y2))
                
                if x2 <= x1 or y2 <= y1:
                    print(f"Invalid normalized coordinates hallucinated: {x1}, {y1}, {x2}, {y2}. Skipping crop.")
                    diagram_mapping[marker_id] = {
                        "alt_text": alt_text,
                        "image_path": None,
                        "coordinates": [int(x1), int(y1), int(x2), int(y2)]
                    }
                    found_diagrams_total += 1
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
                    
                    diagram_mapping[marker_id] = {
                        "alt_text": alt_text,
                        "image_path": img_filename,
                        "coordinates": [int(x1), int(y1), int(x2), int(y2)]
                    }
                    found_diagrams_total += 1
                except Exception as e:
                    print(f"Failed to crop diagram on page {page_idx+1}: {e}")
                    diagram_mapping[marker_id] = {
                        "alt_text": alt_text,
                        "image_path": None,
                        "coordinates": [int(x1), int(y1), int(x2), int(y2)]
                    }
                    found_diagrams_total += 1
                    
            full_markdown += f"\n\n--- PAGE {page_idx+1} ---\n\n" + text_output

        # --- PASS 2: Semantic JSON Routing (Text Only) ---
        print("Pass 2: Restructuring transcript into Schema (Semantic Routing)...")
        pass2_prompt = (
            "You are an AI tasked with restructuring an exam transcript. "
            "Below is the raw markdown transcript of a student's exam. "
            "Extract all the text and logically group it by the question number it answers. "
            "Only output the student's answer text for each question. Do not include the question text itself. "
            "Output your response strictly as a valid JSON object where the keys are the question numbers formatted as 'q1', 'q2', etc. "
            "and the values are the full extracted answer text. "
            "CRITICAL INSTRUCTIONS:\n"
            "1. If you see a diagram marker like [STUDENT_DRAWING_DETECTED_0], you MUST include it verbatim inside the JSON answer string where it belongs. Do not summarize or delete it.\n"
            "2. You MUST perfectly escape all backslashes in LaTeX formulas (e.g., use \\\\( and \\\\frac instead of \\( and \\frac) to ensure the output is valid JSON.\n"
            "If text does not belong to a specific question, put it under the key 'unassigned'. "
            "Do not include any extra markdown outside of the JSON object.\n\n"
            f"RAW TRANSCRIPT:\n{full_markdown}"
        )
        
        messages2 = [
            {"role": "user", "content": [{"type": "text", "text": pass2_prompt}]}
        ]
        
        text_input2 = self.processor.apply_chat_template(messages2, tokenize=False, add_generation_prompt=True)
        inputs2 = self.processor(text=[text_input2], padding=True, return_tensors="pt")
        inputs2 = {key: value.to(self.device) for (key, value) in inputs2.items()}

        with torch.no_grad():
            output2 = self.model.generate(**inputs2, temperature=0.1, max_new_tokens=4000, do_sample=True)

        prompt_length2 = inputs2["input_ids"].shape[1]
        new_tokens2 = output2[:, prompt_length2:]
        text_output2 = self.processor.tokenizer.batch_decode(new_tokens2, skip_special_tokens=True)[0]
        
        # Parse Pass 2 JSON and format to the Dev Team's final schema
        import json_repair
        json_match = re.search(r'\{.*\}', text_output2, re.DOTALL)
        if json_match:
            try:
                parsed_answers = json_repair.loads(json_match.group(0))
                if isinstance(parsed_answers, dict):
                    for k, val_str in parsed_answers.items():
                        k_lower = str(k).lower().strip()
                        val_str = str(val_str).strip()
                        
                        # Find any diagrams embedded in this answer
                        diagram_match = re.search(r'\[STUDENT_DRAWING_DETECTED_\d+\]', val_str)
                        if diagram_match:
                            original_marker = diagram_match.group(0)
                            diag_meta = diagram_mapping.get(original_marker)
                            
                            if diag_meta:
                                # New nested Dev Team schema for diagrams
                                results["answers"][k_lower] = {
                                    "type": "diagram",
                                    "image_data": f"detected_diagrams/{diag_meta['image_path']}" if diag_meta['image_path'] else "crop_failed.png",
                                    "student_transcription": diag_meta['alt_text'],
                                    "text_answer": re.sub(r'\[STUDENT_DRAWING_DETECTED_\d+\]', '', val_str).strip(),
                                    "metadata": {
                                        "is_handwritten": True,
                                        "labels_found": len(diag_meta['alt_text'].split(','))
                                    }
                                }
                                results["diagrams"].append(diag_meta)
                            else:
                                results["answers"][k_lower] = val_str
                        else:
                            results["answers"][k_lower] = val_str
                            
                        # Equation extraction
                        inline_math = re.findall(r'\\\((.*?)\\\)', val_str, re.DOTALL)
                        block_math = re.findall(r'\\\[(.*?)\\\]', val_str, re.DOTALL)
                        results["equations"].extend([eq.strip() for eq in inline_math])
                        results["equations"].extend([eq.strip() for eq in block_math])
                        
            except Exception as e:
                print(f"Warning: Failed to parse Final JSON: {e}")
                
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
