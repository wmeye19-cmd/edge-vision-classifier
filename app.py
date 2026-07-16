import os
import time
import cv2
import gradio as gr
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

# =====================================================================
# 1. LOAD MODEL (LOCKED TO SINGLE CORE FOR CLOUD STABILITY)
# =====================================================================
MODEL_PATH = "optimized_production_model.tflite"

interpreter = tflite.Interpreter(model_path=MODEL_PATH, num_threads=1)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

CLASS_NAMES = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']

# =====================================================================
# HELPER FUNCTION
# =====================================================================
def run_tflite_inference(pil_crop):
    """Resizes any PIL crop to 96x96 and returns probabilities."""
    resized = pil_crop.resize((96, 96)).convert("RGB")
    img_array = np.array(resized, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0]

# =====================================================================
# 2. OPENCV TRADITIONAL EDGE CROPPER (Bulletproofed)
# =====================================================================
def traditional_smart_crop(pil_image):
    img_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    pts = np.argwhere(edges > 0)
    if len(pts) == 0:
        return pil_image
        
    y_min, x_min = pts.min(axis=0)
    y_max, x_max = pts.max(axis=0)
    
    h, w = gray.shape
    pad_x, pad_y = int((x_max - x_min) * 0.15), int((y_max - y_min) * 0.15)
    
    left, top = max(0, x_min - pad_x), max(0, y_min - pad_y)
    right, bottom = min(w, x_max + pad_x), min(h, y_max + pad_y)
    
    side = max(right - left, bottom - top)
    center_x, center_y = left + ((right - left) / 2), top + ((bottom - top) / 2)
    
    left, top = max(0, int(center_x - side / 2)), max(0, int(center_y - side / 2))
    right, bottom = min(w, int(center_x + side / 2)), min(h, int(center_y + side / 2))
    
    # SAFETY GUARDRAIL: If math creates a weird/invisible crop, abort and return full image
    if right - left < 10 or bottom - top < 10:
        return pil_image
    
    return pil_image.crop((left, top, right, bottom))

# =====================================================================
# 3. ADVANCED HYBRID PIPELINE
# =====================================================================
def classify_image(image):
    start_total = time.perf_counter()
    w, h = image.size
    
    # --- STAGE 0: THE FAST TRACK (Full Image) ---
    start_fast = time.perf_counter()
    fast_probs = run_tflite_inference(image)
    fast_time_ms = (time.perf_counter() - start_fast) * 1000
    
    max_confidence = float(np.max(fast_probs))
    if max_confidence >= 0.99:
        total_time_ms = (time.perf_counter() - start_total) * 1000
        return {CLASS_NAMES[i]: float(fast_probs[i]) for i in range(len(CLASS_NAMES))}, f"""
        ### ⚡ Performance Diagnostics (Non-Quantized)
        * **First Pass Inference:** `{fast_time_ms:.1f} ms` (Confidence: `{max_confidence*100:.1f}%`)
        * **OpenCV Edge Crop:** `Bypassed 🚀`
        * **5-Zone Radar Scan:** `Bypassed 🚀`
        * **Total Server Latency:** **`{total_time_ms:.1f} ms`**
        """
        
    # --- STAGE 1: LIGHTWEIGHT MATHEMATICAL CROP ---
    start_crop = time.perf_counter()
    cropped_img = traditional_smart_crop(image)
    crop_probs = run_tflite_inference(cropped_img)
    crop_time_ms = (time.perf_counter() - start_crop) * 1000
    
    max_crop_confidence = float(np.max(crop_probs))
    if max_crop_confidence >= 0.90:
        total_time_ms = (time.perf_counter() - start_total) * 1000
        return {CLASS_NAMES[i]: float(crop_probs[i]) for i in range(len(CLASS_NAMES))}, f"""
        ### ⚡ Performance Diagnostics (Non-Quantized)
        * **First Pass Inference:** `{fast_time_ms:.1f} ms` (Confidence: `{max_confidence*100:.1f}%` ⚠️)
        * **OpenCV Edge Crop:** `{crop_time_ms:.1f} ms` (Confidence: `{max_crop_confidence*100:.1f}%` ✅)
        * **5-Zone Radar Scan:** `Bypassed 🚀`
        * **Total Server Latency:** **`{total_time_ms:.1f} ms`**
        """

    # --- STAGE 2: THE QUADRANT RADAR SCAN (Thread-Safe & Sequential) ---
    start_radar = time.perf_counter()
    
    quadrants = {
        "Top-Left":     image.crop((0, 0, int(w * 0.6), int(h * 0.6))),
        "Top-Right":    image.crop((int(w * 0.4), 0, w, int(h * 0.6))),
        "Bottom-Left":  image.crop((0, int(h * 0.4), int(w * 0.6), h)),
        "Bottom-Right": image.crop((int(w * 0.4), int(h * 0.4), w, h)),
        "Center-Zone":  image.crop((int(w * 0.2), int(h * 0.2), int(w * 0.8), int(h * 0.8)))
    }
    
    best_overall_confidence = -1.0
    best_scoresheet = None
    winning_zone = ""
    
    # Run in a lightning-fast sequential loop to prevent TFLite crashing
    for zone_name, zone_crop in quadrants.items():
        zone_probs = run_tflite_inference(zone_crop)
        zone_max = float(np.max(zone_probs))
        
        if zone_max > best_overall_confidence:
            best_overall_confidence = zone_max
            best_scoresheet = zone_probs
            winning_zone = zone_name
            
    radar_time_ms = (time.perf_counter() - start_radar) * 1000
    total_time_ms = (time.perf_counter() - start_total) * 1000
    
    final_confidences = {CLASS_NAMES[i]: float(best_scoresheet[i]) for i in range(len(CLASS_NAMES))}
    
    return final_confidences, f"""
    ### ⚡ Performance Diagnostics (Non-Quantized)
    * **First Pass Inference:** `{fast_time_ms:.1f} ms` (Confidence: `{max_confidence*100:.1f}%` ⚠️)
    * **OpenCV Edge Crop:** `{crop_time_ms:.1f} ms` (Confidence: `{max_crop_confidence*100:.1f}%` ⚠️)
    * **5-Zone Radar Scan:** `{radar_time_ms:.1f} ms` 🔍 (Winner: `{winning_zone}` at `{best_overall_confidence*100:.1f}%`)
    * **Total Server Latency:** **`{total_time_ms:.1f} ms`**
    """

# =====================================================================
# 4. LAUNCH INTERFACE
# =====================================================================
# 💡 Paste your actual profile and project URLs inside these strings:
GHOST_BLOG_URL = "https://www.williamameyer.com/edge-ai-farm-security-part-1/"
GITHUB_REPO_URL = "https://github.com/wmeye19-cmd/edge-vision-classifier/"
LINKEDIN_PROFILE_URL = "https://www.linkedin.com/in/willameyer/"

portfolio_footer = f"""
---
### 🌐 Project Ecosystem & Portals
* 📝 **System Architecture Deep-Dive:** [Read the 3-Part Breakdown on My Blog]({GHOST_BLOG_URL})
* 💻 **Model Optimization & Pipelines:** [Explore the Codebase on GitHub]({GITHUB_REPO_URL})
* 💼 **Professional Network:** [Connect with Me on LinkedIn]({LINKEDIN_PROFILE_URL})

_Developed as an edge-native computer vision prototype demonstrating high-efficiency local inference capabilities._
"""

interface = gr.Interface(
    fn=classify_image,                       
    inputs=gr.Image(type="pil"),             
    outputs=[gr.Label(num_top_classes=3, label="Predictions"), gr.Markdown()],     
    title="🚜 Localized Edge AI Vision Prototype",
    description="""
    This prototype classifies 10 environmental categories: airplanes, birds, cars, cats, deer, dogs, horses, monkeys, ships, and trucks. 
    
    ⚡ **3-Tier Escalation Architecture:** Full Frame Pass ➡️ OpenCV Smart Edge Crop ➡️ 5-Zone Quadrant Radar Sweep.
    """,
    article=portfolio_footer
)

interface.launch(ssr_mode=False)