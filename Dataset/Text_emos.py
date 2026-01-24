import torch
import torch.nn.functional as F
import os
import sys
import json
import numpy as np
import cv2
from tqdm import tqdm
import gc
from PIL import Image

import transformers.image_utils
if not hasattr(transformers.image_utils, "VideoInput"):
    transformers.image_utils.VideoInput = transformers.image_utils.ImageInput

from transformers import AutoModelForCausalLM, AutoProcessor, RobertaTokenizer, RobertaForSequenceClassification

VLLAMA_PATH = "emos/VideoLLaMA3/VideoLLaMA3-7B"
RTER_PATH = "emos/emotion-english-roberta-large"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if os.path.exists(VLLAMA_PATH):
    sys.path.append(os.path.abspath(VLLAMA_PATH))

class UnifiedEmotionExtractor:
    def __init__(self):
        print(f"--- Initializing Unified Emotion Extractor (Device: {DEVICE}) ---")
        self.v_model = AutoModelForCausalLM.from_pretrained(
            VLLAMA_PATH,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        self.v_processor = AutoProcessor.from_pretrained(VLLAMA_PATH, trust_remote_code=True)
        
        if hasattr(self.v_processor, "image_processor"):
            self.v_processor.image_processor.max_dynamic_patch = 1

        self.r_tokenizer = RobertaTokenizer.from_pretrained(RTER_PATH)
        self.r_model = RobertaForSequenceClassification.from_pretrained(RTER_PATH).to(DEVICE)
        self.r_model.eval()

    def get_1024d_vector(self, text):
        if not text or "Error" in text:
            return np.zeros(1024)
        inputs = self.r_tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        with torch.no_grad():
            outputs = self.r_model(**inputs, output_hidden_states=True)
            return outputs.hidden_states[-1][:, 0, :].detach().cpu().numpy().squeeze()

    def extract_and_resize_frames(self, video_path, max_frames=6, target_height=448):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return None
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return None
        indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
        images = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret: break
            h, w = frame.shape[:2]
            new_w = int(w * (target_height / h))
            resized_frame = cv2.resize(frame, (new_w, target_height), interpolation=cv2.INTER_AREA)
            images.append(Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)))
        cap.release()
        return images if len(images) > 0 else None

    def generate_visual_description(self, video_path, task_type="scene", context_info=None):
        pil_images = self.extract_and_resize_frames(video_path, max_frames=6)
        if pil_images is None: return "Error: Video unreadable"
        
        char_id = context_info.get("char_id", "")
        scene_desc = context_info.get("scene_desc", "")
        
        if task_type == "scene":
            prompt = f"### [Scene Context]\n{scene_desc}\n\n### [Task]\nAnalyze the visual atmosphere and environmental setting of this scene."
        else:
            prompt = (f"### [Task]\n"
                    f"Focusing on the character {char_id}, describe their facial expressions and micro-expressions in detail.")

        content = [{"type": "image", "image": img} for img in pil_images]
        content.append({"type": "text", "text": prompt})

        try:
            torch.cuda.empty_cache()
            inputs = self.v_processor(conversation=[{"role": "user", "content": content}], add_generation_prompt=True, return_tensors="pt")
            inputs = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            if "pixel_values" in inputs: inputs["pixel_values"] = inputs["pixel_values"].to(self.v_model.dtype)
            
            with torch.no_grad():
                output_ids = self.v_model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    repetition_penalty=1.1,
                    use_cache=True
                )
            
            return self.v_processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        except Exception as e: return f"Model Error: {e}"
        finally:
            torch.cuda.empty_cache()
            gc.collect()

def batch_process(base_out="preprocessed_data/features"):
    extractor = UnifiedEmotionExtractor()
    with open("data/dataset.json", 'r', encoding='utf-8') as f:
        index_data = json.load(f)

    for sub in ["text", "scene", "face", "arc"]:
        os.makedirs(os.path.join(base_out, sub), exist_ok=True)

    for movie in index_data.get("movies", []):
        movie_path = movie.get("dataset_path")
        with open(os.path.join("data", movie_path), 'r', encoding='utf-8') as f:
            scene_data = json.load(f)

        for scene in tqdm(scene_data.get("scenes", []), desc=f"Processing {movie_path}"):
            bg_context, scene_id = scene.get("scene_description", ""), scene["scene_id"]
            
            character_map = {char["char_id"]: char.get("persona", "") for char in scene.get("characters", [])}

            for utt in scene.get("utterances", []):
                char_id, idx = utt["char_id"], f"{utt['utterance_index']:03d}"
                video_path = os.path.join("data", utt["video_path"])
                if_face = utt.get("if_face", True)
                persona = character_map.get(char_id, "")
                
                context_info = {
                    "scene_desc": bg_context,
                    "char_id": char_id,
                }
                
                prefix = f"{scene_id}@{char_id}"
                suffix = f"{prefix}_00_{scene_id}_{idx}.npy"
                
                out_paths = {
                    "text": os.path.join(base_out, "text", f"{prefix}-text-{suffix}"),
                    "scene": os.path.join(base_out, "scene", f"{prefix}-scene-{suffix}"),
                    "face": os.path.join(base_out, "face", f"{prefix}-face_desc-{suffix}"),
                    "arc": os.path.join(base_out, "arc", f"{prefix}-arc-{suffix}")
                }

                if all(os.path.exists(p) for p in out_paths.values()): continue

                text_input = f"Persona: {persona} | Context: {bg_context} | Script: {utt.get('text', '')}"
                np.save(out_paths["text"], extractor.get_1024d_vector(text_input))

                np.save(out_paths["arc"], extractor.get_1024d_vector(utt.get("director_arc", "")))

                si_desc = extractor.generate_visual_description(video_path, "scene", context_info=context_info)
                np.save(out_paths["scene"], extractor.get_1024d_vector(si_desc))

                if if_face:
                    fi_desc = extractor.generate_visual_description(video_path, "face", context_info=context_info)
                    face_vec = extractor.get_1024d_vector(fi_desc)
                else:
                    face_vec = np.zeros(1024)
                np.save(out_paths["face"], face_vec)

if __name__ == "__main__":
    batch_process()