import os
import sys
import cv2
import torch
import numpy as np
import shutil
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra

current_dir = os.path.dirname(os.path.abspath(__file__))
sam2_repo_path = os.path.join(current_dir, "sam2")
if sam2_repo_path not in sys.path:
    sys.path.append(sam2_repo_path)

from sam2.build_sam import build_sam2_video_predictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

class SAM2VideoIsolator:
    def __init__(self, checkpoint_name="sam2.1_hiera_large.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.checkpoint = os.path.abspath(os.path.join(self.script_dir, "sam2", checkpoint_name))
        self.config_dir = os.path.abspath(os.path.join(self.script_dir, "sam2", "sam2", "configs", "sam2.1"))
        self.config_name = "sam2.1_hiera_l.yaml"
        self.predictor = self._load_model()

    def _load_model(self):
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        
        print(f"Loading from config directory: {self.config_dir}")
        with initialize_config_dir(config_dir=self.config_dir, version_base=None):
            predictor = build_sam2_video_predictor(
                self.config_name, 
                ckpt_path=self.checkpoint, 
                device=self.device
            )
        print(f"SAM 2.1 model loaded successfully.")
        return predictor

    def get_proposals(self, video_path, proposal_dir="character_proposals"):
        if os.path.exists(proposal_dir):
            shutil.rmtree(proposal_dir)
        os.makedirs(proposal_dir)

        mask_generator = SAM2AutomaticMaskGenerator(self.predictor)
        
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError(f"Unable to read video file: {video_path}")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        print("Generating character proposal previews...")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            masks = mask_generator.generate(frame_rgb)
        
        for i, mask_data in enumerate(masks):
            m = mask_data['segmentation']
            crop = frame.copy()
            crop[~m] = 0 
            x, y, w, h = [int(v) for v in mask_data['bbox']]
            cv2.imwrite(os.path.join(proposal_dir, f"ID_{i}.jpg"), crop[y:y+h, x:x+w])
        
        print(f"Found {len(masks)} objects, please check the {proposal_dir} folder.")
        return masks

    def select_id_interactively(self, proposals):
        print("\n" + "="*30)
        while True:
            val = input("👉 Please check the previews and enter the target character ID (or 'q' to quit): ")
            if val.lower() == 'q': return None
            try:
                selected_id = int(val)
                if 0 <= selected_id < len(proposals):
                    return selected_id
                print(f"Invalid ID, please enter a number between 0 and {len(proposals)-1}.")
            except ValueError:
                print("Please enter a valid numerical ID.")

    def isolate_character(self, video_path, output_path, selected_id, masks_data):
        temp_dir = "temp_frames_proc"
        if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            cv2.imwrite(os.path.join(temp_dir, f"{idx:05d}.jpg"), frame)
            idx += 1
        cap.release()

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = self.predictor.init_state(video_path=temp_dir)
            selected_mask = masks_data[selected_id]['segmentation'].astype(np.float32)
            self.predictor.add_new_mask(inference_state=state, frame_idx=0, obj_id=1, mask=selected_mask)

            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            print(f"Executing full video pixel-level isolation, Character ID: {selected_id}...")
            for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(state):
                mask = (out_mask_logits[0] > 0.0).cpu().numpy().astype(np.uint8)
                frame = cv2.imread(os.path.join(temp_dir, f"{out_frame_idx:05d}.jpg"))
                if len(mask.shape) == 3: mask = mask[0]
                
                black_bg = cv2.bitwise_and(frame, frame, mask=mask)
                out_video.write(black_bg)

            out_video.release()
        
        shutil.rmtree(temp_dir)
        print(f"Isolated video exported successfully: {output_path}")

if __name__ == "__main__":
    isolator = SAM2VideoIsolator()
    video_file = "sam2/input_character_video.mp4" 
    
    all_proposals = isolator.get_proposals(video_file)
    
    if all_proposals:
        selected_id = isolator.select_id_interactively(all_proposals)
        
        if selected_id is not None:
            output_name = f"isolated_character_ID_{selected_id}.avi"
            isolator.isolate_character(
                video_path=video_file,
                output_path=output_name,
                selected_id=selected_id,
                masks_data=all_proposals
            )
            print(f"Task completed! Result saved to: {output_name}")