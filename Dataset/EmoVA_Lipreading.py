import os
import sys
import cv2
import torch
import json
import numpy as np
import shutil
import warnings
from pathlib import Path
from tqdm import tqdm
from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from emonet.emonet.models import EmoNet  
import face_alignment

current_dir = os.path.dirname(os.path.abspath(__file__))
repo_path = os.path.join(current_dir, "Lipreading_using_Temporal_Convolutional_Networks")
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)
    
sam2_repo_path = os.path.join(current_dir, "sam2")
if sam2_repo_path not in sys.path:
    sys.path.append(sam2_repo_path)

from sam2.build_sam import build_sam2_video_predictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from lipreading.model import Lipreading 

warnings.filterwarnings("ignore", category=UserWarning, module="face_alignment")

class CombinedFeatureExtractor:
    def __init__(self, sam_checkpoint="sam2.1_hiera_large.pt", emonet_checkpoint='emonet/pretrained/emonet_8.pth'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.sam_checkpoint = os.path.abspath(os.path.join(self.script_dir, "sam2", sam_checkpoint))
        self.sam_config_dir = os.path.abspath(os.path.join(self.script_dir, "sam2", "sam2", "configs", "sam2.1"))
        self.sam_predictor = self._init_sam()
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device=self.device)
        self.lip_net = self._init_lip_net()
        self.emo_net = self._init_emonet(emonet_checkpoint)

    def _init_sam(self):
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=self.sam_config_dir, version_base=None):
            return build_sam2_video_predictor("sam2.1_hiera_l.yaml", ckpt_path=self.sam_checkpoint, device=self.device)

    def _init_lip_net(self):
        config_path = os.path.join(repo_path, 'configs/lrw_resnet18_mstcn.json')
        model_path = os.path.join(repo_path, 'models/lrw_resnet18_mstcn_video.pth')
        with open(config_path, 'r') as f:
            config = json.load(f)
        tcn_options = {
            'num_layers': config['tcn_num_layers'], 'kernel_size': config['tcn_kernel_size'],
            'dropout': config['tcn_dropout'], 'dwpw': config['tcn_dwpw'], 'width_mult': config['tcn_width_mult']
        }
        net = Lipreading(
            modality='video', hidden_dim=256, backbone_type=config['backbone_type'], 
            relu_type=config['relu_type'], width_mult=config['width_mult'], 
            extract_feats=True, tcn_options=tcn_options
        ).to(self.device)
        ckpt = torch.load(model_path, map_location='cpu')
        state_dict = ckpt.get('model_state_dict', ckpt)
        net.load_state_dict({k.replace('module.',''): v for k, v in state_dict.items()}, strict=False)
        return net.eval()

    def _init_emonet(self, checkpoint):
        net = EmoNet(n_expression=8).to(self.device)
        state_dict = torch.load(checkpoint, map_location='cpu')
        net.load_state_dict({k.replace('module.',''): v for k, v in state_dict.items()}, strict=False)
        return net.eval()

    def _process_frame_features(self, isolated_bgr):
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=False):
            rgb_frame = cv2.cvtColor(isolated_bgr, cv2.COLOR_BGR2RGB)
            preds = self.fa.get_landmarks(rgb_frame)
            if preds is None or len(preds) == 0:
                return None, None
            target_lm = preds[0]
            mouth_points = target_lm[48:68]
            mx_min, my_min = np.min(mouth_points, axis=0).astype(int)
            mx_max, my_max = np.max(mouth_points, axis=0).astype(int)
            m_x1, m_y1, m_x2, m_y2 = max(0, mx_min-15), max(0, my_min-15), mx_max+15, my_max+15
            mouth_roi = isolated_bgr[m_y1:m_y2, m_x1:m_x2]
            lip_feat = None
            if mouth_roi.size > 0:
                m_gray = cv2.resize(cv2.cvtColor(mouth_roi, cv2.COLOR_BGR2GRAY), (96, 96))
                m_tensor = torch.FloatTensor(m_gray).view(1, 1, 1, 96, 96).to(self.device).float() / 255.0
                lip_feat = self.lip_net(m_tensor, [1]).view(-1).cpu().numpy()
            x_min, y_min = np.min(target_lm, axis=0)
            x_max, y_max = np.max(target_lm, axis=0)
            side = max(int((x_max-x_min)*1.4), int((y_max-y_min)*1.4))
            cx, cy = int((x_min+x_max)/2), int((y_min+y_max)/2)
            e_x1, e_y1 = max(0, cx-side//2), max(0, cy-side//2)
            face_crop = isolated_bgr[e_y1:e_y1+side, e_x1:e_x1+side]
            va_feat = None
            if face_crop.size > 0:
                f_rgb = cv2.resize(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB), (256, 256))
                f_tensor = torch.Tensor(f_rgb).permute(2,0,1).unsqueeze(0).to(self.device).float() / 255.0
                emo_out = self.emo_net(f_tensor)
                va_feat = emo_out['va_256'].view(-1).cpu().numpy()
            return lip_feat, va_feat

    def process_video(self, video_path, lip_out_path, va_out_path):
        temp_dir = "temp_combined_frames"
        if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        mid_idx = total_frames // 2
        frame_count = 0
        mid_bgr = None
        while True:
            ret, frame = cap.read()
            if not ret: break
            cv2.imwrite(os.path.join(temp_dir, f"{frame_count:05d}.jpg"), frame)
            if frame_count == mid_idx: mid_bgr = frame.copy()
            frame_count += 1
        cap.release()
        mask_gen = SAM2AutomaticMaskGenerator(self.sam_predictor)
        with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            masks = mask_gen.generate(cv2.cvtColor(mid_bgr, cv2.COLOR_BGR2RGB))
        prop_dir = "combined_proposals"
        os.makedirs(prop_dir, exist_ok=True)
        for i, m in enumerate(masks):
            p = mid_bgr.copy()
            p[~m['segmentation']] = 0
            cv2.imwrite(os.path.join(prop_dir, f"id_{i}.jpg"), p)
        print(f"\n[Task: {os.path.basename(video_path)}]")
        selected_id = int(input(f"Select Target ID (0-{len(masks)-1}): "))
        shutil.rmtree(prop_dir)
        lip_list, va_list = [], []
        with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            state = self.sam_predictor.init_state(video_path=temp_dir)
            self.sam_predictor.add_new_mask(state, frame_idx=mid_idx, obj_id=1, mask=masks[selected_id]['segmentation'])
            for f_idx, _, mask_logits in tqdm(self.sam_predictor.propagate_in_video(state), total=frame_count, desc="Processing"):
                mask = (mask_logits[0] > 0.0).cpu().numpy().astype(np.uint8)
                img = cv2.imread(os.path.join(temp_dir, f"{f_idx:05d}.jpg"))
                isolated = cv2.bitwise_and(img, img, mask=mask if len(mask.shape)==2 else mask[0])
                l_f, v_f = self._process_frame_features(isolated)
                lip_list.append((f_idx, l_f))
                va_list.append((f_idx, v_f))
        def finalize(feat_list, dim):
            feat_list.sort(key=lambda x: x[0])
            res = []
            for _, val in feat_list:
                if val is not None: res.append(val)
                else: res.append(res[-1] if res else np.zeros(dim))
            return np.stack(res)
        np.save(lip_out_path, finalize(lip_list, 512))
        np.save(va_out_path, finalize(va_list, 256))
        shutil.rmtree(temp_dir)
        print(f"Completed: {os.path.basename(lip_out_path)}")

def batch_run(index_json="data/dataset.json"):
    extractor = CombinedFeatureExtractor()
    with open(index_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    lip_root = "preprocessed_data/features/extrated_embedding_gray"
    va_root = "preprocessed_data/features/VA_features"
    os.makedirs(lip_root, exist_ok=True)
    os.makedirs(va_root, exist_ok=True)
    for movie in data.get("movies", []):
        scene_json = os.path.join("data", movie.get("dataset_path"))
        if not os.path.exists(scene_json): continue
        with open(scene_json, 'r', encoding='utf-8') as f:
            scene_data = json.load(f)
        for scene in scene_data.get("scenes", []):
            s_id = scene["scene_id"]
            for utt in scene.get("utterances", []):
                if not utt.get("if_face", True):
                    print(f"Skipping {s_id} - Utt {utt['utterance_index']}: No face detected in meta.")
                    continue
                c_id = utt["char_id"]
                idx = f"{utt['utterance_index']:03d}"
                lip_name = f"{s_id}@{c_id}-face-{s_id}@{c_id}_00_{s_id}_{idx}.npy"
                va_name = f"{s_id}@{c_id}-feature-{s_id}@{c_id}_00_{s_id}_{idx}.npy"
                lip_p = os.path.join(lip_root, lip_name)
                va_p = os.path.join(va_root, va_name)
                if os.path.exists(lip_p) and os.path.exists(va_p): continue
                v_path = os.path.join("data", utt["video_path"])
                extractor.process_video(v_path, lip_p, va_p)

if __name__ == "__main__":
    batch_run()