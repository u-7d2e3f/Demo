import torch
import json
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class DubberSceneDataset(Dataset):
    def __init__(self, root_json_path, split="train", 
                 feature_root="Dataset/preprocessed_data/features",
                 top_k=3):
        self.feature_root = feature_root
        self.split = split
        self.top_k = top_k
        self.rag_index_root = os.path.join(feature_root, "rag_indices")
        
        if not os.path.exists(root_json_path):
            raise FileNotFoundError(f"Missing master config: {root_json_path}")
            
        with open(root_json_path, 'r') as f:
            self.master_cfg = json.load(f)
            
        self.all_scenes = []
        for movie in self.master_cfg['movies']:
            target_scene_ids = movie.get(self.split, [])
            if not target_scene_ids: continue

            movie_json = os.path.join(os.path.dirname(root_json_path), movie['dataset_path'])
            if not os.path.exists(movie_json): continue
                
            with open(movie_json, 'r') as f:
                movie_data = json.load(f)
                for scene in movie_data['scenes']:
                    if scene['scene_id'] in target_scene_ids:
                        self.all_scenes.append(scene)

    def _get_basename(self, scene_id, char_id, u_idx):
        return f"{scene_id}@{char_id}_00_{scene_id}_{u_idx:03d}"

    def _load_feat(self, path, dim_expected):
        if not os.path.exists(path):
            return torch.randn(dim_expected)
        return torch.from_numpy(np.load(path)).float().squeeze()

    def _load_rag_refs(self, basename):
        json_path = os.path.join(self.rag_index_root, f"{basename}-arc_top_100.json")
        
        if not os.path.exists(json_path):
            return torch.zeros(self.top_k, 1280)
            
        with open(json_path, 'r') as f:
            indices = json.load(f)
        
        ref_vectors = []
        for i in range(min(self.top_k, len(indices))):
            ref_file = indices[i]['file_name']
            ref_path = os.path.join(self.feature_root, "emotion", ref_file)
            ref_vectors.append(self._load_feat(ref_path, 1280))
            
        while len(ref_vectors) < self.top_k:
            ref_vectors.append(torch.zeros(1280))
            
        return torch.stack(ref_vectors)

    def __len__(self):
        return len(self.all_scenes)

    def __getitem__(self, idx):
        scene = self.all_scenes[idx]
        scene_id, utterances = scene['scene_id'], scene['utterances']
        
        char_timbre_map = {
            char['char_id']: self._load_feat(char['global_timbre_ref'], 192) 
            for char in scene['characters']
        }

        scene_data = {
            "face": [], "env": [], "text": [], "arc": [], 
            "speaker": [], "ref": [], "target": [],
            "face_mask": [] 
        }

        for utt in utterances:
            u_idx, c_id = utt['utterance_index'], utt['char_id']
            basename = self._get_basename(scene_id, c_id, u_idx)
            prefix = f"{scene_id}@{c_id}"
            
            is_face = 1.0 if utt.get('if_face', True) else 0.0
            scene_data["face_mask"].append(torch.tensor(is_face))

            scene_data["target"].append(self._load_feat(os.path.join(self.feature_root, "emotion", f"{prefix}-emotion-{basename}.npy"), 1280))
            scene_data["face"].append(self._load_feat(os.path.join(self.feature_root, "face", f"{prefix}-face_desc-{basename}.npy"), 1024))
            scene_data["env"].append(self._load_feat(os.path.join(self.feature_root, "scene", f"{prefix}-scene-{basename}.npy"), 1024))
            scene_data["text"].append(self._load_feat(os.path.join(self.feature_root, "text", f"{prefix}-text-{basename}.npy"), 1024))
            scene_data["arc"].append(self._load_feat(os.path.join(self.feature_root, "arc", f"{prefix}-arc-{basename}.npy"), 1024))
            scene_data["speaker"].append(char_timbre_map[c_id])
            scene_data["ref"].append(self._load_rag_refs(basename))

        for k in scene_data:
            scene_data[k] = torch.stack(scene_data[k])
            
        return scene_data

def collate_fn(batch):
    keys, padded_batch = batch[0].keys(), {}
    lengths = torch.LongTensor([item['target'].size(0) for item in batch])
    max_len = lengths.max()
    padding_mask = torch.arange(max_len).expand(len(batch), max_len) >= lengths.unsqueeze(1)

    for k in keys:
        padded_batch[k] = pad_sequence([item[k] for item in batch], batch_first=True)
        
    padded_batch["src_key_padding_mask"] = padding_mask
    return padded_batch