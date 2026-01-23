import os
import torch
import torchaudio
import numpy as np
import json
import gc
from tqdm import tqdm
from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertModel
from omegaconf import OmegaConf

from timbre.indextts.s2mel.modules.campplus.DTDNN import CAMPPlus
from timbre.indextts.gpt.model_v2 import UnifiedVoice
from timbre.indextts.utils.checkpoint import load_checkpoint

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def build_semantic_model(path, local_model_path):
    semantic_model = Wav2Vec2BertModel.from_pretrained(local_model_path)
    semantic_model.eval()
    
    stat_mean_var = torch.load(path)
    semantic_mean = stat_mean_var["mean"]
    semantic_std = torch.sqrt(stat_mean_var["var"])
    return semantic_model, semantic_mean, semantic_std

def load_gpt_model(cfg, model_path, device):
    gpt_model = UnifiedVoice(**cfg.gpt)
    load_checkpoint(gpt_model, model_path)
    gpt_model = gpt_model.to(device)
    gpt_model.eval()
    return gpt_model

class UnifiedAudioExtractor:
    def __init__(self, cfg_path="timbre/checkpoints/config.yaml", model_dir="timbre/checkpoints"):
        self.device = DEVICE
        self.cfg = OmegaConf.load(cfg_path)
        self.model_dir = model_dir
        
        local_w2v_path = "timbre/w2v-bert-2.0"
        self.extract_features = SeamlessM4TFeatureExtractor.from_pretrained(local_w2v_path)
        
        w2v_stat_path = os.path.join(self.model_dir, self.cfg.w2v_stat)
        self.semantic_model, self.semantic_mean, self.semantic_std = build_semantic_model(
            w2v_stat_path, local_w2v_path
        )
        self.semantic_model = self.semantic_model.to(self.device)
        self.semantic_mean = self.semantic_mean.to(self.device)
        self.semantic_std = self.semantic_std.to(self.device)
        
        gpt_path = os.path.join(self.model_dir, self.cfg.gpt_checkpoint)
        self.gpt_model = load_gpt_model(self.cfg, gpt_path, self.device)
        
        campplus_ckpt_path = "timbre/campplus/campplus_cn_common.bin"
        self.campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
        self.campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
        self.campplus_model = self.campplus_model.to(self.device)
        self.campplus_model.eval()

    def _prepare_audio(self, audio_path, target_sr=16000):
        waveform, sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
            waveform = resampler(waveform)
        return waveform

    def extract_timbre_vector(self, audio_path, output_path=None):
        audio_16k = self._prepare_audio(audio_path, target_sr=16000).to(self.device)
        feat = torchaudio.compliance.kaldi.fbank(
            audio_16k, num_mel_bins=80, dither=0, sample_frequency=16000
        )
        feat = feat - feat.mean(dim=0, keepdim=True)
        with torch.no_grad():
            timbre_vector = self.campplus_model(feat.unsqueeze(0))
        if output_path:
            np.save(output_path, timbre_vector.cpu().numpy())
        return timbre_vector

    def extract_emotion_vector(self, audio_path, output_path=None):
        audio_16k = self._prepare_audio(audio_path, target_sr=16000)
        audio_inputs = audio_16k.squeeze().numpy()
        inputs = self.extract_features(audio_inputs, sampling_rate=16000, return_tensors="pt")
        input_values = inputs["input_features"].to(self.device)
        with torch.no_grad():
            outputs = self.semantic_model(input_values, output_hidden_states=True)
            semantic_emb = outputs.hidden_states[17]
            semantic_emb = (semantic_emb - self.semantic_mean) / self.semantic_std
            lengths = torch.LongTensor([semantic_emb.size(1)]).to(self.device)
            emotion_vector = self.gpt_model.get_emovec(semantic_emb, lengths)
        if output_path:
            np.save(output_path, emotion_vector.cpu().numpy())
        return emotion_vector

def process_audio(base_out="preprocessed_data/features"):
    extractor = UnifiedAudioExtractor()
    
    with open("data/dataset.json", 'r', encoding='utf-8') as f:
        index_data = json.load(f)

    timbre_dir = os.path.join(base_out, "timbre")
    emotion_dir = os.path.join(base_out, "emotion")
    os.makedirs(timbre_dir, exist_ok=True)
    os.makedirs(emotion_dir, exist_ok=True)

    for movie in index_data.get("movies", []):
        movie_path = movie.get("dataset_path")
        with open(os.path.join("data", movie_path), 'r', encoding='utf-8') as f:
            scene_data = json.load(f)

        for scene in tqdm(scene_data.get("scenes", []), desc=f"Processing Audio: {movie_path}"):
            scene_id = scene["scene_id"]
            for utt in scene.get("utterances", []):
                char_id = utt["char_id"].capitalize() 
                idx = f"{utt['utterance_index']:03d}"
                
                audio_rel_path = utt.get("audio_path", utt.get("video_path"))
                if not audio_rel_path: continue
                audio_path = os.path.join("data", audio_rel_path)
                
                prefix = f"{scene_id}@{char_id}"
                suffix = f"{prefix}_00_{scene_id}_{idx}.npy"
                timbre_path = os.path.join(timbre_dir, f"{prefix}-timbre-{suffix}")
                emotion_path = os.path.join(emotion_dir, f"{prefix}-emotion-{suffix}")

                if os.path.exists(timbre_path) and os.path.exists(emotion_path):
                    continue

                if not os.path.exists(audio_path): continue

                try:
                    extractor.extract_timbre_vector(audio_path, output_path=timbre_path)
                    extractor.extract_emotion_vector(audio_path, output_path=emotion_path)
                except Exception as e:
                    print(f"Error processing {audio_path}: {e}")
                finally:
                    torch.cuda.empty_cache()
                    gc.collect()

if __name__ == "__main__":
    process_audio()