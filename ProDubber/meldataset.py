import os
import os.path as osp
import numpy as np
import random
import pandas as pd
import soundfile as sf
import librosa
import torch
import torchaudio
from torch.utils.data import DataLoader
import torch.nn.functional as F

_pad = "$"
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)
dicts = {s: i for i, s in enumerate(symbols)}

class TextCleaner:
    def __init__(self, dummy=None): 
        self.word_index_dictionary = dicts
    def __call__(self, text):
        return [self.word_index_dictionary[char] for char in text if char in self.word_index_dictionary]

to_mel = torchaudio.transforms.MelSpectrogram(n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
mean, std = -4, 4

def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor

class FilePathDataset_Stage1(torch.utils.data.Dataset):
    def __init__(self, data_list, root_path, feature_root_path, validation=False, use_random_ref=True, **kwargs):
        _data_list = [l.strip().split('|') for l in data_list]
        self.data_list = [data if len(data) == 3 else (*data, 0) for data in _data_list]
        self.text_cleaner = TextCleaner()
        self.df = pd.DataFrame(self.data_list)
        self.root_path = root_path
        self.feature_root_path = feature_root_path
        self.use_random_ref = use_random_ref
        self.is_val = validation

    def __len__(self): 
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        path = data[0]
        wave, text_tensor, speaker_id = self._load_tensor(data)
        
        mel_tensor = preprocess(wave).squeeze() 
        acoustic_feature = mel_tensor[:, :(mel_tensor.size(1) - mel_tensor.size(1) % 2)]
        
        basename = path.split('/')[-1].replace('.wav', '')
        speaker = basename.split('_')[0]
        t_v = torch.from_numpy(np.load(osp.join(self.feature_root_path, "timbre", f"{speaker}-timbre-{basename}.npy"))).float()

        if not self.use_random_ref or self.is_val:
            ref_t_v = t_v
        else:
            ref_data = (self.df[self.df[2] == str(speaker_id)]).sample(n=1).iloc[0].tolist()
            ref_bn = ref_data[0].split('/')[-1].replace('.wav', '')
            ref_spk = ref_bn.split('_')[0]
            ref_t_v = torch.from_numpy(np.load(osp.join(self.feature_root_path, "timbre", f"{ref_spk}-timbre-{ref_bn}.npy"))).float()
        return (speaker_id, acoustic_feature, text_tensor, t_v, ref_t_v, wave)

    def _load_tensor(self, data):
        wave_path, text, spk_id = data
        wave, sr = sf.read(osp.join(self.root_path, wave_path))
        if len(wave.shape) == 2: 
            wave = wave[:, 0]
        if sr != 24000: 
            wave = librosa.resample(wave, orig_sr=sr, target_sr=24000)
        wave = np.concatenate([np.zeros([5000]), wave, np.zeros([5000])], axis=0)
        return wave, torch.LongTensor([0]+self.text_cleaner(text)+[0]), int(spk_id)

class Collater_Stage1(object):
    def __call__(self, batch):
        batch = sorted(batch, key=lambda x: x[1].shape[1], reverse=True)
        batch_size = len(batch)

        max_mel_len = max([b[1].shape[1] for b in batch])
        max_text_len = max([b[2].shape[0] for b in batch])

        mels = torch.zeros((batch_size, 80, max_mel_len)).float()
        texts = torch.zeros((batch_size, max_text_len)).long()
        t_vecs = torch.zeros((batch_size, 192)).float()
        ref_t_vecs = torch.zeros((batch_size, 192)).float()
        input_lengths = torch.zeros(batch_size).long()
        output_lengths = torch.zeros(batch_size).long()
        waves = [None] * batch_size

        for bid, (sid, acoustic, text, tv, rtv, wave) in enumerate(batch):
            mels[bid, :, :acoustic.size(1)] = acoustic
            texts[bid, :text.size(0)] = text
            input_lengths[bid] = text.size(0)
            output_lengths[bid] = acoustic.size(1)
            t_vecs[bid], ref_t_vecs[bid] = tv, rtv
            waves[bid] = wave

        return waves, texts, input_lengths, t_vecs, input_lengths, mels, output_lengths, ref_t_vecs

class FilePathDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, root_path, feature_root_path, validation=False, use_random_ref=True, **kwargs):
        _data_list = [l.strip().split('|') for l in data_list]
        self.data_list = [data if len(data) == 3 else (*data, 0) for data in _data_list]
        self.text_cleaner = TextCleaner()
        self.df = pd.DataFrame(self.data_list)
        self.root_path = root_path
        self.feature_root_path = feature_root_path
        self.use_random_ref = use_random_ref
        self.is_val = validation

    def __len__(self): 
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        path = data[0]
        wave, text_tensor, speaker_id = self._load_tensor(data)
        
        mel_tensor = preprocess(wave).squeeze()
        acoustic_feature = mel_tensor[:, :(mel_tensor.size(1) - mel_tensor.size(1) % 2)]
        
        basename = path.split('/')[-1].replace('.wav', '')
        speaker = basename.split('_00')[0] 

        t_v = torch.from_numpy(np.load(osp.join(self.feature_root_path, "timbre", f"{speaker}-timbre-{basename}.npy"))).float().squeeze()
        
        e_v_path = osp.join(self.feature_root_path, "emotion", f"{speaker}-emotion-{basename}.npy")
        e_v = torch.from_numpy(np.load(e_v_path)).float().squeeze()
    
        if not self.use_random_ref or self.is_val:
            ref_t_v = t_v
        else:
            ref_data = (self.df[self.df[2] == str(speaker_id)]).sample(n=1).iloc[0].tolist()
            ref_bn = ref_data[0].split('/')[-1].replace('.wav', '')
            ref_spk = ref_bn.split('_')[0]
            ref_t_v = torch.from_numpy(np.load(osp.join(self.feature_root_path, "timbre", f"{ref_spk}-timbre-{ref_bn}.npy"))).float()
         
        lip = self._load_npy("extrated_embedding_V2C_gray", f"{speaker}-face-{basename}.npy", dim=512)
        v_emo = self._load_npy("VA_features", f"{speaker}-feature-{basename}.npy", dim=256)
        atm = self._load_npy("scene", f"{speaker}-scene-{basename}.npy", dim=1024) 

        return (speaker_id, acoustic_feature, text_tensor, t_v, e_v, lip, v_emo, atm, ref_t_v, wave)

    def _load_tensor(self, data):
        wave_path, text, spk_id = data
        wave, sr = sf.read(osp.join(self.root_path, wave_path))
        if len(wave.shape) == 2: 
            wave = wave[:, 0]
        if sr != 24000: 
            wave = librosa.resample(wave, orig_sr=sr, target_sr=24000)
        wave = np.concatenate([np.zeros([5000]), wave, np.zeros([5000])], axis=0)
        return wave, torch.LongTensor([0]+self.text_cleaner(text)+[0]), spk_id

    def _load_npy(self, folder, filename, dim=256): 
        path = osp.join(self.feature_root_path, folder, filename)
        return torch.from_numpy(np.load(path)).float() if osp.exists(path) else torch.zeros(dim)

class Collater(object):
    def __call__(self, batch):
        batch = sorted(batch, key=lambda x: x[1].shape[1], reverse=True)
        batch_size = len(batch)

        max_mel_len = max([b[1].shape[1] for b in batch])
        max_txt_len = max([b[2].shape[0] for b in batch])
        max_vis_len = max([b[5].shape[0] for b in batch])

        mels = torch.zeros((batch_size, 80, max_mel_len)).float()
        texts = torch.zeros((batch_size, max_txt_len)).long()
        t_vecs = torch.zeros((batch_size, 192)).float()
        e_vecs = torch.zeros((batch_size, 1280)).float() 
        ref_t_vecs = torch.zeros((batch_size, 192)).float()
        
        lip_f = torch.zeros((batch_size, max_vis_len, 512)).float()
        v_emo_f = torch.zeros((batch_size, max_vis_len, 256)).float()
        atm_f = torch.zeros((batch_size, 1024)).float()
        
        input_lens = torch.zeros(batch_size).long()
        output_lens = torch.zeros(batch_size).long()
        vis_lens = torch.zeros(batch_size).long()
        waves = [None] * batch_size

        for bid, (sid, acoustic, txt, tv, ev, lip, ve, atm, rtv, wave) in enumerate(batch):
            mels[bid, :, :acoustic.size(1)] = acoustic
            texts[bid, :txt.size(0)] = txt
            input_lens[bid] = txt.size(0)
            output_lens[bid] = acoustic.size(1)
            
            t_vecs[bid] = tv
            e_vecs[bid] = ev 
            ref_t_vecs[bid] = rtv
            
            lip_f[bid, :lip.size(0), :] = lip
            v_emo_f[bid, :ve.size(0), :] = ve
            atm_f[bid] = atm
            vis_lens[bid] = lip.size(0)
            waves[bid] = wave

        return waves, texts, input_lens, t_vecs, e_vecs, mels, output_lens, ref_t_vecs, lip_f, v_emo_f, atm_f, vis_lens

def build_dataloader_Stage1(path_list, root_path, feature_root_path, batch_size=4, num_workers=1, **kwargs):
    dataset = FilePathDataset_Stage1(path_list, root_path, feature_root_path, **kwargs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=Collater_Stage1(), num_workers=num_workers)

def build_dataloader(path_list, root_path, feature_root_path, batch_size=4, num_workers=1, **kwargs):
    dataset = FilePathDataset(path_list, root_path, feature_root_path, **kwargs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=Collater(), num_workers=num_workers)