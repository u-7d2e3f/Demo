import os
import torch
import numpy as np
import yaml
import torchaudio
import librosa
import json
from munch import Munch
from collections import OrderedDict
from nltk.tokenize import word_tokenize

from ProDubber.models import build_model, load_ASR_models, load_F0_models
from ProDubber.Utils.PLBERT.util import load_plbert
from ProDubber.Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule
from EmotionGateformer.EmotionGateformer import EmotionGateformer
from ProDubber.text_utils import TextCleaner
import phonemizer
global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True, with_stress=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def recursive_munch(d):
    if isinstance(d, dict): return Munch({k: recursive_munch(v) for k, v in d.items()})
    elif isinstance(d, list): return [recursive_munch(i) for i in d]
    return d

def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask

class ActorAgent:
    def __init__(self, gate_config, pro_config, gate_ckpt, pro_ckpt):
        with open(gate_config, 'r', encoding='utf-8') as f: self.gate_cfg = yaml.safe_load(f)
        with open(pro_config, 'r', encoding='utf-8') as f: self.pro_cfg = yaml.safe_load(f)
        
        self.feat_root = self.gate_cfg['path']['feature_root']
        self.pro_feat_root = self.pro_cfg['data_params']['feature_root_path']
        self.wav_root = self.pro_cfg.get('root_path', 'Dataset/preprocessed_data/wavs')

        spec_p = self.pro_cfg['preprocess_params']['spect_params']
        self.mel_transform = torchaudio.transforms.MelSpectrogram(n_mels=80, **spec_p).to(device)
        self.m_params = self.pro_cfg['model_params']
        dist_cfg = self.m_params['diffusion']['dist']
        self.mean, self.std, self.sr = dist_cfg['mean'], dist_cfg['std'], self.pro_cfg['preprocess_params']['sr']

        self._init_gateformer(gate_ckpt)
        self._init_produbber(pro_ckpt)
        
        self.sampler = DiffusionSampler(self.model_pro.diffusion.diffusion, sampler=ADPM2Sampler(),
                                        sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0), clamp=False)
        self.text_cleaner = TextCleaner()
        self.phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True, with_stress=True)

    def _init_gateformer(self, ckpt):
        gm = self.gate_cfg['model']
        self.gateformer = EmotionGateformer(
            d_speaker=gm['d_speaker'], d_face=gm['d_face'], d_env=gm['d_env'],
            d_text=gm['d_text'], d_ref=gm['d_ref'], d_arc=gm['d_arc']
        ).to(device)
        checkpoint = torch.load(ckpt, map_location=device)
        self.gateformer.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
        self.gateformer.eval()

    def _init_produbber(self, ckpt):
        self.model_pro = build_model(recursive_munch(self.pro_cfg['model_params']), 
                                     load_ASR_models(self.pro_cfg['ASR_path'], self.pro_cfg['ASR_config']), 
                                     load_F0_models(self.pro_cfg['F0_path']), load_plbert(self.pro_cfg['PLBERT_dir']))
        pro_ckpt_data = torch.load(ckpt, map_location='cpu', weights_only=False)
        for key in self.model_pro:
            if key in pro_ckpt_data['net']:
                sd = pro_ckpt_data['net'][key]
                if any(k.startswith('module.') for k in sd.keys()): sd = OrderedDict((k[7:], v) for k, v in sd.items())
                self.model_pro[key].load_state_dict(sd, strict=False)
                self.model_pro[key].eval().to(device)

    def preprocess(self, wave, mel_transform, mean, std):
        wave_tensor = torch.from_numpy(wave).float().to(device)
        mel_tensor = mel_transform(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
        return mel_tensor

    def _load_npy(self, path, dim=1024):
        return torch.from_numpy(np.load(path)).float().to(device).squeeze()

    def _dub_utterance(self, text, t_v_raw, e_v_raw, lip_feat, emo_feat, atm_feat, target_mel_len):
        s_dim = self.m_params['style_dim']

        with torch.no_grad():
            ref_s_id = self.model_pro.style_encoder(t_v_raw).view(1, -1)      
            ref_s_emo = self.model_pro.predictor_encoder(e_v_raw).view(1, -1) 
            ref_s = torch.cat([ref_s_id, ref_s_emo], dim=1)

            ps = global_phonemizer.phonemize([text.replace('"', '').strip()])
            tokens = self.text_cleaner(' '.join(word_tokenize(ps[0])))
            tokens.insert(0, 0)
            tokens_tensor = torch.LongTensor([tokens]).to(device).view(1, -1)
            input_lengths = torch.LongTensor([tokens_tensor.shape[-1]]).to(device)
            text_mask = length_to_mask(input_lengths).to(device)

            bert_dur = self.model_pro.bert(tokens_tensor, attention_mask=(~text_mask).int())
            prosody_phoneme_feature = self.model_pro.bert_encoder(bert_dur).transpose(-1, -2) 

            if lip_feat is not None and emo_feat is not None:
                v_mask = length_to_mask(torch.LongTensor([emo_feat.shape[1]])).to(device)
                prosody_phoneme_feature_emotion = self.model_pro.prosody_fusion(prosody_phoneme_feature, text_mask, v_mask, emo_feat, atm_feat) + prosody_phoneme_feature
                duration = self.model_pro.duration_predictor_visual(prosody_phoneme_feature, lip_feat, input_lengths, text_mask, v_mask)
            else:
                prosody_phoneme_feature_emotion = prosody_phoneme_feature
                duration = self.model_pro.duration_predictor_audio(prosody_phoneme_feature, ref_s_emo, input_lengths, text_mask)

            s_pred = self.sampler(noise=torch.randn((1, 640)).unsqueeze(1).to(device), 
                                  embedding=prosody_phoneme_feature_emotion.transpose(-1, -2),
                                  embedding_scale=1, num_steps=5).squeeze(1)
            
            s_mixed_emo = (0.3 * s_pred[:, s_dim:] + 0.7 * ref_s[:, s_dim:]).view(1, -1)
            ref_id_vec = ref_s[:, :s_dim]

            duration = torch.sigmoid(duration).sum(axis=-1)
            duration = ( (duration / duration.sum()) * target_mel_len ) / 2
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)
            pred_dur[0] += (target_mel_len / 2) - pred_dur.sum()

            pred_aln_trg = torch.zeros(input_lengths.item(), int(pred_dur.sum().data)).to(device)
            curr = 0
            for i in range(pred_aln_trg.size(0)):
                pred_aln_trg[i, curr:curr + int(pred_dur[i].data)] = 1
                curr += int(pred_dur[i].data)

            p_en = self.model_pro.predictor(prosody_phoneme_feature_emotion, s_mixed_emo, input_lengths, pred_aln_trg.unsqueeze(0), text_mask)
            asr = (self.model_pro.text_encoder(tokens_tensor, input_lengths, text_mask) @ pred_aln_trg.unsqueeze(0))
            
            F0_pred, N_pred = self.model_pro.predictor.F0Ntrain(p_en, s_mixed_emo)
            out = self.model_pro.decoder(asr, F0_pred, N_pred, ref_id_vec)
            
            return out.squeeze().cpu().numpy()[..., :-50]