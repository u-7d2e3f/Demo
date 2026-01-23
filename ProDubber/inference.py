import os
import torch
import numpy as np
import yaml
import argparse
import librosa
import torchaudio
import soundfile as sf
from munch import Munch
from nltk.tokenize import word_tokenize
from collections import OrderedDict
import random
import phonemizer

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.serialization.add_safe_globals([getattr])

from ProDubber.models import *
from ProDubber.utils import *
from ProDubber.text_utils import TextCleaner
from ProDubber.Utils.PLBERT.util import load_plbert
from ProDubber.Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule

device = 'cuda' if torch.cuda.is_available() else 'cpu'
textclenaer = TextCleaner()
global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True, with_stress=True)

def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask

def preprocess(wave, mel_transform, mean, std):
    wave_tensor = torch.from_numpy(wave).float()
    wave_tensor = wave_tensor.to(device)
    mel_tensor = mel_transform(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor

def run_custom_pro_inference(args, model, sampler, config, mel_transform):
    m_params = config['model_params']
    dist_cfg = m_params['diffusion']['dist']
    mean, std, sr = dist_cfg['mean'], dist_cfg['std'], config['preprocess_params']['sr']
    s_dim, e_dim = m_params['style_dim'], m_params['audio_emotion_dim']

    wave_ref, _ = librosa.load(args.ref_wav_for_len, sr=sr)
    target_mel_len = preprocess(wave_ref, mel_transform, mean, std).shape[-1]
    
    t_v_raw = torch.from_numpy(np.load(args.timbre_path)).float().to(device).view(1, 1, -1)
    e_v_raw = torch.from_numpy(np.load(args.emotion_path)).float().to(device).view(1, 1, -1)
    
    with torch.no_grad():
        ref_s_id = model.style_encoder(t_v_raw).view(1, -1)      
        ref_s_emo = model.predictor_encoder(e_v_raw).view(1, -1) 
        ref_s = torch.cat([ref_s_id, ref_s_emo], dim=1)

    ps = global_phonemizer.phonemize([args.text.replace('"', '').strip()])
    tokens = textclenaer(' '.join(word_tokenize(ps[0])))
    tokens.insert(0, 0)
    tokens_tensor = torch.LongTensor([tokens]).to(device).view(1, -1)
    input_lengths = torch.LongTensor([tokens_tensor.shape[-1]]).to(device)
    text_mask = length_to_mask(input_lengths).to(device)

    with torch.no_grad():
        bert_dur = model.bert(tokens_tensor, attention_mask=(~text_mask).int())
        prosody_phoneme_feature = model.bert_encoder(bert_dur).transpose(-1, -2) 

        root = config['data_params']['feature_root_path']
        speaker = args.target_id.split('_00')[0] if '_00' in args.target_id else args.target_id.split('@')[1].split('_')[0]
        lip_feat = torch.from_numpy(np.load(os.path.join(root, "extrated_embedding_V2C_gray", f"{speaker}-face-{args.target_id}.npy"))).float().to(device).unsqueeze(0)
        emo_feat = torch.from_numpy(np.load(os.path.join(root, "VA_feature", f"{speaker}-feature-{args.target_id}.npy"))).float().to(device).unsqueeze(0)
        atm_feat = torch.from_numpy(np.load(os.path.join(root, "emos", f"{speaker}-emo-{args.target_id}.npy"))).float().to(device).unsqueeze(0)
        v_mask = length_to_mask(torch.LongTensor([emo_feat.shape[1]])).to(device)
        
        if args.dur_mode == 'visual':
            prosody_phoneme_feature_emotion = model.prosody_fusion(prosody_phoneme_feature, text_mask, v_mask, emo_feat, atm_feat) + prosody_phoneme_feature
            duration = model.duration_predictor_visual(prosody_phoneme_feature, lip_feat, input_lengths, text_mask, v_mask)
        else:
            prosody_phoneme_feature_emotion = prosody_phoneme_feature
            duration = model.duration_predictor_audio(prosody_phoneme_feature, ref_s_emo, input_lengths, text_mask)

        s_pred = sampler(noise=torch.randn((1, 640)).unsqueeze(1).to(device), 
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

        p_en = model.predictor(prosody_phoneme_feature_emotion, s_mixed_emo, input_lengths, pred_aln_trg.unsqueeze(0), text_mask)
        asr = (model.text_encoder(tokens_tensor, input_lengths, text_mask) @ pred_aln_trg.unsqueeze(0))
        
        F0_pred, N_pred = model.predictor.F0Ntrain(p_en, s_mixed_emo)
        out = model.decoder(asr, F0_pred, N_pred, ref_id_vec)
        wav = out.squeeze().cpu().numpy()[..., :-50]

    os.makedirs(args.save_path, exist_ok=True)
    out_file = os.path.join(args.save_path, f"final.wav")
    sf.write(out_file, wav, sr)
