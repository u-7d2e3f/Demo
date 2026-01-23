import os
import json
import torch
import numpy as np
import yaml
import soundfile as sf
import torchaudio
import librosa
from munch import Munch
from nltk.tokenize import word_tokenize
from collections import OrderedDict
import time

# 核心模型与工具导入
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

def preprocess(wave, mel_transform, mean, std):
    wave_tensor = torch.from_numpy(wave).float().to(device)
    mel_tensor = mel_transform(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor


class ActorSystem:
    def __init__(self, gate_config, pro_config, gate_ckpt, pro_ckpt):
        print("\n" + "="*60)
        print("🚀 [系统初始化] 正在加载导演 (Director) 与演员 (Actor) 协作架构...")
        start_init = time.time()
        
        with open(gate_config, 'r', encoding='utf-8') as f: self.gate_cfg = yaml.safe_load(f)
        with open(pro_config, 'r', encoding='utf-8') as f: self.pro_cfg = yaml.safe_load(f)
        
        self.feat_root = self.gate_cfg['path']['feature_root']
        self.pro_feat_root = self.pro_cfg['data_params']['feature_root_path']
        self.wav_root = self.pro_cfg.get('root_path', 'Dataset/preprocessed_data/wavs')

        # 演员 Agent 参数配置
        spec_p = self.pro_cfg['preprocess_params']['spect_params']
        self.mel_transform = torchaudio.transforms.MelSpectrogram(n_mels=80, **spec_p).to(device)
        self.m_params = self.pro_cfg['model_params']
        dist_cfg = self.m_params['diffusion']['dist']
        self.mean, self.std, self.sr = dist_cfg['mean'], dist_cfg['std'], self.pro_cfg['preprocess_params']['sr']

        # A. 加载导演 Agent (EmotionGateformer)
        print(" -> [1/3] 正在加载导演 Agent (EmotionGateformer)...")
        gm = self.gate_cfg['model']
        self.gateformer = EmotionGateformer(
            d_speaker=gm['d_speaker'], d_face=gm['d_face'], d_env=gm['d_env'],
            d_text=gm['d_text'], d_ref=gm['d_ref'], d_arc=gm['d_arc']
        ).to(device)
        gate_checkpoint = torch.load(gate_ckpt, map_location=device)
        self.gateformer.load_state_dict(gate_checkpoint.get('model_state_dict', gate_checkpoint))
        self.gateformer.eval()

        # B. 构建演员 Agent (ProDubber)
        print(" -> [2/3] 正在构建演员 Agent (ProDubber)...")
        self.model_pro = build_model(recursive_munch(self.pro_cfg['model_params']), 
                                     load_ASR_models(self.pro_cfg['ASR_path'], self.pro_cfg['ASR_config']), 
                                     load_F0_models(self.pro_cfg['F0_path']), load_plbert(self.pro_cfg['PLBERT_dir']))
        
        # C. 注入权重
        pro_ckpt_data = torch.load(pro_ckpt, map_location='cpu', weights_only=False)
        for key in self.model_pro:
            if key in pro_ckpt_data['net']:
                sd = pro_ckpt_data['net'][key]
                if any(k.startswith('module.') for k in sd.keys()): sd = OrderedDict((k[7:], v) for k, v in sd.items())
                self.model_pro[key].load_state_dict(sd, strict=False)
                self.model_pro[key].eval().to(device)

        self.sampler = DiffusionSampler(self.model_pro.diffusion.diffusion, sampler=ADPM2Sampler(),
                                        sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0), clamp=False)
        self.text_cleaner = TextCleaner()
        print(f"✅ [初始化完成] 耗时: {time.time() - start_init:.2f}s")
        print("="*60 + "\n")

    def _load_npy(self, path, dim=1024):
        return torch.from_numpy(np.load(path)).float().to(device).squeeze()

    def _get_rag_ref(self, base, top_k=3):
        p = f"{self.feat_root}/rag_indices/{base}-arc_top_100.json"
        if not os.path.exists(p): return torch.zeros(top_k, 1280).to(device)
        with open(p, 'r') as f: idxs = json.load(f)
        return torch.stack([self._load_npy(f"{self.feat_root}/emotion/{i['file_name']}", 1280) for i in idxs[:top_k]])

    def run_scene(self, master_json, movie_id, scene_id, output_dir):
        print(f"🎬 [场景启动] 电影: {movie_id} | 场景 ID: {scene_id}")
        with open(master_json, 'r') as f: master = json.load(f)
        movie_meta = next(m for m in master['movies'] if m['movie_id'] == movie_id)
        with open(os.path.join(os.path.dirname(master_json), movie_meta['dataset_path']), 'r') as f: movie_data = json.load(f)
        scene_data = next(s for s in movie_data['scenes'] if s['scene_id'] == scene_id)
        
        char_timbre_map = {c['char_id']: self._load_npy(c['global_timbre_ref'], 192).view(1, 1, -1) for c in scene_data['characters']}

        # ----------------------------------------------------------------------
        # Stage 1: 全局剧本研读 (建立全场时序记忆)
        # ----------------------------------------------------------------------
        print(f"📦 [Stage 1] 正在通读剧本，建立全场情感时序背景场...")
        all_raw = {"face":[], "env":[], "text":[], "spk":[], "face_mask":[]}
        for utt in scene_data['utterances']:
            # 1. 提取 if_face 标志位并保存
            is_face = 1.0 if utt.get('if_face', True) else 0.0
            all_raw["face_mask"].append(torch.tensor(is_face))

            prefix, base = f"{scene_id}@{utt['char_id']}", f"{scene_id}@{utt['char_id']}_00_{scene_id}_{utt['utterance_index']:03d}"
            all_raw["face"].append(self._load_npy(f"{self.feat_root}/face/{prefix}-face_desc-{base}.npy"))
            all_raw["env"].append(self._load_npy(f"{self.feat_root}/scene/{prefix}-scene-{base}.npy"))
            all_raw["text"].append(self._load_npy(f"{self.feat_root}/text/{prefix}-text-{base}.npy"))
            all_raw["spk"].append(char_timbre_map[utt['char_id']].squeeze())
            

        s_in = {k: torch.stack(v).unsqueeze(0) for k, v in all_raw.items()}
        
        f_mask = s_in['face_mask'].to(device).unsqueeze(-1)
        print(f"f_mask: {f_mask}")

        with torch.no_grad():
            f_e_all = self.gateformer.face_norm(self.gateformer.face_proj(s_in['face']))
            f_e_all = f_e_all * f_mask
            e_e_all = self.gateformer.env_norm(self.gateformer.env_proj(s_in['env']))
            t_e_all = self.gateformer.text_norm(self.gateformer.text_proj(s_in['text']))
            temp_in = self.gateformer._add_positional_encoding(self.gateformer.input_proj_temporal(torch.cat([f_e_all, e_e_all, t_e_all, s_in['spk']], dim=-1)))
            spk_bias = self.gateformer._generate_similarity_bias_mask(s_in['spk']).repeat_interleave(self.gateformer.nhead, dim=0)
            temp_out_all = self.gateformer.temporal_encoder(temp_in, mask=spk_bias)
            
           

            # ----------------------------------------------------------------------
            # Stage 2: 逐句循环演绎 (广播融合逻辑)
            # ----------------------------------------------------------------------
            curr_tgt_emb = torch.zeros(1, 1, self.gateformer.d_model, device=device)
            os.makedirs(output_dir, exist_ok=True)

            print(f"\n🎭 [Stage 2] 开始逐句演绎循环推理...")
            for t, utt in enumerate(scene_data['utterances']):
                u_idx, c_id = utt['utterance_index'], utt['char_id']
                prefix, base_id = f"{scene_id}@{c_id}", f"{scene_id}@{c_id}_00_{scene_id}_{u_idx:03d}"
                print(f"--- [台词 {t+1}/{len(scene_data['utterances'])}] 角色: {c_id} ---")

                # 1. 导演 Agent 动态决策 (完全对齐 EmotionGateformer 广播逻辑)
                print(f"   -> 导演正在进行广播融合决策...")
                ref_t = self._get_rag_ref(base_id).unsqueeze(0).unsqueeze(0) 
                arc_t = self._load_npy(f"{self.feat_root}/arc/{prefix}-arc-{base_id}.npy").view(1, 1, 1024) 
                a_e_t = self.gateformer.arc_norm(self.gateformer.arc_proj(arc_t))
                
                # 提取当前时刻切片特征
                f_e_t, e_e_t, t_e_t = f_e_all[:, t:t+1, :], e_e_all[:, t:t+1, :], t_e_all[:, t:t+1, :]
                
                # 瞬时细节交互 (Feature Encoder 只看当前 $t$)
                dyn_q_t = self.gateformer.context_to_query(torch.cat([f_e_t, t_e_t, a_e_t], dim=-1))
                r_fused_t = self.gateformer._fusion_ref_dynamic(ref_t, dyn_q_t)
                feat_stack_t = torch.stack([f_e_t, e_e_t, t_e_t, r_fused_t, a_e_t], dim=2)
                feat_enc_t = self.gateformer.feature_encoder(self.gateformer._add_positional_encoding(feat_stack_t.view(1, 5, 512)))
                feat_out_t = self.gateformer.feature_aggregator(feat_enc_t.flatten(start_dim=1).unsqueeze(1)) # [B, 1, D]

                # 广播融合核心：将当前细节 (feat_out_t) 与全场记忆 (temp_out_all) 融合
                # 这里不切片 temp_out_all，而是让当前细节作用于全局剧本背景场
                gate_t = self.gateformer.gate_sigmoid(self.gateformer.gate_w1(temp_out_all) + self.gateformer.gate_w2(feat_out_t))
                memory_t = gate_t * temp_out_all + (1 - gate_t) * feat_out_t
                print(f"      [门控监控] Gate 均值: {gate_t.mean().item():.4f} (均值大代表更依从剧本记忆)")

                # 解码当前情感向量
                dec_in_step = self.gateformer._add_positional_encoding(self.gateformer.decoder_input_proj(torch.cat([curr_tgt_emb, s_in['spk'][:, :curr_tgt_emb.size(1), :]], dim=-1)))
                out_dec = self.gateformer.decoder(dec_in_step, memory_t, tgt_mask=self.gateformer._generate_causal_mask(curr_tgt_emb.size(1)).to(device))
                pred_1280 = self.gateformer.output_projection(out_dec[:, -1:, :])
                e_v_raw = pred_1280
                
                # 2. 演员 Agent 表现合成
                wave_ref, _ = librosa.load(os.path.join(self.wav_root, f"{base_id}.wav"), sr=self.sr)
                target_mel_len = preprocess(wave_ref, self.mel_transform, self.mean, self.std).shape[-1]
                
                atm_feat = self._load_npy(os.path.join(self.pro_feat_root, "scene", f"{prefix}-scene-{base_id}.npy")).view(1, 1024)
                if utt.get('if_face', True):
                    lip_feat = torch.from_numpy(np.load(os.path.join(self.pro_feat_root, "extrated_embedding_V2C_gray", f"{prefix}-face-{base_id}.npy"))).float().to(device).unsqueeze(0)
                    emo_feat = torch.from_numpy(np.load(os.path.join(self.pro_feat_root, "VA_features", f"{prefix}-feature-{base_id}.npy"))).float().to(device).unsqueeze(0)
                else:
                    lip_feat, emo_feat = None, None
                    
                wav = self._dub_utterance(utt['text'], char_timbre_map[c_id], e_v_raw, lip_feat, emo_feat, atm_feat, target_mel_len)
                sf.write(os.path.join(output_dir, f"{base_id}.wav"), wav, self.sr)
                print(f"   ✅ 完成！配音音频已保存。")

                # 3. 自回归更新记忆
                if t < len(scene_data['utterances']) - 1:
                    curr_tgt_emb = torch.cat([curr_tgt_emb, self.gateformer.tgt_proj(pred_1280)], dim=1)

    def _dub_utterance(self, text, t_v_raw, e_v_raw, lip_feat, emo_feat, atm_feat, target_mel_len):
        s_dim = self.m_params['style_dim']

        with torch.no_grad():
            # A. 风格提取
            ref_s_id = self.model_pro.style_encoder(t_v_raw).view(1, -1)      
            ref_s_emo = self.model_pro.predictor_encoder(e_v_raw).view(1, -1) 
            ref_s = torch.cat([ref_s_id, ref_s_emo], dim=1) # 640D

            # B. 文本编码
            ps = global_phonemizer.phonemize([text.replace('"', '').strip()])
            tokens = self.text_cleaner(' '.join(word_tokenize(ps[0])))
            tokens.insert(0, 0)
            tokens_tensor = torch.LongTensor([tokens]).to(device).view(1, -1)
            input_lengths = torch.LongTensor([tokens_tensor.shape[-1]]).to(device)
            text_mask = length_to_mask(input_lengths).to(device)

            bert_dur = self.model_pro.bert(tokens_tensor, attention_mask=(~text_mask).int())
            prosody_phoneme_feature = self.model_pro.bert_encoder(bert_dur).transpose(-1, -2) 

            # C. 视觉融合与时长预测
            if lip_feat is not None and emo_feat is not None:
                print("   [ 时长 ] -> 视觉驱动 (对口型模式)")
                v_mask = length_to_mask(torch.LongTensor([emo_feat.shape[1]])).to(device)
                prosody_phoneme_feature_emotion = self.model_pro.prosody_fusion(prosody_phoneme_feature, text_mask, v_mask, emo_feat, atm_feat) + prosody_phoneme_feature
                duration = self.model_pro.duration_predictor_visual(prosody_phoneme_feature, lip_feat, input_lengths, text_mask, v_mask)
            else:
                print("   [ 时长 ] -> 音频驱动 (自然语速模式)")
                prosody_phoneme_feature_emotion = prosody_phoneme_feature
                duration = self.model_pro.duration_predictor_audio(prosody_phoneme_feature, ref_s_emo, input_lengths, text_mask)

            # D. 扩散采样风格
            print("   -> 正在执行扩散采样 (Diffusion Sampling)...")
            s_pred = self.sampler(noise=torch.randn((1, 640)).unsqueeze(1).to(device), 
                                 embedding=prosody_phoneme_feature_emotion.transpose(-1, -2),
                                 embedding_scale=1, num_steps=5).squeeze(1)
            
            s_mixed_emo = (0.3 * s_pred[:, s_dim:] + 0.7 * ref_s[:, s_dim:]).view(1, -1)
            ref_id_vec = ref_s[:, :s_dim]

            # E. 时长归一化与分配
            duration = torch.sigmoid(duration).sum(axis=-1)
            duration = ( (duration / duration.sum()) * target_mel_len ) / 2
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)
            pred_dur[0] += (target_mel_len / 2) - pred_dur.sum()

            # F. 对齐矩阵生成
            pred_aln_trg = torch.zeros(input_lengths.item(), int(pred_dur.sum().data)).to(device)
            curr = 0
            for i in range(pred_aln_trg.size(0)):
                pred_aln_trg[i, curr:curr + int(pred_dur[i].data)] = 1
                curr += int(pred_dur[i].data)

            # G. 声学预测与 HiFi-GAN 合成
            print("   -> 正在生成声学特征并解码波形...")
            p_en = self.model_pro.predictor(prosody_phoneme_feature_emotion, s_mixed_emo, input_lengths, pred_aln_trg.unsqueeze(0), text_mask)
            asr = (self.model_pro.text_encoder(tokens_tensor, input_lengths, text_mask) @ pred_aln_trg.unsqueeze(0))
            
            F0_pred, N_pred = self.model_pro.predictor.F0Ntrain(p_en, s_mixed_emo)
            out = self.model_pro.decoder(asr, F0_pred, N_pred, ref_id_vec)
            
            return out.squeeze().cpu().numpy()[..., :-50]

# ==============================================================================
# [ 3. 运行测试入口 ]
# ==============================================================================
if __name__ == "__main__":
    system = ActorSystem(
        gate_config="EmotionGateformer/Configs/Config.yml", 
        pro_config="ProDubber/output/stage2/config.yml",
        gate_ckpt="EmotionGateformer/output/checkpoints/gateformer_ep10.pth",
        pro_ckpt="ProDubber/output/stage2/ckpt/epoch_2nd_210.pth"
    )
    system.run_scene(
        master_json="Dataset/data/dataset.json", 
        movie_id="KFP1", 
        scene_id="KFP001", 
        output_dir="results/KFP001"
    )