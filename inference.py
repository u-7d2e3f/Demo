import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import soundfile as sf
import argparse
import yaml
import librosa
import shutil
from ActorAgent import ActorAgent
from DirectorAgent import DirectorAgent
from ReviewerAgent import ReviewerAgent 
from transformers import RobertaTokenizer, RobertaForSequenceClassification

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class DynamicRAGManager:
    def __init__(self, rag_root, device):
        self.rag_root = rag_root 
        self.device = device
        self._load_gallery()

    def _load_gallery(self):
        self.arc_features, self.emo_file_names = [], []
        arc_dir = os.path.join(self.rag_root, "arc")
        if not os.path.exists(arc_dir): os.makedirs(arc_dir)
        for f in sorted(os.listdir(arc_dir)):
            if f.endswith(".npy"):
                vec = np.load(os.path.join(arc_dir, f)).squeeze()
                self.arc_features.append(torch.from_numpy(vec).float())
                self.emo_file_names.append(f.replace("-arc-", "-emotion-"))
        self.arc_matrix = torch.stack(self.arc_features).to(self.device) if self.arc_features else torch.empty(0, 1024).to(self.device)

    def update_database(self, arc_vec, emo_vec, base_id, scene_id, c_id):
        arc_path = os.path.join(self.rag_root, "arc", f"{scene_id}@{c_id}-arc-{base_id}.npy")
        emo_path = os.path.join(self.rag_root, "emotion", f"{scene_id}@{c_id}-emotion-{base_id}.npy")
        os.makedirs(os.path.dirname(arc_path), exist_ok=True)
        os.makedirs(os.path.dirname(emo_path), exist_ok=True)
        np.save(arc_path, arc_vec.cpu().numpy())
        np.save(emo_path, emo_vec.cpu().numpy())
        self._load_gallery() 

    def retrieve_top_k(self, current_arc_vec, top_k):
        if self.arc_matrix.size(0) == 0: return torch.zeros(top_k, 1280).to(self.device)
        query = F.normalize(current_arc_vec.view(1, 1024), p=2, dim=1)
        sims = torch.mm(query, F.normalize(self.arc_matrix, p=2, dim=1).t()).squeeze(0)
        top_idx = torch.topk(sims, min(top_k, self.arc_matrix.size(0))).indices
        refs = [torch.from_numpy(np.load(os.path.join(self.rag_root, "emotion", self.emo_file_names[i]))).float().to(self.device).squeeze() for i in top_idx]
        while len(refs) < top_k: refs.append(torch.zeros(1280).to(self.device))
        return torch.stack(refs)

class TextEmotionExtractor:
    def __init__(self, model_path="Dataset/emos/emotion-english-roberta-large"):
        self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
        self.model = RobertaForSequenceClassification.from_pretrained(model_path).to(device)
        self.model.eval()
    def get_1024d_vector(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            return self.model(**inputs, output_hidden_states=True).hidden_states[-1][:, 0, :].view(1, 1, 1024)

def load_npy(path):
    return torch.from_numpy(np.load(path)).float().to(device).squeeze()

def run_inference(args):
    with open(args.gate_config, 'r') as f: full_cfg = yaml.safe_load(f)
    rag_gallery_path, feat_root = full_cfg['path']['ref_footage_root'], full_cfg['path']['feature_root']

    actor = ActorAgent(args.gate_config, args.pro_config, args.gate_ckpt, args.pro_ckpt)
    director = DirectorAgent(api_key=args.api_key)
    reviewer = ReviewerAgent() 
    extractor, rag_engine = TextEmotionExtractor(), DynamicRAGManager(rag_gallery_path, device)
    
    with open(args.master_json, 'r') as f: master = json.load(f)
    movie_meta = next(m for m in master['movies'] if m['movie_id'] == args.movie_id)
    dataset_path = os.path.join(os.path.dirname(args.master_json), movie_meta['dataset_path'])
    with open(dataset_path, 'r') as f: movie_data = json.load(f)
    scene_data = next(s for s in movie_data['scenes'] if s['scene_id'] == args.scene_id)
    char_timbre_map = {c['char_id']: load_npy(c['global_timbre_ref']).view(1, 1, -1) for c in scene_data['characters']}

    os.makedirs(args.save_path, exist_ok=True)
    step_log_file = os.path.join(args.save_path, f"step_details_{args.scene_id}.jsonl")
    final_log_file = os.path.join(args.save_path, f"final_summary_{args.scene_id}.jsonl")

    all_raw = {"face":[], "env":[], "text":[], "spk":[], "face_mask":[]}
    for utt in scene_data['utterances']:
        p, b = f"{args.scene_id}@{utt['char_id']}", f"{args.scene_id}@{utt['char_id']}_00_{args.scene_id}_{utt['utterance_index']:03d}" 
        all_raw["face_mask"].append(torch.tensor(1.0 if utt.get('if_face', True) else 0.0))
        all_raw["face"].append(load_npy(os.path.join(feat_root, "face", f"{p}-face_desc-{b}.npy")))
        all_raw["env"].append(load_npy(os.path.join(feat_root, "scene", f"{p}-scene-{b}.npy")))
        all_raw["text"].append(load_npy(os.path.join(feat_root, "text", f"{p}-text-{b}.npy")))
        all_raw["spk"].append(char_timbre_map[utt['char_id']].squeeze())

    s_in = {k: torch.stack(v).unsqueeze(0) for k, v in all_raw.items()}
    f_mask = s_in['face_mask'].to(device).unsqueeze(-1)

    with torch.no_grad():
        gf = actor.gateformer
        f_e_all = gf.face_norm(gf.face_proj(s_in['face'])) * f_mask 
        e_e_all, t_e_all = gf.env_norm(gf.env_proj(s_in['env'])), gf.text_norm(gf.text_proj(s_in['text']))
        temp_out_all = gf.temporal_encoder(gf._add_positional_encoding(gf.input_proj_temporal(torch.cat([f_e_all, e_e_all, t_e_all, s_in['spk']], dim=-1))))

        curr_tgt_emb = torch.zeros(1, 1, gf.d_model, device=device)
        for t, utt in enumerate(scene_data['utterances']):
            u_idx, c_id = utt['utterance_index'], utt['char_id']
            base_id = f"{args.scene_id}@{c_id}_00_{args.scene_id}_{u_idx:03d}"
            char_persona = next(c['persona'] for c in scene_data['characters'] if c['char_id'] == c_id)
            v_p, ts = os.path.join(os.path.dirname(args.master_json), utt['video_path']), [utt.get('start_time', 0.0), utt.get('end_time', 0.0)]
            
            target_score, is_satisfied, total_attempts = 4.5, False, 0
            last_arc, last_aud, feedback, history = None, None, None, []

            while not is_satisfied:
                for attempt_at_level in range(3):
                    if total_attempts == 0:
                        existing_arc = utt.get('director_arc')
                        if existing_arc and existing_arc not in ["FAILED", ""]:
                            arc_text = existing_arc
                            dir_res = {"Director Arc": arc_text}
                        else:
                            dir_res = director.run_task("INITIAL", utt['text'], c_id, char_persona, u_idx, ts, 
                                                       if_face=utt.get('if_face', True), 
                                                       scene_desc=scene_data['scene_description'], 
                                                       video_path=v_p)
                            arc_text = dir_res.get("Director Arc", "Neutral.")
                            utt['director_arc'] = arc_text 
                            with open(dataset_path, 'w', encoding='utf-8') as f:
                                json.dump(movie_data, f, ensure_ascii=False, indent=2)
                    else:
                        dir_res = director.run_task("REVISION", utt['text'], c_id, char_persona, u_idx, ts, 
                                                   if_face=utt.get('if_face', True), 
                                                   scene_desc=scene_data['scene_description'], 
                                                   video_path=v_p, audio_path=last_aud, 
                                                   feedback=feedback, last_direct=last_arc)
                        arc_text = dir_res.get("Director Arc", "Neutral.")
                    
                    arc_t = extractor.get_1024d_vector(arc_text)
                    ref_t = rag_engine.retrieve_top_k(arc_t, top_k=full_cfg['train'].get('top_k', 2)).unsqueeze(0).unsqueeze(0)
                    a_e_t = gf.arc_norm(gf.arc_proj(arc_t))
                    r_fused_t = gf._fusion_ref_dynamic(ref_t, gf.context_to_query(torch.cat([f_e_all[:, t:t+1, :], t_e_all[:, t:t+1, :], a_e_t], dim=-1)))
                    feat_out_t = gf.feature_aggregator(gf.feature_encoder(gf._add_positional_encoding(torch.stack([f_e_all[:, t:t+1, :], e_e_all[:, t:t+1, :], t_e_all[:, t:t+1, :], r_fused_t, a_e_t], dim=2).view(1, 5, 512))).flatten(start_dim=1).unsqueeze(1))
                    
                    gate_t = gf.gate_sigmoid(gf.gate_w1(temp_out_all) + gf.gate_w2(feat_out_t))
                    memory_t = gate_t * temp_out_all + (1 - gate_t) * feat_out_t
                    out_dec = gf.decoder(gf._add_positional_encoding(gf.decoder_input_proj(torch.cat([curr_tgt_emb, s_in['spk'][:, :curr_tgt_emb.size(1), :]], dim=-1))), memory_t, tgt_mask=gf._generate_causal_mask(curr_tgt_emb.size(1)).to(device))
                    pred_1280 = gf.output_projection(out_dec[:, -1:, :])

                    wave_ref, _ = librosa.load(os.path.join(actor.wav_root, f"{base_id}.wav"), sr=actor.sr)
                    target_mel_len = actor.preprocess(wave_ref, actor.mel_transform, actor.mean, actor.std).shape[-1]
                    curr_aud_p = os.path.join(args.save_path, f"{base_id}_v{total_attempts}.wav")
                    atm = load_npy(os.path.join(actor.pro_feat_root, "scene", f"{args.scene_id}@{c_id}-scene-{base_id}.npy")).view(1, 1024)
                    lip, emo = None, None
                    if utt.get('if_face', True):
                        lip = torch.from_numpy(np.load(os.path.join(actor.pro_feat_root, "extrated_embedding_V2C_gray", f"{args.scene_id}@{c_id}-face-{base_id}.npy"))).float().to(device).unsqueeze(0)
                        emo = torch.from_numpy(np.load(os.path.join(actor.pro_feat_root, "VA_features", f"{args.scene_id}@{c_id}-feature-{base_id}.npy"))).float().to(device).unsqueeze(0)
                    
                    wav = actor._dub_utterance(utt['text'], char_timbre_map[c_id], pred_1280, lip, emo, atm, target_mel_len)
                    sf.write(curr_aud_p, wav, actor.sr)

                    rev = reviewer.run_consultant_evaluation(None, scene_data['scene_description'], char_persona, c_id, utt['text'], curr_aud_p, last_aud, v_p, total_attempts > 0)
                    feedback = f"SQC:{rev.get('sqc_report')} | SQA:{rev.get('sqa_report')} | SQI:{rev.get('sqi_suggestions')}"
                    score_res = director.run_task(task_type="SCORING",script_text=utt['text'],char_id=c_id,char_persona=char_persona,utt_idx=u_idx,timestamp_info=ts,scene_desc=scene_data['scene_description'],video_path=v_p,audio_path=curr_aud_p,feedback=feedback)
                    c_score = score_res.get("Comprehensive_Score", 0.0)
                    
                    current_step = {"base_id": base_id, "total_attempts": total_attempts, "target_threshold": target_score, "director_instruction": dir_res, "comprehensive_score": score_res, "expert_evaluation": rev, "audio_path": curr_aud_p}
                    with open(step_log_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(current_step, ensure_ascii=False) + '\n')

                    history.append(current_step)
                    last_arc, last_aud = arc_text, curr_aud_p
                    total_attempts += 1 

                    if c_score >= target_score:
                        is_satisfied = True
                        if target_score >= 4.5: rag_engine.update_database(arc_t.squeeze(), pred_1280.squeeze(), base_id, args.scene_id,c_id)
                        break 
                
                if not is_satisfied: 
                    target_score -= 0.5
                    if target_score < 1.0: break

            shutil.copy(last_aud, os.path.join(args.save_path, f"{base_id}_final.wav"))
            with open(final_log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps({"base_id": base_id, "satisfied": is_satisfied, "final_score": history[-1]["comprehensive_score"]}, ensure_ascii=False) + '\n')
            if t < len(scene_data['utterances']) - 1: curr_tgt_emb = torch.cat([curr_tgt_emb, gf.tgt_proj(pred_1280)], dim=1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, default="sk-lhiymlfpnqvvgpwgwtdfktnkyjyghoixawtwglkyfomfpluc")
    parser.add_argument("--gate_config", default="EmotionGateformer/Configs/Config.yml")
    parser.add_argument("--pro_config", default="ProDubber/output/stage2/config.yml")
    parser.add_argument("--gate_ckpt", default="EmotionGateformer/output/checkpoints/gateformer_ep10.pth")
    parser.add_argument("--pro_ckpt", default="ProDubber/output/stage2/ckpt/epoch_2nd_210.pth")
    parser.add_argument("--master_json", default="Dataset/data/dataset.json")
    parser.add_argument("--movie_id", default="Zootopia")
    parser.add_argument("--scene_id", default="Zootopia004")
    parser.add_argument("--save_path", default="results/Zootopia004")
    run_inference(parser.parse_args())