import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class EmotionGateformer(nn.Module):
    def __init__(self, d_speaker, d_face, d_env, d_text, d_ref, d_arc, 
                 d_model=512, d_out=1280, 
                 nhead=8, 
                 num_layers_temp=6,      
                 num_layers_feat=6,      
                 num_layers_dec=6,       
                 dim_feedforward=2048, dropout=0.1, max_seq_len=2000):
        super(EmotionGateformer, self).__init__()
        
        self.d_model = d_model
        self.d_out = d_out
        self.nhead = nhead
        self.d_speaker = d_speaker
        self.max_seq_len = max_seq_len
        
        self.face_proj = nn.Linear(d_face, d_model)
        self.env_proj = nn.Linear(d_env, d_model)
        self.text_proj = nn.Linear(d_text, d_model)
        self.arc_proj = nn.Linear(d_arc, d_model)
        
        self.face_norm = nn.LayerNorm(d_model)
        self.env_norm = nn.LayerNorm(d_model)
        self.text_norm = nn.LayerNorm(d_model)
        self.arc_norm = nn.LayerNorm(d_model)

        self.context_to_query = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        
        self.ref_proj_individual = nn.Linear(d_ref, d_model) 
        self.ref_attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.ref_norm = nn.LayerNorm(d_model)
        self.ref_dropout = nn.Dropout(dropout)
        
        self.input_proj_temporal = nn.Linear(3 * d_model + d_speaker, d_model) 
        
        self.feature_aggregator = nn.Sequential(
            nn.Linear(5 * d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        
        enc_layer_temp = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.temporal_encoder = nn.TransformerEncoder(enc_layer_temp, num_layers_temp)
        
        enc_layer_feat = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.feature_encoder = nn.TransformerEncoder(enc_layer_feat, num_layers_feat)
        
        self.decoder_input_proj = nn.Linear(d_model + d_speaker, d_model)
        self.tgt_proj = nn.Linear(d_out, d_model) 
        
        dec_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers_dec)
        
        self.output_projection = nn.Linear(d_model, d_out) 
        
        self.gate_w1 = nn.Linear(d_model, d_model)
        self.gate_w2 = nn.Linear(d_model, d_model)
        self.gate_sigmoid = nn.Sigmoid()
        
        self.bias_same = nn.Parameter(torch.tensor(0.0)) 
        self.bias_diff = nn.Parameter(torch.tensor(0.0)) 
        self.similarity_threshold = 0.85 
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)

    def _bool_to_float_mask(self, bool_mask):
        if bool_mask is None: return None
        return torch.zeros_like(bool_mask, dtype=torch.float).masked_fill(bool_mask, float('-inf'))

    def _generate_causal_mask(self, seq_len):
        return torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)

    def _generate_similarity_bias_mask(self, speaker_vec):
        spk_norm = F.normalize(speaker_vec, p=2, dim=-1)
        sim_matrix = torch.bmm(spk_norm, spk_norm.transpose(1, 2))
        bias = torch.zeros_like(sim_matrix).masked_fill(sim_matrix > self.similarity_threshold, self.bias_same)
        bias = bias.masked_fill(sim_matrix <= self.similarity_threshold, self.bias_diff)
        return bias

    def _add_positional_encoding(self, tensor):
        pos = torch.arange(tensor.size(1), device=tensor.device).unsqueeze(0)
        return tensor + self.pos_emb(pos)

    def _fusion_ref_dynamic(self, ref, dynamic_query):
        ref_emb = self.ref_proj_individual(ref) 
        B, S, N, D = ref_emb.shape
        q = dynamic_query.view(B * S, 1, D)
        x = ref_emb.view(B * S, N, D)
        attn_out, _ = self.ref_attn(q, x, x)
        out = self.ref_norm(attn_out + q)
        return self.ref_dropout(out).view(B, S, D)

    def forward(self, face, env, text, ref, arc, speaker_vec, tgt, 
                src_key_padding_mask=None, tgt_key_padding_mask=None,
                face_mask=None): 
        B, S, _ = face.shape
        device = face.device 
        src_mask = self._bool_to_float_mask(src_key_padding_mask)
        tgt_mask_pad = self._bool_to_float_mask(tgt_key_padding_mask)
        
        f_e = self.face_norm(self.face_proj(face))
        
        if face_mask is not None:
            f_e = f_e * face_mask.unsqueeze(-1)
            
        e_e = self.env_norm(self.env_proj(env))
        t_e = self.text_norm(self.text_proj(text))
        a_e = self.arc_norm(self.arc_proj(arc))
        s_e = speaker_vec
        
        dyn_q = self.context_to_query(torch.cat([f_e, t_e, a_e], dim=-1))
        r_fused = self._fusion_ref_dynamic(ref, dyn_q)
        
        temp_combined = torch.cat([f_e, e_e, t_e, s_e], dim=-1)
        temp_in = self._add_positional_encoding(self.input_proj_temporal(temp_combined))
        
        spk_bias = self._generate_similarity_bias_mask(speaker_vec).repeat_interleave(self.nhead, dim=0)
        temp_out = self.temporal_encoder(temp_in, mask=spk_bias, src_key_padding_mask=src_mask)
        
        feat_stack = torch.stack([f_e, e_e, t_e, r_fused, a_e], dim=2)
        feat_in = self._add_positional_encoding(feat_stack.view(B*S, 5, self.d_model))
        feat_enc = self.feature_encoder(feat_in).view(B, S, 5, self.d_model)
        feat_out = self.feature_aggregator(feat_enc.flatten(start_dim=2))
        
        gate = self.gate_sigmoid(self.gate_w1(temp_out) + self.gate_w2(feat_out))
        memory = gate * temp_out + (1 - gate) * feat_out
        
        tgt_emb = self.tgt_proj(tgt) 
        sos = torch.zeros(B, 1, self.d_model, device=device)
        tgt_shifted = torch.cat([sos, tgt_emb[:, :-1, :]], dim=1)
        
        dec_in_combined = torch.cat([tgt_shifted, s_e], dim=-1)
        dec_in = self._add_positional_encoding(self.decoder_input_proj(dec_in_combined))
        
        out = self.decoder(dec_in, memory, tgt_mask=self._generate_causal_mask(S).to(device), 
                           tgt_key_padding_mask=tgt_mask_pad, memory_key_padding_mask=src_mask)
        
        return self.output_projection(out)

    def inference(self, face, env, text, ref, arc, speaker_vec, src_key_padding_mask=None, face_mask=None):
     
        B, S, _ = face.shape
        device = face.device 
        src_mask = self._bool_to_float_mask(src_key_padding_mask)
        
        with torch.no_grad():
            f_e_all = self.face_norm(self.face_proj(face))
    
            if face_mask is not None:
                f_e_all = f_e_all * face_mask.unsqueeze(-1)
                
            e_e_all = self.env_norm(self.env_proj(env))
            t_e_all = self.text_norm(self.text_proj(text))
            s_e = speaker_vec
            
            temp_in_all = self._add_positional_encoding(
                self.input_proj_temporal(torch.cat([f_e_all, e_e_all, t_e_all, s_e], dim=-1))
            )
            spk_bias = self._generate_similarity_bias_mask(speaker_vec).repeat_interleave(self.nhead, dim=0)
            temp_out_all = self.temporal_encoder(temp_in_all, mask=spk_bias, src_key_padding_mask=src_mask)

            curr_tgt_emb = torch.zeros(B, 1, self.d_model, device=device)
            res = []
            
            for t in range(S):
                f_e_t = f_e_all[:, t:t+1, :]
                e_e_t = e_e_all[:, t:t+1, :]
                t_e_t = t_e_all[:, t:t+1, :]
                ref_t = ref[:, t:t+1, :, :] 
                arc_t = arc[:, t:t+1, :]    
                a_e_t = self.arc_norm(self.arc_proj(arc_t))
                
                dyn_q_t = self.context_to_query(torch.cat([f_e_t, t_e_t, a_e_t], dim=-1))
                r_fused_t = self._fusion_ref_dynamic(ref_t, dyn_q_t)

                feat_stack_t = torch.stack([f_e_t, e_e_t, t_e_t, r_fused_t, a_e_t], dim=2)
                feat_in_t = self._add_positional_encoding(feat_stack_t.view(B, 5, self.d_model))
                feat_enc_t = self.feature_encoder(feat_in_t)
                
                feat_out_t = self.feature_aggregator(feat_enc_t.flatten(start_dim=1).unsqueeze(1))

                gate_t = self.gate_sigmoid(self.gate_w1(temp_out_all) + self.gate_w2(feat_out_t))
                memory_t = gate_t * temp_out_all + (1 - gate_t) * feat_out_t

                dec_in_step = self._add_positional_encoding(
                    self.decoder_input_proj(torch.cat([curr_tgt_emb, s_e[:, :curr_tgt_emb.size(1), :]], dim=-1))
                )
                
                out = self.decoder(dec_in_step, memory_t, 
                                   tgt_mask=self._generate_causal_mask(curr_tgt_emb.size(1)).to(device), 
                                   memory_key_padding_mask=src_mask)
                
                pred_1280 = self.output_projection(out[:, -1:, :])
                res.append(pred_1280)
                
                if t < S - 1:
                    curr_tgt_emb = torch.cat([curr_tgt_emb, self.tgt_proj(pred_1280)], dim=1)
            
            return torch.cat(res, dim=1)