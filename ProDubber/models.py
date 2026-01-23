import os
import os.path as osp
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

from Utils.ASR.models import ASRCNN
from Utils.JDC.model import JDCNet

from Modules.diffusion.sampler import KDiffusion, LogNormalDistribution
from Modules.diffusion.modules import Transformer1d, StyleTransformer1d
from Modules.diffusion.diffusion import AudioDiffusionConditional
from Modules.discriminators import MultiPeriodDiscriminator, MultiResSpecDiscriminator, WavLMDiscriminator

from munch import Munch
import yaml

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model, device=device)
        position = torch.arange(0, max_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * -(math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        x = x + self.encoding[:, :x.size(1)].to(x.device)
        return x

class IndexMappingEncoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, out_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(out_dim * 2, out_dim),
            nn.LayerNorm(out_dim)
        )
    def forward(self, x):
        return self.fc(x)

class AffineLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(AffineLinear, self).__init__()
        self.affine = nn.Linear(in_dim, out_dim)
    def forward(self, input):
        return self.affine(input)

class SALN(nn.Module):
    def __init__(self, in_channel, style_dim):
        super(SALN, self).__init__()
        self.in_channel = in_channel
        self.norm = nn.LayerNorm(in_channel, elementwise_affine=False)
        self.style = AffineLinear(style_dim, in_channel * 2)
        self.style.affine.bias.data[:in_channel] = 1
        self.style.affine.bias.data[in_channel:] = 0
    def forward(self, input, style_code):
        style = self.style(style_code).unsqueeze(1)
        gamma, beta = style.chunk(2, dim=-1)
        return gamma * input + beta

class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)
        torch.nn.init.xavier_uniform_(self.linear_layer.weight, gain=torch.nn.init.calculate_gain(w_init_gain))
    def forward(self, x):
        return self.linear_layer(x)

class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))
    def forward(self, x):
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)
    
class TextEncoder(nn.Module):
    def __init__(self, channels, kernel_size, depth, n_symbols, actv=nn.LeakyReLU(0.2)):
        super().__init__()
        self.embedding = nn.Embedding(n_symbols, channels)
        padding = (kernel_size - 1) // 2
        self.cnn = nn.ModuleList([
            nn.Sequential(
                weight_norm(nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)),
                LayerNorm(channels), actv, nn.Dropout(0.2),
            ) for _ in range(depth)
        ])
        self.lstm = nn.LSTM(channels, channels//2, 1, batch_first=True, bidirectional=True)

    def forward(self, x, input_lengths, m):
        x = self.embedding(x).transpose(1, 2)
        m = m.to(input_lengths.device).unsqueeze(1)
        for c in self.cnn:
            x = c(x)
            x.masked_fill_(m, 0.0)
        x = x.transpose(1, 2)
        x = nn.utils.rnn.pack_padded_sequence(x, input_lengths.cpu().numpy(), batch_first=True, enforce_sorted=False)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = x.transpose(-1, -2)
        x_pad = torch.zeros([x.shape[0], x.shape[1], m.shape[-1]], device=x.device)
        x_pad[:, :, :x.shape[-1]] = x
        return x_pad.masked_fill(m, 0.0)

class AdaIN1d(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm1d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)
    def forward(self, x, s):
        h = self.fc(s).unsqueeze(-1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta

class AdainResBlk1d(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64, actv=nn.LeakyReLU(0.2), upsample=False, dropout_p=0.0):
        super().__init__()
        self.actv, self.upsample = actv, upsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.pool = weight_norm(nn.ConvTranspose1d(dim_in, dim_in, 3, 2, 1, 1)) if upsample else nn.Identity()
        
    def _build_weights(self, dim_in, dim_out, style_dim):
        self.conv1 = weight_norm(nn.Conv1d(dim_in, dim_out, 3, 1, 1))
        self.conv2 = weight_norm(nn.Conv1d(dim_out, dim_out, 3, 1, 1))
        self.norm1 = AdaIN1d(style_dim, dim_in)
        self.norm2 = AdaIN1d(style_dim, dim_out)
        if self.learned_sc: self.conv1x1 = weight_norm(nn.Conv1d(dim_in, dim_out, 1, 1, 0, bias=False))

    def forward(self, x, s):
        res = F.interpolate(x, scale_factor=2) if self.upsample else x
        if self.learned_sc: res = self.conv1x1(res)
        out = self.norm1(x, s)
        out = self.actv(out)
        out = self.pool(out)
        out = self.conv1(self.dropout(out))
        out = self.norm2(out, s)
        out = self.actv(out)
        out = self.conv2(self.dropout(out))
        return (out + res) / math.sqrt(2)

class AdaLayerNorm(nn.Module):
    def __init__(self, style_dim, channels, eps=1e-5):
        super().__init__()
        self.channels, self.eps = channels, eps
        self.fc = nn.Linear(style_dim, channels*2)
    def forward(self, x, s):
        h = self.fc(s).unsqueeze(1)
        gamma, beta = torch.chunk(h, chunks=2, dim=-1)
        x = F.layer_norm(x.transpose(1, 2), (self.channels,), eps=self.eps)
        x = (1 + gamma) * x + beta
        return x.transpose(1, 2)

class ProsodyEncoder(nn.Module):
    def __init__(self, sty_dim, d_model, nlayers, dropout=0.1):
        super().__init__()
        self.lstms = nn.ModuleList()
        for _ in range(nlayers):
            self.lstms.append(nn.LSTM(d_model + sty_dim, d_model // 2, 1, batch_first=True, bidirectional=True))
            self.lstms.append(AdaLayerNorm(sty_dim, d_model))
        self.dropout, self.sty_dim = dropout, sty_dim

    def forward(self, x, style, text_lengths, m):
        x = x.transpose(1, 2)
        s = style.unsqueeze(1).expand(-1, x.size(1), -1)
        for block in self.lstms:
            if isinstance(block, AdaLayerNorm):
                x = block(x.transpose(1, 2), style).transpose(1, 2)
            else:
                x_in = torch.cat([x, s], dim=-1)
                x_in = nn.utils.rnn.pack_padded_sequence(x_in, text_lengths.cpu().numpy(), batch_first=True, enforce_sorted=False)
                block.flatten_parameters()
                x, _ = block(x_in)
                x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x.transpose(1, 2)

class ProsodyPredictor(nn.Module):
    def __init__(self, style_dim, d_hid, nlayers, max_dur=50, dropout=0.1):
        super().__init__() 
        self.text_encoder = ProsodyEncoder(style_dim, d_hid, nlayers, dropout)
        self.shared = nn.LSTM(d_hid + style_dim, d_hid // 2, 1, batch_first=True, bidirectional=True)
        self.F0 = nn.ModuleList([AdainResBlk1d(d_hid, d_hid, style_dim), AdainResBlk1d(d_hid, d_hid//2, style_dim, upsample=True), AdainResBlk1d(d_hid//2, d_hid//2, style_dim)])
        self.N = nn.ModuleList([AdainResBlk1d(d_hid, d_hid, style_dim), AdainResBlk1d(d_hid, d_hid//2, style_dim, upsample=True), AdainResBlk1d(d_hid//2, d_hid//2, style_dim)])
        self.F0_proj, self.N_proj = nn.Conv1d(d_hid//2, 1, 1), nn.Conv1d(d_hid//2, 1, 1)

    def forward(self, texts, style, text_lengths, alignment, text_mask):
        d = self.text_encoder(texts, style, text_lengths, text_mask)
        return d @ alignment 
    
    def F0Ntrain(self, x, s):
        x = x.transpose(1, 2)
        s_expand = s.unsqueeze(1).expand(-1, x.size(1), -1)
        x_in = torch.cat([x, s_expand], dim=-1)
        
        x, _ = self.shared(x_in)
        F0 = N = x.transpose(1, 2)
        for f, n in zip(self.F0, self.N):
            F0, N = f(F0, s), n(N, s)
        return self.F0_proj(F0).squeeze(1), self.N_proj(N).squeeze(1)

class DurationPredictor(nn.Module):
    def __init__(self, style_dim, d_hid, nlayers, max_dur=50, dropout=0.1):
        super().__init__() 
        self.text_encoder = ProsodyEncoder(style_dim, d_hid, nlayers, dropout)
        self.lstm = nn.LSTM(d_hid + style_dim, d_hid // 2, 1, batch_first=True, bidirectional=True)
        self.duration_proj = LinearNorm(d_hid, max_dur)

    def forward(self, texts, style, text_lengths, text_mask):
        d = self.text_encoder(texts, style, text_lengths, text_mask).transpose(1, 2)
        s_expand = style.unsqueeze(1).expand(-1, d.size(1), -1)
        d_in = torch.cat([d, s_expand], dim=-1)
        
        x = nn.utils.rnn.pack_padded_sequence(d_in, text_lengths.cpu().numpy(), batch_first=True, enforce_sorted=False)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        return self.duration_proj(F.dropout(x, 0.5, training=self.training)).squeeze(-1)

class DurationEncoder(nn.Module):
    def __init__(self, d_model, nlayers, dropout=0.1):
        super().__init__()
        self.lstms = nn.ModuleList([nn.LSTM(d_model, d_model // 2, 1, batch_first=True, bidirectional=True) for _ in range(nlayers)])
        self.dropout = dropout
    def forward(self, x, text_lengths, m):
        x = x.transpose(1, 2)
        for block in self.lstms:
            x_p = nn.utils.rnn.pack_padded_sequence(x, text_lengths.cpu().numpy(), batch_first=True, enforce_sorted=False)
            block.flatten_parameters()
            x, _ = block(x_p)
            x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class DurationPredictor_Re2(nn.Module):
    def __init__(self, style_dim, d_hid, nlayers, max_dur=50, dropout=0.1):
        super().__init__() 
        self.text_encoder = DurationEncoder(d_hid, nlayers, dropout)
        self.lip_phoneme_attention = nn.MultiheadAttention(512, 8, dropout=0.1)
        self.lstm = nn.LSTM(d_hid, d_hid // 2, 1, batch_first=True, bidirectional=True)
        self.duration_proj = LinearNorm(d_hid, max_dur)
        self.pos_enc = PositionalEncoding(512)
    def forward(self, texts, lip_f, text_lengths, text_mask, visual_mask):
        d = self.text_encoder(texts, text_lengths, text_mask)
        lip_f = F.normalize(lip_f, p=2, dim=-1)
        motion = (lip_f[:, 1:] - lip_f[:, :-1]) * (~visual_mask[:, 1:]).unsqueeze(-1)
        motion = motion + self.pos_enc(motion)
        fusion, _ = self.lip_phoneme_attention(d.transpose(0, 1), motion.transpose(0, 1), motion.transpose(0, 1), key_padding_mask=visual_mask[:, 1:])
        d = d + fusion.transpose(0, 1)
        x = nn.utils.rnn.pack_padded_sequence(d, text_lengths.cpu().numpy(), batch_first=True, enforce_sorted=False)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        return self.duration_proj(F.dropout(x, 0.5, training=self.training)).squeeze(-1)

class IDEA(nn.Module):
    def __init__(self):
        super().__init__()
        self.atm_proj = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        self.bert2emb = nn.Linear(512, 256)
        self.emo_fc_v = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256))
        self.emo_fc_a = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256))
        self.val_attn = nn.MultiheadAttention(256, 8, dropout=0.1)
        self.aro_attn = nn.MultiheadAttention(256, 8, dropout=0.1)
        self.pos_enc = PositionalEncoding(256)
        self.ALN_v, self.ALN_a = SALN(256, 256), SALN(256, 256)

    def forward(self, x, src_mask, vis_mask, feat_256, atm):
        atm_256 = self.atm_proj(atm)
        m = self.bert2emb(x.transpose(1, 2))
        v = F.normalize(self.emo_fc_v(feat_256), p=2, dim=-1)
        chg = (v[:, 1:] - v[:, :-1]) * (~vis_mask[:, 1:]).unsqueeze(-1)
        ctx_v, _ = self.val_attn(m.transpose(0, 1), (chg + self.pos_enc(chg)).transpose(0, 1), chg.transpose(0, 1), key_padding_mask=vis_mask[:, 1:])
        ctx_v = self.ALN_v(ctx_v.transpose(0, 1), atm_256)
        a = F.normalize(self.emo_fc_a(feat_256), p=2, dim=-1)
        ctx_a, _ = self.aro_attn(m.transpose(0, 1), (a + self.pos_enc(a)).transpose(0, 1), a.transpose(0, 1), key_padding_mask=vis_mask)
        ctx_a = self.ALN_a(ctx_a.transpose(0, 1), atm_256)
        return torch.cat([ctx_a, ctx_v], dim=-1).transpose(1, 2)

def load_F0_models(path):
    f0 = JDCNet(num_class=1, seq_len=192)
    f0.load_state_dict(torch.load(path, map_location='cpu')['net'])
    return f0.eval()

def load_ASR_models(path, config):
    with open(config) as f: cfg = yaml.safe_load(f)['model_params']
    asr = ASRCNN(**cfg)
    asr.load_state_dict(torch.load(path, map_location='cpu', weights_only=False)['model'])
    return asr.eval()

def build_model(args, text_aligner, pitch_extractor, bert):
    total_style_dim = args.style_dim + args.audio_emotion_dim
    
    if args.decoder.type == "istftnet":
        from Modules.istftnet import Decoder
        decoder = Decoder(
            dim_in=args.hidden_dim, 
            style_dim=args.style_dim, 
            dim_out=args.n_mels,
            resblock_kernel_sizes=args.decoder.resblock_kernel_sizes,
            upsample_rates=args.decoder.upsample_rates,
            upsample_initial_channel=args.decoder.upsample_initial_channel,
            resblock_dilation_sizes=args.decoder.resblock_dilation_sizes,
            upsample_kernel_sizes=args.decoder.upsample_kernel_sizes, 
            gen_istft_n_fft=args.decoder.gen_istft_n_fft, 
            gen_istft_hop_size=args.decoder.gen_istft_hop_size
        ) 
    else:
        from Modules.hifigan import Decoder
        decoder = Decoder(
            dim_in=args.hidden_dim, 
            style_dim=args.style_dim, 
            dim_out=args.n_mels,
            resblock_kernel_sizes=args.decoder.resblock_kernel_sizes,
            upsample_rates=args.decoder.upsample_rates,
            upsample_initial_channel=args.decoder.upsample_initial_channel,
            resblock_dilation_sizes=args.decoder.resblock_dilation_sizes,
            upsample_kernel_sizes=args.decoder.upsample_kernel_sizes
        ) 
        
    text_encoder = TextEncoder(args.hidden_dim, 5, args.n_layer, args.n_token)
    predictor = ProsodyPredictor(args.audio_emotion_dim, args.hidden_dim, args.n_layer, args.max_dur, args.dropout)
    duration_predictor_visual = DurationPredictor_Re2(args.audio_emotion_dim, args.hidden_dim, args.n_layer, args.max_dur, args.dropout)
    duration_predictor_audio = DurationPredictor(args.audio_emotion_dim, args.hidden_dim, args.n_layer, args.max_dur, args.dropout)
    
    style_encoder = IndexMappingEncoder(192, args.style_dim)
    predictor_encoder = IndexMappingEncoder(1280, args.audio_emotion_dim)
        
    transformer = Transformer1d(channels=total_style_dim, context_embedding_features=args.hidden_dim, **args.diffusion.transformer)
    
    diffusion = AudioDiffusionConditional(in_channels=1, embedding_max_length=bert.config.max_position_embeddings,
        embedding_features=args.hidden_dim, embedding_mask_proba=args.diffusion.embedding_mask_proba,
        channels=total_style_dim, context_features=total_style_dim)
    diffusion.diffusion = KDiffusion(net=transformer, sigma_distribution=LogNormalDistribution(args.diffusion.dist.mean, args.diffusion.dist.std),
        sigma_data=args.diffusion.dist.sigma_data, dynamic_threshold=0.0)
    diffusion.unet = transformer
    
    return Munch(bert=bert, bert_encoder=nn.Linear(bert.config.hidden_size, args.hidden_dim), predictor=predictor,
        duration_predictor_visual=duration_predictor_visual, duration_predictor_audio=duration_predictor_audio,
        decoder=decoder, text_encoder=text_encoder, predictor_encoder=predictor_encoder, style_encoder=style_encoder,
        diffusion=diffusion, text_aligner=text_aligner, pitch_extractor=pitch_extractor, mpd=MultiPeriodDiscriminator(),
        msd=MultiResSpecDiscriminator(), wd=WavLMDiscriminator(args.slm.hidden, args.slm.nlayers, args.slm.initial_channel), prosody_fusion=IDEA()) 

def load_checkpoint(model, optimizer, path, load_only_params=True, ignore_modules=[]):
    state = torch.load(path, map_location='cpu')
    params = state['net']
    for key in model:
        if key in params and key not in ignore_modules:
            print('%s loaded' % key)
            try:
                model[key].load_state_dict(params[key], strict=True)
            except:
                new_sd = torch.nn.modules.container.OrderedDict()
                for (k_m, v_m), (k_c, v_c) in zip(model[key].state_dict().items(), params[key].items()):
                    new_sd[k_m] = v_c
                model[key].load_state_dict(new_sd, strict=True)
    if not load_only_params:
        if optimizer and "optimizer" in state: optimizer.load_state_dict(state["optimizer"])
        return model, optimizer, state.get("epoch", 0), state.get("iters", 0)
    return model, optimizer, 0, 0