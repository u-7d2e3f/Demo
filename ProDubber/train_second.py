import os
import os.path as osp
import random
import yaml
import time
import copy
from munch import Munch
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa
import click
import shutil
import traceback
import warnings
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from meldataset import build_dataloader
from Utils.ASR.models import ASRCNN
from Utils.JDC.model import JDCNet
from Utils.PLBERT.util import load_plbert

from models import *
from losses import *
from utils import *

from Modules.slmadv import SLMAdversarialLoss
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule
from optimizers import build_optimizer

warnings.simplefilter('ignore')

class MyDataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
        
import logging
from logging import StreamHandler
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = StreamHandler()
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)

@click.command()
@click.option('-p', '--config_path', default='Configs/config.yml', type=str)
def main(config_path):
    config = yaml.safe_load(open(config_path))
    exp_name = config.get('exp_name', 'debug')
    log_dir = config['log_dir'].format(exp_name)
    ckpt_dir = os.path.join(log_dir, 'ckpt')
    script_dir = os.path.join(log_dir, 'script')
    
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(script_dir, exist_ok=True)
    
    if not osp.exists(log_dir): os.makedirs(log_dir, exist_ok=True)
    shutil.copy(config_path, osp.join(log_dir, osp.basename(config_path)))
    writer = SummaryWriter(log_dir + "/tensorboard")

    file_handler = logging.FileHandler(osp.join(log_dir, 'train.log'))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(levelname)s:%(asctime)s: %(message)s'))
    logger.addHandler(file_handler)

    file_to_save = ['inference_v2c.py', 'losses.py', 'meldataset.py','models.py',
                    'optimizers.py', 'train_first.py',
                    'train_second.py', 'utils.py',]
    folder_to_copy = ['Modules', 'Configs']
    for x in file_to_save: shutil.copy2(x, script_dir)
    for x in folder_to_copy:
        if os.path.exists('{}/{}'.format(script_dir, x)): shutil.rmtree('{}/{}'.format(script_dir, x))
        shutil.copytree(x, '{}/{}'.format(script_dir, x))

    batch_size = config.get('batch_size', 10)
    epochs = config.get('epochs_2nd', 200)
    log_interval = config.get('log_interval', 10)
    saving_epoch = config.get('save_freq', 2)
    data_params = config.get('data_params', None)
    sr = config['preprocess_params'].get('sr', 24000)
    train_path = data_params['train_data']
    val_path = data_params['val_data']
    root_path = data_params['root_path']
    feature_root_path = data_params['feature_root_path']
    min_length = data_params['min_length']
    OOD_data = data_params.get('OOD_data', None)
    max_len = config.get('max_len', 200)
    
    loss_params = Munch(config['loss_params'])
    diff_epoch = loss_params.diff_epoch
    optimizer_params = Munch(config['optimizer_params'])
    
    train_list, val_list = get_data_path_list(train_path, val_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    use_random_ref = config['random_ref']
    speaker_finetune = config.get('speaker_finetune', False)
    
    train_dataloader = build_dataloader(train_list, root_path, feature_root_path, 
                                        batch_size=batch_size, num_workers=2, use_random_ref=use_random_ref)
    
    val_dataloader = build_dataloader(val_list,
                                      root_path,
                                      feature_root_path,
                                      OOD_data=OOD_data,
                                      min_length=min_length,
                                      batch_size=batch_size,
                                      validation=True,
                                      num_workers=0,
                                      device=device,
                                      dataset_config={})
    
    text_aligner = load_ASR_models(config.get('ASR_path'), config.get('ASR_config'))
    pitch_extractor = load_F0_models(config.get('F0_path'))
    plbert = load_plbert(config.get('PLBERT_dir'))
    
    model_params = recursive_munch(config['model_params'])
    model = build_model(model_params, text_aligner, pitch_extractor, plbert)
    _ = [model[key].to(device) for key in model]
    for key in model:
        if key not in ["mpd", "msd", "wd"]: model[key] = MyDataParallel(model[key])
            
    start_epoch = 0
    iters = 0
    load_pretrained = config.get('pretrained_model', '') != '' and config.get('second_stage_load_pretrained', False)
    
    if not load_pretrained:
        if config.get('first_stage_path', '') != '':
            first_stage_path = config.get('first_stage_path', 'first_stage.pth')
            model, _, start_epoch, iters = load_checkpoint(model, None, first_stage_path,
                load_only_params=True, ignore_modules=['bert', 'bert_encoder', 'predictor', 'predictor_encoder', 'msd', 'mpd', 'wd', 'diffusion', 
                                'prosody_fusion', 'duration_predictor', 'duration_predictor_visual', 'duration_predictor_audio']) 
            diff_epoch += start_epoch
            epochs += start_epoch
        else: raise ValueError('Path to the first stage model must be specified.')

    gl = MyDataParallel(GeneratorLoss(model.mpd, model.msd).to(device))
    dl = MyDataParallel(DiscriminatorLoss(model.mpd, model.msd).to(device))
    wl = MyDataParallel(WavLMLoss(os.path.join(os.path.dirname(__file__), model_params.slm.model), model.wd, sr, model_params.slm.sr).to(device))
    
    sampler = DiffusionSampler(model.diffusion.diffusion, sampler=ADPM2Sampler(), sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0), clamp=False)
    
    scheduler_params = {"max_lr": optimizer_params.lr, "pct_start": float(0), "epochs": epochs, "steps_per_epoch": len(train_dataloader)}
    scheduler_params_dict = {key: scheduler_params.copy() for key in model}
    scheduler_params_dict['bert']['max_lr'] = optimizer_params.bert_lr * 2
    scheduler_params_dict['decoder']['max_lr'] = optimizer_params.ft_lr * 2
    scheduler_params_dict['style_encoder']['max_lr'] = optimizer_params.ft_lr * 2
    
    optimizer = build_optimizer({key: model[key].parameters() for key in model}, scheduler_params_dict=scheduler_params_dict, lr=optimizer_params.lr)
    
    for g in optimizer.optimizers['bert'].param_groups:
        g['betas'] = (0.9, 0.99); g['lr'] = optimizer_params.bert_lr; g['initial_lr'] = optimizer_params.bert_lr; g['min_lr'] = 0; g['weight_decay'] = 0.01
    for module in ["decoder", "style_encoder"]:
        for g in optimizer.optimizers[module].param_groups:
            g['betas'] = (0.0, 0.99); g['lr'] = optimizer_params.ft_lr; g['initial_lr'] = optimizer_params.ft_lr; g['min_lr'] = 0; g['weight_decay'] = 1e-4
        
    if load_pretrained: model, optimizer, start_epoch, iters = load_checkpoint(model, optimizer, config['pretrained_model'], load_only_params=config.get('load_only_params', True))
        
    n_down = model.text_aligner.n_down
    stft_loss = MultiResolutionSTFTLoss().to(device)
    running_std = []

    def compute_single_dur_loss(preds, gt, lengths):
        l_ce = 0
        l_dur = 0
        for _s2s_pred, _text_input, _text_length in zip(preds, gt, lengths):
            _s2s_pred = _s2s_pred[:_text_length, :]
            _text_input = _text_input[:_text_length].long()
            _s2s_trg = torch.zeros_like(_s2s_pred)
            for p in range(_s2s_trg.shape[0]):
                _s2s_trg[p, :_text_input[p]] = 1
            _dur_pred = torch.sigmoid(_s2s_pred).sum(axis=1)

            l_dur += F.l1_loss(_dur_pred[1:_text_length-1], 
                                   _text_input[1:_text_length-1])
            l_ce += F.binary_cross_entropy_with_logits(_s2s_pred.flatten(), _s2s_trg.flatten())
        return l_dur, l_ce

    start_ds = False
    
    for epoch in range(start_epoch, epochs):
        _ = [model[key].eval() for key in model]
        model.predictor.train()
        model.duration_predictor_visual.train(); model.duration_predictor_audio.train()
        model.bert_encoder.train(); model.bert.train(); model.msd.train(); model.mpd.train()
        model.prosody_fusion.train(); model.predictor_encoder.train()
        if speaker_finetune: model.style_encoder.train()

        if epoch >= diff_epoch:
            start_ds = True
            model.diffusion.train(); model.style_encoder.eval()

        running_loss = 0
        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch+1}/{epochs}")
        
        for i, batch in pbar:
            (texts, input_lengths, t_vecs, e_vecs, mels, 
             mel_input_length, ref_t_vecs, lip_features, 
             emotion_features, atm_features, visual_lengths) = [b.to(device) if torch.is_tensor(b) else b for b in batch[1:]]

            speaker_style = model.style_encoder(t_vecs)             
            prosody_mel_feature = model.predictor_encoder(e_vecs)   
            style_trg = torch.cat([speaker_style, prosody_mel_feature], dim=-1).detach()
            ref_speaker_style = model.style_encoder(ref_t_vecs)

            with torch.no_grad():
                mask = length_to_mask(mel_input_length // (2 ** n_down)).to(device)
                text_mask = length_to_mask(input_lengths).to(device)
                visual_mask = length_to_mask(visual_lengths).to(device)
                try:
                    _, _, s2s_attn = model.text_aligner(mels, mask, texts)
                    s2s_attn = s2s_attn.transpose(-1, -2)[..., 1:].transpose(-1, -2)
                    mask_ST = mask_from_lens(s2s_attn, input_lengths, mel_input_length // (2 ** n_down))
                    s2s_attn_mono = maximum_path(s2s_attn, mask_ST)
                    phoneme = model.text_encoder(texts, input_lengths, text_mask)
                    acoustic_phoneme = (phoneme @ s2s_attn_mono)
                    duration_gt = s2s_attn_mono.sum(axis=-1).detach()
                except: continue

            bert_output = model.bert(texts, attention_mask=(~text_mask).int())
            prosody_phoneme_feature = model.bert_encoder(bert_output).transpose(-1, -2)

            drop_visual = random.random() < 0.2
            if drop_visual:
                prosody_phoneme_feature_emotion = prosody_phoneme_feature 
                duration_pred = model.duration_predictor_audio(prosody_phoneme_feature, prosody_mel_feature, input_lengths, text_mask)
            else:
                prosody_phoneme_feature_emotion = model.prosody_fusion(prosody_phoneme_feature, text_mask, visual_mask, emotion_features, atm_features) + prosody_phoneme_feature
                duration_pred = model.duration_predictor_visual(prosody_phoneme_feature, lip_features, input_lengths, text_mask, visual_mask)

            if start_ds:
                num_steps = np.random.randint(3, 5)
                if model_params.diffusion.dist.estimate_sigma_data:
                    model.diffusion.module.diffusion.sigma_data = style_trg.std(axis=-1).mean().item()
                    running_std.append(model.diffusion.module.diffusion.sigma_data)

                loss_diff = model.diffusion.module.diffusion(style_trg.unsqueeze(1), embedding=prosody_phoneme_feature_emotion.transpose(-1, -2)).mean()
                s_preds = sampler(noise=torch.randn_like(style_trg).unsqueeze(1).to(device), embedding=prosody_phoneme_feature_emotion.transpose(-1, -2), num_steps=num_steps).squeeze(1)
                loss_sty = F.l1_loss(s_preds, style_trg.detach())
            else: loss_diff = loss_sty = 0

            prosody_feature = model.predictor(prosody_phoneme_feature_emotion, prosody_mel_feature, input_lengths, s2s_attn_mono, text_mask)
            
            mel_len = min(int(mel_input_length.min().item() / 2 - 1), max_len // 2)
            en, p_en, gt = [], [], []
            for bib in range(len(mel_input_length)):
                r_st = np.random.randint(0, int(mel_input_length[bib]/2) - mel_len)
                en.append(acoustic_phoneme[bib, :, r_st:r_st+mel_len])
                p_en.append(prosody_feature[bib, :, r_st:r_st+mel_len])
                gt.append(mels[bib, :, (r_st*2):((r_st+mel_len)*2)])
            
            en, p_en, gt = torch.stack(en), torch.stack(p_en), torch.stack(gt).detach()
            if gt.size(-1) < 80: continue

            with torch.no_grad():
                F0_real, _, _ = model.pitch_extractor(gt.unsqueeze(1))
                N_real = log_norm(gt.unsqueeze(1)).squeeze(1)
                wav = model.decoder(en, F0_real, N_real, speaker_style) 

            F0_fake, N_fake = model.predictor.F0Ntrain(p_en, prosody_mel_feature)
            y_rec = model.decoder(en, F0_fake, N_fake, ref_speaker_style)
            
            if start_ds:
                optimizer.zero_grad()
                d_loss = dl(wav.detach(), y_rec.detach()).mean()
                d_loss.backward()
                optimizer.step('msd'); optimizer.step('mpd')
            else: d_loss = 0

            optimizer.zero_grad()
            loss_mel = stft_loss(y_rec, wav)
            loss_gen_all = gl(wav, y_rec).mean() if start_ds else 0
            loss_lm = wl(wav.detach().squeeze(), y_rec.squeeze()).mean()
            loss_F0_rec = (F.smooth_l1_loss(F0_real, F0_fake)) / 10
            loss_norm_rec = F.smooth_l1_loss(N_real, N_fake)
            
            l_dur_val, l_ce_val = compute_single_dur_loss(duration_pred, duration_gt, input_lengths)
            loss_dur, loss_ce = l_dur_val / texts.size(0), l_ce_val / texts.size(0)

            g_loss = (loss_params.lambda_mel * loss_mel + loss_params.lambda_F0 * loss_F0_rec + 
                      loss_params.lambda_ce * loss_ce + loss_params.lambda_norm * loss_norm_rec + 
                      loss_params.lambda_dur * loss_dur + loss_params.lambda_gen * loss_gen_all + 
                      loss_params.lambda_slm * loss_lm + loss_params.lambda_sty * loss_sty + 
                      loss_params.lambda_diff * loss_diff)
            
            if torch.isnan(g_loss):
                logger.error("NaN Loss! Triggering debugger...")
                from IPython.core.debugger import set_trace; set_trace()
            
            g_loss.backward()

            for key in ['bert_encoder', 'bert', 'predictor', 'predictor_encoder']: optimizer.step(key)

            if drop_visual: optimizer.step('duration_predictor_audio')
            else: optimizer.step('prosody_fusion'); optimizer.step('duration_predictor_visual')
            
            if start_ds: optimizer.step('diffusion')
            if speaker_finetune and not start_ds: optimizer.step('style_encoder')
            
            iters += 1; running_loss += loss_mel.item()
            
            if (i+1)%log_interval == 0:
                logger.info ('Epoch [%d/%d], Step [%d], Loss: %.5f, Disc Loss: %.5f, Dur Loss: %.5f, CE Loss: %.5f, Norm Loss: %.5f, F0 Loss: %.5f, LM Loss: %.5f, Gen Loss: %.5f, Sty Loss: %.5f, Diff Loss: %.5f'
                    %(epoch+1, epochs, i+1, running_loss / log_interval, d_loss, loss_dur, loss_ce, loss_norm_rec, loss_F0_rec, loss_lm, loss_gen_all, loss_sty, loss_diff))
                
                writer.add_scalar('train/mel_loss', running_loss / log_interval, iters)
                writer.add_scalar('train/gen_loss', loss_gen_all, iters)
                writer.add_scalar('train/d_loss', d_loss, iters)
                writer.add_scalar('train/ce_loss', loss_ce, iters)
                writer.add_scalar('train/dur_loss', loss_dur, iters)
                writer.add_scalar('train/slm_loss', loss_lm, iters)
                writer.add_scalar('train/norm_loss', loss_norm_rec, iters)
                writer.add_scalar('train/F0_loss', loss_F0_rec, iters)
                writer.add_scalar('train/sty_loss', loss_sty, iters)
                writer.add_scalar('train/diff_loss', loss_diff, iters)
                
                running_loss = 0

        if not exp_name == 'debug' and epoch % saving_epoch == 0:
            print(f'Saving Checkpoint for Epoch {epoch}...')
            state = {
                    'net': {k: model[k].state_dict() for k in model},
                    'optimizer': optimizer.state_dict(),
                    'iters': iters,
                    'val_loss': None,
                    'epoch': epoch,
                }
            torch.save(state, osp.join(ckpt_dir, f'epoch_2nd_{epoch}.pth'))

            if model_params.diffusion.dist.estimate_sigma_data and len(running_std) > 0:
                config['model_params']['diffusion']['dist']['sigma_data'] = float(np.mean(running_std))
                with open(osp.join(log_dir, osp.basename(config_path)), 'w') as outfile:
                    yaml.dump(config, outfile, default_flow_style=True)

if __name__=="__main__":
    main()