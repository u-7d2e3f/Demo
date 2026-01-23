import os
import os.path as osp
import re
import sys
import yaml
import shutil
import numpy as np
import torch
import click
import warnings
import random
from munch import Munch
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa
import time
import logging
from tqdm import tqdm 

from models import *
from meldataset import build_dataloader_Stage1
from utils import *
from losses import *
from optimizers import build_optimizer

from accelerate import Accelerator
from accelerate.utils import LoggerType
from accelerate import DistributedDataParallelKwargs
from torch.utils.tensorboard import SummaryWriter
from accelerate.logging import get_logger

warnings.simplefilter('ignore')
logger = get_logger(__name__, log_level="DEBUG")

@click.command()
@click.option('-p', '--config_path', default='Configs/config.yml', type=str)
def main(config_path):
    config = yaml.safe_load(open(config_path))
    exp_name = config.get('exp_name', 'debug')

    log_dir = config['log_dir'].format(exp_name)
    ckpt_dir = os.path.join(log_dir, 'ckpt')
    os.makedirs(ckpt_dir, exist_ok=True)
    script_dir = os.path.join(log_dir, 'script')
    os.makedirs(script_dir, exist_ok=True)
    if not osp.exists(log_dir): os.makedirs(log_dir, exist_ok=True)
    shutil.copy(config_path, osp.join(log_dir, osp.basename(config_path)))
    
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(project_dir=log_dir, split_batches=True, kwargs_handlers=[ddp_kwargs])    
    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir + "/tensorboard")

    file_handler = logging.FileHandler(osp.join(log_dir, 'train.log'))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(levelname)s:%(asctime)s: %(message)s'))
    logger.logger.addHandler(file_handler)
    
    file_to_save = ['losses.py', 'meldataset.py','models.py', 'train_first.py', 'utils.py', 'optimizers.py']
    for x in file_to_save: shutil.copy2(x, script_dir)

    batch_size = config.get('batch_size', 10)
    device = accelerator.device
    epochs = config.get('epochs_1st', 200)
    log_interval = config.get('log_interval', 10)
    saving_epoch = config.get('save_freq', 2)
    
    data_params = config.get('data_params', None)
    sr = config['preprocess_params'].get('sr', 24000)
    train_path = data_params['train_data']
    root_path, feature_root_path = data_params['root_path'], data_params['feature_root_path']
    max_len = config.get('max_len', 200)
    
    train_list = get_data_path_list(train_path, None)[0]
    use_random_ref = config['random_ref']
    train_dataloader = build_dataloader_Stage1(train_list, root_path, feature_root_path, batch_size=batch_size, num_workers=2, use_random_ref=use_random_ref)
    
    with accelerator.main_process_first():
        text_aligner = load_ASR_models(config.get('ASR_path'), config.get('ASR_config'))
        pitch_extractor = load_F0_models(config.get('F0_path'))
        from Utils.PLBERT.util import load_plbert
        plbert = load_plbert(config.get('PLBERT_dir'))

    model_params = recursive_munch(config['model_params'])
    multispeaker = model_params.multispeaker
    model = build_model(model_params, text_aligner, pitch_extractor, plbert)

    try: n_down = model.text_aligner.module.n_down
    except: n_down = model.text_aligner.n_down

    scheduler_params = {"max_lr": float(config['optimizer_params'].get('lr', 1e-4)), "pct_start": 0.0, "epochs": epochs, "steps_per_epoch": len(train_dataloader)}
 
    model_params_for_opt = {}
    ignore_keys = ['duration_predictor_visual', 'duration_predictor_audio', 'duration_predictor']
    for key in model:
        if key not in ignore_keys:
           model_params_for_opt[key] = model[key].parameters()
    
    optimizer = build_optimizer(model_params_for_opt, scheduler_params_dict={k: scheduler_params.copy() for k in model}, lr=float(config['optimizer_params'].get('lr', 1e-4)))
    
    with accelerator.main_process_first():
        if config.get('pretrained_model', '') != '':
            print('Load Pretrain Model checkpoint: {}'.format(config['pretrained_model']))
            model, optimizer, start_epoch, iters = load_checkpoint(model, optimizer, config['pretrained_model'],
                                        load_only_params=config.get('load_only_params', True))
        else:
            start_epoch = 0
            iters = 0

    for k in model: model[k] = accelerator.prepare(model[k])
    train_dataloader = accelerator.prepare(train_dataloader)
    for k in optimizer.optimizers:
        optimizer.optimizers[k] = accelerator.prepare(optimizer.optimizers[k]); optimizer.schedulers[k] = accelerator.prepare(optimizer.schedulers[k])

    stft_loss = MultiResolutionSTFTLoss().to(device)
    gl, dl = GeneratorLoss(model.mpd, model.msd).to(device), DiscriminatorLoss(model.mpd, model.msd).to(device)
    wl = WavLMLoss(model_params.slm.model, model.wd, sr, model_params.slm.sr).to(device)
    
    loss_params = Munch(config['loss_params'])
    TMA_epoch = loss_params.TMA_epoch
    

    for epoch in range(start_epoch, epochs):
        running_loss = 0
        _ = [model[key].train() for key in model]

        batch_pbar = tqdm(enumerate(train_dataloader), desc=f"Epoch {epoch+1}/{epochs}", total=len(train_dataloader), disable=not accelerator.is_main_process)
        
        for i, batch in batch_pbar:
            waves = batch[0]
            texts, input_lengths, t_vecs, _, mels, mel_input_length, ref_t_vecs = [b.to(device) if torch.is_tensor(b) else b for b in batch[1:]]
            
            with torch.no_grad():
                mask = length_to_mask(mel_input_length // (2 ** n_down)).to(device)
                text_mask = length_to_mask(input_lengths).to(device)

            ppgs, s2s_pred, s2s_attn = model.text_aligner(mels, mask, texts)
            s2s_attn = s2s_attn.transpose(-1, -2)[..., 1:].transpose(-1, -2)

            with torch.no_grad():
                attn_mask = (~mask).unsqueeze(-1).expand(mask.shape[0], mask.shape[1], text_mask.shape[-1]).float().transpose(-1, -2)
                attn_mask = attn_mask.float() * (~text_mask).unsqueeze(-1).expand(text_mask.shape[0], text_mask.shape[1], mask.shape[-1]).float()
                s2s_attn.masked_fill_((attn_mask < 1), 0.0)
                s2s_attn_mono = maximum_path(s2s_attn, mask_from_lens(s2s_attn, input_lengths, mel_input_length // (2 ** n_down)))

            t_en = model.text_encoder(texts, input_lengths, text_mask)
            asr = (t_en @ (s2s_attn if bool(random.getrandbits(1)) else s2s_attn_mono))
    
            mel_input_length_all = accelerator.gather(mel_input_length)
            mel_len = min([int(mel_input_length_all.min().item() / 2 - 1), max_len // 2])
            
            en, gt, wav = [], [], []
            for bib in range(len(mel_input_length)):
                mel_length_half = int(mel_input_length[bib].item() / 2)
                r_st = np.random.randint(0, mel_length_half - mel_len)
                en.append(asr[bib, :, r_st:r_st+mel_len])
                gt.append(mels[bib, :, (r_st * 2):((r_st+mel_len) * 2)])
                wav.append(torch.from_numpy(waves[bib][(r_st * 2) * 300:((r_st+mel_len) * 2) * 300]).to(device))

            en, gt, wav = torch.stack(en), torch.stack(gt).detach(), torch.stack(wav).float().detach()
            
            if gt.shape[-1] < 80:
                continue
            
            s = model.style_encoder(ref_t_vecs if multispeaker else t_vecs)
            
            with torch.no_grad():    
                real_norm = log_norm(gt.unsqueeze(1)).squeeze(1).detach()
                F0_real, _, _ = model.pitch_extractor(gt.unsqueeze(1))
            
            y_rec = model.decoder(en, F0_real, real_norm, s)
            
            if epoch >= TMA_epoch:
                optimizer.zero_grad()
                d_loss = dl(wav.unsqueeze(1), y_rec.detach()).mean()
                accelerator.backward(d_loss)
                optimizer.step('msd'); optimizer.step('mpd')
            else: d_loss = torch.tensor(0.0)

            optimizer.zero_grad()
            loss_mel = stft_loss(y_rec.squeeze(), wav)
            
            if epoch >= TMA_epoch: 
                loss_s2s = 0
                for _s2s_pred, _text_input, _text_length in zip(s2s_pred, texts, input_lengths):
                    loss_s2s += F.cross_entropy(_s2s_pred[:_text_length], _text_input[:_text_length])
                loss_s2s /= texts.size(0)
                loss_mono = F.l1_loss(s2s_attn, s2s_attn_mono) * 10
                loss_gen_all = gl(wav.unsqueeze(1).float(), y_rec).mean()
                loss_slm = wl(wav, y_rec).mean()
                g_loss = loss_params.lambda_mel * loss_mel + loss_params.lambda_mono * loss_mono + \
                         loss_params.lambda_s2s * loss_s2s + loss_params.lambda_gen * loss_gen_all + \
                         loss_params.lambda_slm * loss_slm
            else:
                loss_s2s = loss_mono = loss_gen_all = loss_slm = 0
                g_loss = loss_mel

            running_loss += accelerator.gather(loss_mel).mean().item()
            accelerator.backward(g_loss)
            
            optimizer.step('text_encoder'); optimizer.step('style_encoder'); optimizer.step('decoder')
            if epoch >= TMA_epoch: 
                optimizer.step('text_aligner'); optimizer.step('pitch_extractor')
            
            iters += 1
            if (i+1)%log_interval == 0 and accelerator.is_main_process:
                batch_pbar.set_postfix({"Mel": f"{running_loss/log_interval:.4f}"})
                log_print ('Epoch [%d/%d], Step [%d/%d], Mel Loss: %.5f, Gen Loss: %.5f, Disc Loss: %.5f, Mono Loss: %.5f, S2S Loss: %.5f, SLM Loss: %.5f'
                        %(epoch+1, epochs, i+1, len(train_dataloader), running_loss / log_interval, loss_gen_all, d_loss, loss_mono, loss_s2s, loss_slm), logger)
                
                writer.add_scalar('train/mel_loss', running_loss / log_interval, iters)
                writer.add_scalar('train/gen_loss', loss_gen_all, iters)
                writer.add_scalar('train/d_loss', d_loss, iters)
                writer.add_scalar('train/mono_loss', loss_mono, iters)
                writer.add_scalar('train/s2s_loss', loss_s2s, iters)
                writer.add_scalar('train/slm_loss', loss_slm, iters)
                running_loss = 0

        if accelerator.is_main_process:
            if exp_name != 'debug':
                if epoch % saving_epoch == 0:
                    print(f'Saving Checkpoint for Epoch {epoch}...')
                    state = {
                        'net': {k: model[k].state_dict() for k in model}, 
                        'optimizer': optimizer.state_dict(), 
                        'iters': iters, 
                        'val_loss': None, 
                        'epoch': epoch
                    }
                    torch.save(state, osp.join(log_dir, 'ckpt', 'epoch_1st_%05d.pth' % epoch))

    if accelerator.is_main_process:
        print('Saving Final Model...')
        state = {'net': {k: model[k].state_dict() for k in model}, 'optimizer': optimizer.state_dict(), 'iters': iters, 'val_loss': None, 'epoch': epoch}
        torch.save(state, osp.join(log_dir, config.get('first_stage_path', 'first_stage.pth')))

if __name__=="__main__":
    main()