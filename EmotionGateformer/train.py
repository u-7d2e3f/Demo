import torch
import yaml
import os
import argparse
import logging
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW

from DataLoader import DubberSceneDataset, collate_fn
from EmotionGateformer import EmotionGateformer
from losses import EmotionGateformerLoss

def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def parse_args():
    parser = argparse.ArgumentParser(description="Training for EmotionGateformer")
    parser.add_argument("--config", type=str, default="Configs/Config.yml", help="Path to the yaml config")
    return parser.parse_args()

def setup_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("Training")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(os.path.join(log_dir, "train.log"))
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.addHandler(logging.StreamHandler())
    return logger


def train():
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    base_output = cfg['path'].get('base_output', "output")
    ckpt_dir = os.path.join(base_output, cfg['path'].get('ckpt_subdir', "checkpoints"))
    log_dir = os.path.join(base_output, cfg['path'].get('log_subdir', "logs"))
    tb_dir = os.path.join(log_dir, "tensorboard")
  
    
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)
    
    
    writer = SummaryWriter(tb_dir)
    logger = setup_logger(log_dir)

    train_ds = DubberSceneDataset(
        root_json_path=cfg['path']['root_json'],
        split="train",
        feature_root=cfg['path']['feature_root'],
        top_k=cfg['train']['top_k']
    )
    train_loader = DataLoader(
        train_ds, 
        batch_size=cfg['train']['batch_size'], 
        shuffle=True, 
        collate_fn=collate_fn
    )

    model = EmotionGateformer(
        d_speaker=cfg['model']['d_speaker'],
        d_face=cfg['model']['d_face'],
        d_env=cfg['model']['d_env'],
        d_text=cfg['model']['d_text'],
        d_ref=cfg['model']['d_ref'],
        d_arc=cfg['model']['d_arc'],
        d_model=cfg['model']['d_model'],
        d_out=cfg['model']['d_out'],
        nhead=cfg['model']['nhead'],
        num_layers_temp=cfg['model']['num_layers_temp'],
        num_layers_feat=cfg['model']['num_layers_feat'],
        num_layers_dec=cfg['model']['num_layers_dec'],
        dim_feedforward=cfg['model']['dim_feedforward'],
        dropout=cfg['model']['dropout'],
        max_seq_len=cfg['model']['max_seq_len']
    ).to(device)

    criterion = EmotionGateformerLoss(
        w_ccc=cfg['loss']['w_ccc'],
        w_delta=cfg['loss']['w_delta'],
        w_huber=cfg['loss']['w_huber']
    ).to(device)
    
    optimizer = AdamW(model.parameters(), lr=float(cfg['train']['learning_rate']))
    
    model.train()
    for epoch in range(cfg['train']['epochs']):
        epoch_loss = 0
        for i, batch in enumerate(train_loader):
            face = batch['face'].to(device)
            env = batch['env'].to(device)
            text = batch['text'].to(device)
            arc = batch['arc'].to(device)
            ref = batch['ref'].to(device)
            spk = batch['speaker'].to(device)
            target = batch['target'].to(device)
            mask = batch['src_key_padding_mask'].to(device)
            f_mask = batch['face_mask'].to(device) 

            optimizer.zero_grad()
            
            pred = model(
                face, env, text, ref, arc, spk, target, 
                src_key_padding_mask=mask,
                face_mask=f_mask 
            )
            
            loss, details = criterion(pred, target, spk, mask)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

            if (i + 1) % cfg['train']['print_step'] == 0:
                logger.info(f"Epoch [{epoch}]  | Loss: {loss.item():.4f} | Huber: {details['huber']:.4f} | CCC: {details['ccc']:.4f} | Delta: {details['delta']:.4f}")
                step = epoch * len(train_loader) + i
                for k, v in details.items():
                    writer.add_scalar(f"Loss/{k}", v, step)

        if (epoch + 1) % cfg['train']['save_step'] == 0:
            save_path = os.path.join(ckpt_dir, f"gateformer_ep{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)

    writer.close()
    logger.info("Training Complete!")

if __name__ == "__main__":
    train()
