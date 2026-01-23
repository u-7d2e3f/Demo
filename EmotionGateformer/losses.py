import torch
import torch.nn as nn
import torch.nn.functional as F

class EmotionGateformerLoss(nn.Module):
    def __init__(self, w_ccc=1.0, w_delta=5.0, w_huber=1.0, eps=1e-8):
        super(EmotionGateformerLoss, self).__init__()
        self.w_ccc = w_ccc
        self.w_delta = w_delta
        self.w_huber = w_huber
        self.eps = eps
        self.huber_fn = nn.HuberLoss(reduction='none')

    def compute_huber(self, pred, gt, mask=None):
        loss = self.huber_fn(pred, gt)
        if mask is not None:
            active_mask = (~mask).unsqueeze(-1).float()
            loss = (loss * active_mask).sum() / (active_mask.sum() * pred.size(-1) + self.eps)
        else:
            loss = loss.mean()
        return loss

    def compute_ccc(self, pred, gt, mask=None):
        if mask is not None:
            active_mask = ~mask
            pred_f = pred[active_mask]
            gt_f = gt[active_mask]
        else:
            pred_f = pred.view(-1, pred.size(-1))
            gt_f = gt.view(-1, gt.size(-1))

        mu_p = torch.mean(pred_f, dim=0)
        mu_g = torch.mean(gt_f, dim=0)
        var_p = torch.var(pred_f, dim=0)
        var_g = torch.var(gt_f, dim=0)
        cov = torch.mean((pred_f - mu_p) * (gt_f - mu_g), dim=0)

        ccc_per_dim = (2 * cov) / (var_p + var_g + (mu_p - mu_g)**2 + self.eps)
        
        return 1.0 - torch.mean(ccc_per_dim)

    def compute_speaker_delta(self, pred, gt, d_speaker, mask=None):
        B, S, D = pred.shape
        device = pred.device

        spk_norm = F.normalize(d_speaker, p=2, dim=-1)
        sim_matrix = torch.bmm(spk_norm, spk_norm.transpose(1, 2))

        causal_mask = torch.tril(torch.ones(S, S, device=device), diagonal=-1)
        identity_mask = (sim_matrix > 0.99) * causal_mask.unsqueeze(0)
        
        if mask is not None:
            valid_pair = ~(mask.unsqueeze(1) | mask.unsqueeze(2))
            identity_mask = identity_mask * valid_pair

        idx_weight = torch.arange(S, device=device).view(1, 1, S)
        prev_indices = torch.argmax(identity_mask * idx_weight, dim=-1)
        
        has_prev = identity_mask.sum(dim=-1) > 0

        batch_idx = torch.arange(B, device=device).view(B, 1).expand(B, S)
        prev_pred = pred[batch_idx, prev_indices]
        prev_gt = gt[batch_idx, prev_indices]

        delta_pred = pred - prev_pred
        delta_gt = gt - prev_gt
        
        diff_sq = torch.sum((delta_pred - delta_gt)**2, dim=-1)
        loss = (diff_sq * has_prev).sum() / (has_prev.sum() + self.eps)
        return loss / D

    def forward(self, pred, gt, d_speaker, padding_mask=None):
        l_huber = self.compute_huber(pred, gt, padding_mask)
        l_ccc = self.compute_ccc(pred, gt, padding_mask)
        l_delta = self.compute_speaker_delta(pred, gt, d_speaker, padding_mask)

        total_loss = (self.w_huber * l_huber) + \
                     (self.w_ccc * l_ccc) + \
                     (self.w_delta * l_delta)

        return total_loss, {
            "total": total_loss.item(),
            "ccc": l_ccc.item(),
            "delta": l_delta.item(),
            "huber": l_huber.item()
        }