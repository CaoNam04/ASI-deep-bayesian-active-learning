"""Training and AUC evaluation for the ISIC experiment (section 5.5)."""
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from isic.config import ISICConfig
from isic.model import enable_dropout


def build_optimizer(model, cfg: ISICConfig, n_train_points: int):
    """Adam with the paper's length-scale weight decay: (1 - p) * l^2 / N."""
    decay = (1.0 - cfg.dropout_p) * cfg.length_scale_sq / max(n_train_points, 1)
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.Adam(params, lr=cfg.lr, weight_decay=decay)


def train(model, dataset, cfg: ISICConfig, device, n_train_points):
    """Fine-tune the model on the current labelled set for a fixed budget."""
    optimizer = build_optimizer(model, cfg, n_train_points)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _ in range(cfg.epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()


@torch.no_grad()
def evaluate_auc(model, dataset, cfg: ISICConfig, device):
    """MC-dropout AUC: average P(malignant) over T passes, then ROC-AUC."""
    model.eval()
    enable_dropout(model)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    probs_sum, ys = None, []
    for t in range(cfg.mc_samples):
        batch_probs, batch_ys = [], []
        for x, y in loader:
            p = torch.softmax(model(x.to(device)), dim=1)[:, 1]  # P(malignant)
            batch_probs.append(p.cpu())
            if t == 0:
                batch_ys.append(y)
        cur = torch.cat(batch_probs)
        probs_sum = cur if probs_sum is None else probs_sum + cur
        if t == 0:
            ys = torch.cat(batch_ys).numpy()
    mean_probs = (probs_sum / cfg.mc_samples).numpy()
    return roc_auc_score(ys, mean_probs)
