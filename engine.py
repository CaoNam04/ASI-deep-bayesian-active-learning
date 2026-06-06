"""Training and evaluation routines for the Bayesian CNN."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import Config
from model import BayesianCNN


def build_model(cfg: Config, n_train_points: int, device):
    """Create a fresh model + Adam optimizer.

    Weight decay scales with dataset size (length-scale heuristic from the paper),
    so smaller training sets get stronger regularisation.
    """
    model = BayesianCNN(cfg).to(device)
    decay = cfg.weight_decay_base / max(n_train_points, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=decay)
    return model, optimizer


def train(model, optimizer, dataset, cfg: Config, device):
    """Train the model to convergence on the current labelled set."""
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    model.train()
    last_loss = 0.0
    for _ in range(cfg.epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            last_loss = loss.item()
    return last_loss


@torch.no_grad()
def evaluate(model, dataset, cfg: Config, device):
    """Return (accuracy %, average loss) using deterministic dropout (eval mode)."""
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss(reduction="sum")
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item()
        correct += (logits.argmax(dim=1) == y).sum().item()
        n += y.size(0)
    return 100.0 * correct / n, total_loss / n
