"""Training and evaluation routines for the Bayesian CNN."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import Config
from model import BayesianCNN, enable_dropout


def build_model(cfg: Config, n_train_points: int, device):
    """Create a fresh model + Adam optimizer.

    Weight decay scales with dataset size (length-scale heuristic from the paper)
    but is (1) capped to a sane maximum and (2) applied only to the classifier
    (dense) layers, matching the reference setup. Regularising the conv filters
    with the very large decay that arises at small N caused training to collapse.
    """
    model = BayesianCNN(cfg).to(device)
    decay = min(cfg.weight_decay_base / max(n_train_points, 1), cfg.weight_decay_max)
    optimizer = torch.optim.Adam(
        [
            {"params": model.features.parameters(), "weight_decay": 0.0},
            {"params": model.classifier.parameters(), "weight_decay": decay},
        ],
        lr=cfg.lr,
    )
    return model, optimizer


def train(model, optimizer, dataset, val_dataset, cfg: Config, device):
    """Train on the labelled set, keeping the weights with best validation acc.

    Using the best checkpoint (instead of the final-epoch weights) prevents a
    single bad epoch from tanking the reported test accuracy, which removes the
    sharp V-shaped dips seen when reporting the last epoch only.
    """
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()

    best_acc = -1.0
    best_state = None
    for _ in range(cfg.epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        # Checkpoint on (deterministic, fast) validation accuracy.
        val_acc, _ = evaluate(model, val_dataset, cfg, device)
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_acc


@torch.no_grad()
def evaluate(model, dataset, cfg: Config, device, mc: bool = False):
    """Return (accuracy %, average loss).

    mc=False : single deterministic pass (dropout off) -- fast, used for checkpoints.
    mc=True  : MC-dropout, averaging softmax over `cfg.mc_samples` passes, as the
               paper does at test time. Slightly more stable, a bit slower.
    """
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)
    model.eval()
    if mc:
        enable_dropout(model)

    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if mc:
            probs = torch.zeros(x.size(0), cfg.num_classes, device=device)
            for _ in range(cfg.mc_samples):
                probs += torch.softmax(model(x), dim=1)
            probs /= cfg.mc_samples
            total_loss += nn.functional.nll_loss(
                torch.log(probs + 1e-12), y, reduction="sum").item()
            correct += (probs.argmax(dim=1) == y).sum().item()
        else:
            logits = model(x)
            total_loss += nn.functional.cross_entropy(
                logits, y, reduction="sum").item()
            correct += (logits.argmax(dim=1) == y).sum().item()
        n += y.size(0)
    return 100.0 * correct / n, total_loss / n
