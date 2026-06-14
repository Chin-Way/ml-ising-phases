"""A small convolutional network that learns the Ising phase from raw spins.

Where :mod:`ising.model` flattens each ``L x L`` configuration and hands it to a
fully-connected net, this CNN keeps the lattice intact and looks at it the way
the physics does -- **locally** and with the lattice's **periodic boundary
conditions** (the convolutions use ``circular`` padding, so a spin on the right
edge really does neighbor the left edge). After two convolutions it takes a
**global average** over the lattice, which makes the network translation
invariant and, as a bonus, independent of ``L`` -- the same weights run on any
size, which is convenient for the finite-size study.

The question this module exists to answer is *does that spatial inductive bias
help?* It is trained exactly like the baseline -- only on configurations drawn
deep in each phase -- and asked for the ``P(disordered) = 1/2`` crossing in the
critical region it never saw. See :mod:`ising.compare` for the head-to-head.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .dataset import make_dataset
from .model import average_by_temperature, crossing
from .montecarlo import TC_EXACT


class IsingCNN(nn.Module):
    """Two circular-padded conv layers, global average pool, small dense head.

    Deliberately tiny (a few thousand parameters): enough to pick up local
    alignment, small enough to train in seconds on a CPU.
    """

    def __init__(self, n_filters: int = 8, hidden: int = 32):
        super().__init__()
        pad = dict(padding=1, padding_mode="circular")
        self.conv1 = nn.Conv2d(1, n_filters, kernel_size=3, **pad)
        self.conv2 = nn.Conv2d(n_filters, n_filters, kernel_size=3, **pad)
        self.fc1 = nn.Linear(n_filters, hidden)
        self.fc2 = nn.Linear(hidden, 2)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = x.mean(dim=(2, 3))  # global average pool -> (N, n_filters)
        x = self.act(self.fc1(x))
        return self.fc2(x)


def _as_images(X: np.ndarray, L: int) -> torch.Tensor:
    """Reshape flat ``(N, L*L)`` spins to a ``(N, 1, L, L)`` float tensor."""
    return torch.as_tensor(X, dtype=torch.float32).reshape(-1, 1, L, L)


def train_cnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    L: int,
    epochs: int = 40,
    lr: float = 1e-3,
    batch_size: int = 64,
    n_filters: int = 8,
    hidden: int = 32,
    seed: int = 0,
) -> IsingCNN:
    """Train :class:`IsingCNN` on flattened deep-phase configurations."""
    torch.manual_seed(seed)
    net = IsingCNN(n_filters=n_filters, hidden=hidden)
    images = _as_images(X_train, L)
    labels = torch.as_tensor(y_train, dtype=torch.long)

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    n = images.shape[0]
    rng = np.random.default_rng(seed)

    net.train()
    for _ in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            opt.zero_grad()
            loss = loss_fn(net(images[idx]), labels[idx])
            loss.backward()
            opt.step()
    return net


def cnn_proba(net: IsingCNN, X: np.ndarray, L: int) -> np.ndarray:
    """Return ``P(disordered)`` for each configuration."""
    net.eval()
    with torch.no_grad():
        logits = net(_as_images(X, L))
        return torch.softmax(logits, dim=1)[:, 1].numpy()


def estimate_tc_cnn(
    L: int = 16,
    n_per_temp: int = 80,
    train_below: float = 1.7,
    train_above: float = 2.9,
    epochs: int = 40,
    seed: int = 0,
):
    """CNN counterpart of :func:`ising.model.estimate_tc`.

    Trains on ``T < train_below`` or ``T > train_above`` only, then reads off the
    ``P(disordered) = 1/2`` crossing across the full temperature range.
    """
    temperatures = np.linspace(1.0, 3.5, 26)
    X, y, T = make_dataset(L=L, temperatures=temperatures, n_per_temp=n_per_temp, seed=seed)

    train = (T < train_below) | (T > train_above)
    y_train = (T[train] > train_above).astype(int)

    net = train_cnn(X[train], y_train, L, epochs=epochs, seed=seed)

    train_acc = float(((cnn_proba(net, X[train], L) > 0.5).astype(int) == y_train).mean())
    proba = cnn_proba(net, X, L)
    p_by_T = average_by_temperature(proba, T, temperatures)
    tc_est = crossing(temperatures, p_by_T)

    return {
        "temperatures": temperatures,
        "p_disordered": p_by_T,
        "tc_estimate": tc_est,
        "tc_exact": TC_EXACT,
        "train_accuracy": train_acc,
        "network": net,
        "train_below": train_below,
        "train_above": train_above,
    }
