"""
Train a Behavior Cloning policy from recorded human play CSV files saved by game_engine.start_recording().

Saves a PyTorch model and normalization metadata (mean/std) into models/.

Usage:
  python scripts/train_bc.py --data-dir data --out models/bc_policy.pth --epochs 30
"""
import os
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def load_data(data_dir):
    pattern = os.path.join(data_dir, "record_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No recording CSV files found in {data_dir}")
    xs = []
    ys = []
    for f in files:
        data = np.loadtxt(f, delimiter=",", skiprows=1)
        if data.size == 0:
            continue
        if data.ndim == 1:
            data = data.reshape(1, -1)
        x = data[:, :-1]
        y = data[:, -1]
        xs.append(x)
        ys.append(y)
    X = np.vstack(xs).astype(np.float32)
    Y = np.hstack(ys).astype(np.float32)
    return X, Y


class BCModel(nn.Module):
    def __init__(self, input_dim=5, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train(args):
    X, Y = load_data(args.data_dir)
    # Normalize
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    Xn = (X - mean) / std

    dataset = TensorDataset(torch.from_numpy(Xn), torch.from_numpy(Y))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = BCModel(input_dim=X.shape[1], hidden=args.hidden)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
        avg = total_loss / len(dataset)
        if epoch % max(1, args.epochs // 10) == 0 or epoch == args.epochs:
            print(f"Epoch {epoch}/{args.epochs}  loss={avg:.6f}")

    # Save model and metadata
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(model.cpu(), args.out)
    meta_path = os.path.splitext(args.out)[0] + '_meta.npz'
    np.savez(meta_path, mean=mean, std=std)
    print(f"Saved BC model to {args.out} and metadata to {meta_path}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default=os.path.join(os.path.dirname(__file__), '..', 'data'))
    p.add_argument('--out', default=os.path.join(os.path.dirname(__file__), '..', 'models', 'bc_policy.pth'))
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--hidden', type=int, default=64)
    args = p.parse_args()
    train(args)
