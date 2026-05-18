#!/usr/bin/env python3
"""
Train UNet on full dataset with masked loss.
Valid pixels: not NaN AND not building (X>0 = building, Y=0 at buildings).
"""
from __future__ import annotations
import argparse, logging, random, time
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader, TensorDataset
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "models"))
from unet_lu import UNetLu

ROOT = Path(__file__).resolve().parent.parent

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--n-c", type=int, default=16)
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data/full_masked_640")
    p.add_argument("--results-dir", type=Path, default=ROOT / "results/full_masked_640/seed_1")
    p.add_argument("--device", default=None)
    return p.parse_args()

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    train_p = torch.load(args.data_dir / "train.pt", map_location="cpu", weights_only=False)
    test_p = torch.load(args.data_dir / "test.pt", map_location="cpu", weights_only=False)

    X_tr, Y_tr, mask_tr = train_p["X"], train_p["Y"], train_p["nan_mask"]
    X_te, Y_te, mask_te = test_p["X"], test_p["Y"], test_p["nan_mask"]

    # Valid = not NaN AND X <= 0 (open ground, not building)
    # X > 0 = building height, Y = 0 at buildings
    valid_tr = ((~mask_tr) & (X_tr <= 0)).float()
    valid_te = ((~mask_te) & (X_te <= 0)).float()

    logging.info(f"Train: {X_tr.shape[0]} patches, {int(valid_tr.sum().item()):,} valid pixels")
    logging.info(f"Test:  {X_te.shape[0]} patches, {int(valid_te.sum().item()):,} valid pixels")

    train_loader = DataLoader(TensorDataset(X_tr, Y_tr, valid_tr),
                              batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(TensorDataset(X_te, Y_te, valid_te),
                             batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = UNetLu(n_c=args.n_c).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    logging.info(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    def masked_l1_loss(pred, target, valid):
        # NaN-safe: (pred-target) can be NaN where Y=NaN, so use where
        diff = torch.where(valid.bool(), (pred - target).abs(), torch.zeros_like(pred))
        return diff.sum() / valid.sum().clamp(min=1)

    def run_epoch(loader, optimizer=None):
        model.train() if optimizer else model.eval()
        total_loss = 0.0
        with torch.set_grad_enabled(optimizer is not None):
            for batch in loader:
                x, y, v = [b.to(device) for b in batch]
                pred = model(x)
                loss = masked_l1_loss(pred, y, v)
                if optimizer:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                total_loss += loss.item() * x.size(0)
        return total_loss / len(loader.dataset)

    out = args.results_dir
    out.mkdir(parents=True, exist_ok=True)
    best_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = run_epoch(train_loader, optimizer)
        test_loss = run_epoch(test_loader, None)
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), out / "model_best.pt")
        if epoch % args.log_interval == 0 or epoch == 1:
            dt = time.time() - t0
            logging.info(f"Epoch {epoch:4d}/{args.epochs}  train={train_loss:.6f}  test={test_loss:.6f}  best={best_loss:.6f}  {dt:.1f}s")

    torch.save(model.state_dict(), out / "model_final.pt")
    logging.info(f"Done. Best test loss: {best_loss:.6f}")

if __name__ == "__main__":
    main()
