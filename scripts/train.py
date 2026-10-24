"""Train PA-TSMixer or a TSMixer control with the settings of the paper and evaluate it on the test split.

    python scripts/train.py --data data --dataset trajair --horizon 30 --seed 42
    python scripts/train.py --data data --dataset trajair --horizon 30 --seed 42 --model TSMixer
    python scripts/train.py ... --model TSMixer --d-model 167 --last-residual        # matched-size control

Training (CONFIGS.md): AdamW, learning rate 1e-3, weight decay 1e-4, batch 128, gradient-norm clipping at 1,
cosine decay over at most 30 epochs, early stopping with patience 8, MSE on standardized coordinates, FP32 with
TF32 disabled. The checkpoint with the lowest validation 3D ADE (m) is restored and evaluated on the test split.
GPU training is not bitwise deterministic, so a retrained model matches the released one up to seed-level variation.
"""
import argparse, json, random, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from pa_tsmixer.data import Windows                # noqa: E402
from pa_tsmixer.models import build                # noqa: E402
from evaluate import evaluate                      # noqa: E402


def standardized(windows):
    """All windows of one split as standardized history [n, 60, 3] and target [n, H, 3] (float32)."""
    history, truth = windows.batch(0, len(windows))
    target = ((truth - windows.mean) / windows.scale).astype(np.float32)
    return torch.from_numpy(history), torch.from_numpy(target)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=ROOT / "data")
    p.add_argument("--dataset", choices=["trajair", "tartanaviation"], required=True)
    p.add_argument("--horizon", type=int, choices=[10, 20, 30, 60], required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", choices=["PA-TSMixer", "TSMixer"], default="PA-TSMixer")
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--last-residual", action=argparse.BooleanOptionalAction, default=None,
                   help="add the forecast to the last observation (default: on for PA-TSMixer, off for TSMixer)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", type=Path, default=ROOT / "runs")
    a = p.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = torch.device(a.device)
    kwargs = dict(d_model=a.d_model)
    if a.last_residual is not None:
        kwargs["last_residual"] = a.last_residual
    model = build(a.model, a.horizon, **kwargs).to(device)
    name = f"{a.dataset}_H{a.horizon}_{a.model.lower()}_d{a.d_model}_s{a.seed}"
    out = a.output / name
    out.mkdir(parents=True, exist_ok=True)

    train = Windows(a.data / a.dataset, "train", a.horizon)
    val = Windows(a.data / a.dataset, "val", a.horizon)
    x_train, y_train = standardized(train)
    x_train, y_train = x_train.to(device), y_train.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=a.epochs)
    order = torch.Generator().manual_seed(a.seed)
    best, stale, history = float("inf"), 0, []
    print(f"{name}: {sum(q.numel() for q in model.parameters()):,} parameters, {len(train):,} training windows",
          flush=True)
    for epoch in range(1, a.epochs + 1):
        begin = time.perf_counter()
        model.train()
        total = 0.
        for idx in torch.randperm(len(train), generator=order).split(a.batch_size):
            idx = idx.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(x_train[idx]), y_train[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(idx)
        v = {k: float(x) for k, x in evaluate(model, val, device).items()}
        improved = v["ADE"] < best
        if improved:
            best, stale = v["ADE"], 0
            torch.save(model.state_dict(), out / "checkpoint.pt")
        else:
            stale += 1
        history.append(dict(epoch=epoch, train_mse=total / len(train), val_ADE=v["ADE"], selected=improved))
        print(f"epoch {epoch:2d}  train MSE {total / len(train):.5f}  val ADE {v['ADE']:8.2f} m"
              f"{'  *' if improved else ''}  ({time.perf_counter() - begin:.0f} s)", flush=True)
        if stale >= a.patience:
            break
        schedule.step()

    model.load_state_dict(torch.load(out / "checkpoint.pt", map_location=device, weights_only=True))
    test = {k: float(x) for k, x in evaluate(model, Windows(a.data / a.dataset, "test", a.horizon), device).items()}
    print(f"test  ADE {test['ADE']:.2f} m  FDE {test['FDE']:.2f} m  MSE {test['MSE']:.1f} m^2", flush=True)
    (out / "result.json").write_text(json.dumps(dict(args={k: str(v) for k, v in vars(a).items()}, model_kwargs=kwargs,
                                                     test=test, best_val_ADE=best, history=history), indent=1),
                                     encoding="utf-8")


if __name__ == "__main__":
    main()
