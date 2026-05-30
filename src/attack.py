import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import get_model


def fgsm(model, x, y, epsilon):
    x_adv = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x_adv), y)
    model.zero_grad(set_to_none=True)
    loss.backward()
    return torch.clamp(x_adv + epsilon * x_adv.grad.sign(), 0.0, 1.0).detach()


def pgd(model, x, y, epsilon, alpha, steps):
    x_orig = x.clone().detach()
    x_adv = x_orig.clone().detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        model.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():
            x_adv = x_adv + alpha * x_adv.grad.sign()
            delta = torch.clamp(x_adv - x_orig, min=-epsilon, max=epsilon)
            x_adv = torch.clamp(x_orig + delta, 0.0, 1.0).detach()
    return x_adv


def mifgsm(model, x, y, epsilon, alpha, steps, decay=1.0):
    x_orig = x.clone().detach()
    x_adv = x_orig.clone().detach()
    momentum = torch.zeros_like(x_adv)
    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        model.zero_grad(set_to_none=True)
        loss.backward()
        grad = x_adv.grad.detach()
        grad = grad / (grad.abs().mean(dim=(1, 2, 3), keepdim=True) + 1e-8)
        momentum = decay * momentum + grad
        with torch.no_grad():
            x_adv = x_adv + alpha * momentum.sign()
            delta = torch.clamp(x_adv - x_orig, min=-epsilon, max=epsilon)
            x_adv = torch.clamp(x_orig + delta, 0.0, 1.0).detach()
    return x_adv


def evaluate_attack(model, loader, device, attack_name, epsilon, alpha, steps):
    model.eval()
    clean_correct = 0
    adv_correct = 0
    attacked_correct_samples = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            clean_pred = model(x).argmax(dim=1)
        clean_mask = clean_pred.eq(y)
        clean_correct += clean_mask.sum().item()
        total += y.size(0)

        if attack_name == "fgsm":
            x_adv = fgsm(model, x, y, epsilon)
        elif attack_name == "pgd":
            x_adv = pgd(model, x, y, epsilon, alpha, steps)
        elif attack_name == "mifgsm":
            x_adv = mifgsm(model, x, y, epsilon, alpha, steps)
        else:
            raise ValueError("attack_name must be fgsm, pgd, or mifgsm")

        with torch.no_grad():
            adv_pred = model(x_adv).argmax(dim=1)
        adv_correct += adv_pred.eq(y).sum().item()

        # Untargeted Attack Success Rate: among originally correct samples, how many are now wrong?
        attacked_correct_samples += (clean_mask & adv_pred.ne(y)).sum().item()

    clean_acc = clean_correct / total
    adv_acc = adv_correct / total
    asr = attacked_correct_samples / max(clean_correct, 1)
    return clean_acc, adv_acc, asr


def main():
    parser = argparse.ArgumentParser(description="Attack trained MNIST model using FGSM, PGD, and Momentum I-FGSM.")
    parser.add_argument("--model", choices=["mlp", "cnn", "transformer"], default="cnn")
    parser.add_argument("--attacks", nargs="+", choices=["fgsm", "pgd", "mifgsm"], default=["fgsm", "pgd", "mifgsm"])
    parser.add_argument("--epsilon", type=float, default=0.30)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--out", default="results/attack_results.csv")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = Path(args.model_dir) / f"{args.model}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}. Train first with python src/train.py --model {args.model}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model = get_model(args.model).to(device)
    model.load_state_dict(ckpt["state_dict"])

    test_ds = datasets.MNIST(args.data_dir, train=False, download=True, transform=transforms.ToTensor())
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    rows = []
    for attack_name in args.attacks:
        clean_acc, adv_acc, asr = evaluate_attack(model, loader, device, attack_name, args.epsilon, args.alpha, args.steps)
        row = {
            "model": args.model,
            "attack": attack_name,
            "epsilon": args.epsilon,
            "alpha": args.alpha if attack_name != "fgsm" else "n/a",
            "steps": args.steps if attack_name != "fgsm" else "n/a",
            "clean_accuracy_percent": round(clean_acc * 100, 2),
            "adversarial_accuracy_percent": round(adv_acc * 100, 2),
            "attack_success_rate_percent": round(asr * 100, 2),
        }
        rows.append(row)
        print(row)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved attack results to {args.out}")


if __name__ == "__main__":
    main()
