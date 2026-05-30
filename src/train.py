# Same training entry point as assignment #4.1. Train a model first, then attack it.
# Example: python src/train.py --model cnn --epochs 3
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from models import get_model


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


def main():
    parser = argparse.ArgumentParser(description="Train model to be attacked on MNIST.")
    parser.add_argument("--model", choices=["mlp", "cnn", "transformer"], default="cnn")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--out-dir", type=str, default="models")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    transform = transforms.ToTensor()
    train_ds = datasets.MNIST(args.data_dir, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(args.data_dir, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = get_model(args.model).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch}: clean test accuracy={acc*100:.2f}%")
        if acc > best_acc:
            best_acc = acc
            torch.save({"model_name": args.model, "state_dict": model.state_dict(), "test_acc": acc},
                       Path(args.out_dir) / f"{args.model}.pt")
    print(f"Best clean accuracy: {best_acc*100:.2f}%")


if __name__ == "__main__":
    main()
