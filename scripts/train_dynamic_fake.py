import argparse
import csv
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from generate_fake_data import generate_epoch_fake_data
from utils import clear_dir, create_model, ensure_dir, get_device, list_images, set_seed


def compute_metrics(y_true: list[int], y_pred: list[int], positive_label: int = 0) -> tuple[dict, list[list[int]]]:
    num_classes = 2
    cm = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1

    total = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    tp = cm[positive_label][positive_label]
    fp = sum(cm[r][positive_label] for r in range(num_classes) if r != positive_label)
    fn = sum(cm[positive_label][c] for c in range(num_classes) if c != positive_label)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = correct / total if total else 0.0
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1}, cm


def build_eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def build_train_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def resolve_class_root(root: Path) -> Path:
    train_burger = root / "train" / "burger"
    if train_burger.exists():
        return root / "train"
    raise FileNotFoundError(f"Expected manually prepared train split at {root / 'train'}")


def load_real_datasets(real_data_dir: Path, image_size: int, val_dir: Optional[Path] = None) -> tuple[datasets.ImageFolder, datasets.ImageFolder, list[str]]:
    train_root = resolve_class_root(real_data_dir)
    train_ds = datasets.ImageFolder(train_root, transform=build_train_transform(image_size))
    val_root = val_dir if val_dir is not None else real_data_dir / "val"
    if not val_root.exists():
        raise FileNotFoundError(f"Expected manually prepared validation split at {val_root}")
    val_ds = datasets.ImageFolder(val_root, transform=build_eval_transform(image_size))
    if not train_ds.samples:
        raise ValueError("Training split is empty.")
    if not val_ds.samples:
        raise ValueError("Validation split is empty.")
    return train_ds, val_ds, train_ds.classes


def limit_real_train_dataset(train_ds: datasets.ImageFolder, max_real_train: Optional[int], seed: int):
    if max_real_train is None or max_real_train <= 0 or len(train_ds) <= max_real_train:
        return train_ds

    rng = torch.Generator().manual_seed(seed)
    class_to_indices: dict[int, list[int]] = {}
    for idx, (_, label) in enumerate(train_ds.samples):
        class_to_indices.setdefault(label, []).append(idx)

    selected: list[int] = []
    classes = sorted(class_to_indices.keys())
    per_class = max_real_train // len(classes)
    remainder = max_real_train % len(classes)

    for pos, label in enumerate(classes):
        indices = class_to_indices[label]
        shuffled_order = torch.randperm(len(indices), generator=rng).tolist()
        take = min(len(indices), per_class + (1 if pos < remainder else 0))
        selected.extend(indices[i] for i in shuffled_order[:take])

    if len(selected) < max_real_train:
        used = set(selected)
        remaining = [idx for idx in range(len(train_ds)) if idx not in used]
        shuffled_remaining = torch.randperm(len(remaining), generator=rng).tolist()
        need = min(max_real_train - len(selected), len(remaining))
        selected.extend(remaining[i] for i in shuffled_remaining[:need])

    selected = selected[:max_real_train]
    print(f"Using fixed real training images: {len(selected)} / {len(train_ds)}")
    return Subset(train_ds, selected)


def build_train_loader(real_train_ds, synthetic_dir: Path, image_size: int, batch_size: int, num_workers: int):
    synthetic_burger_dir = synthetic_dir / "burger"
    synthetic_ds = datasets.ImageFolder(synthetic_dir, transform=build_train_transform(image_size))
    datasets_to_concat = [real_train_ds]
    if synthetic_burger_dir.exists() and list_images(synthetic_burger_dir):
        datasets_to_concat.append(synthetic_ds)
    combined = ConcatDataset(datasets_to_concat)
    return DataLoader(combined, batch_size=batch_size, shuffle=True, num_workers=num_workers)


def train_one_epoch(model, loader, criterion, optimizer, device) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total = 0
    correct = 0
    for images, labels in tqdm(loader, leave=False):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device) -> tuple[float, float, list[int], list[int]]:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for images, labels in tqdm(loader, leave=False):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            preds = outputs.argmax(dim=1)
            total_loss += loss.item() * images.size(0)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
    return total_loss / total, correct / total, y_true, y_pred


def save_curves(history: list[dict], out_dir: Path) -> None:
    epochs = [row["epoch"] for row in history]
    plt.figure()
    plt.plot(epochs, [row["train_loss"] for row in history], label="train_loss")
    plt.plot(epochs, [row["val_loss"] for row in history], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "figures" / "loss_curve.png", dpi=160)
    plt.close()

    plt.figure()
    plt.plot(epochs, [row["train_accuracy"] for row in history], label="train_accuracy")
    plt.plot(epochs, [row["val_accuracy"] for row in history], label="val_accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "figures" / "accuracy_curve.png", dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train burger/non-burger classifier with dynamic fake data each epoch.")
    parser.add_argument("--real_data_dir", required=True)
    parser.add_argument("--cutout_dir", required=True)
    parser.add_argument("--background_dir", required=True)
    parser.add_argument("--synthetic_dir", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--fake_per_epoch", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model", default="resnet18", choices=["resnet18", "mobilenet_v3_small"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--val_dir", default=None, help="Optional override for validation split path.")
    parser.add_argument("--test_dir", default=None, help="Optional override for test split path.")
    parser.add_argument("--max_real_train", type=int, default=None, help="Use a fixed subset of this many real training images.")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    out_dir = ensure_dir(args.output_dir)
    ensure_dir(out_dir / "models")
    ensure_dir(out_dir / "figures")
    ensure_dir(out_dir / "predictions")

    real_data_dir = Path(args.real_data_dir)
    train_real_ds, val_ds, classes = load_real_datasets(real_data_dir, args.image_size, Path(args.val_dir) if args.val_dir else None)
    train_real_ds = limit_real_train_dataset(train_real_ds, args.max_real_train, args.seed)

    model = create_model(args.model, num_classes=len(classes), pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    synthetic_epoch_dir = Path(args.synthetic_dir) / "current_epoch"
    history: list[dict] = []
    best_val_f1 = -1.0
    best_path = out_dir / "models" / "best_model.pth"
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    if "burger" not in class_to_idx or "non_burger" not in class_to_idx:
        raise ValueError("Expected classes: burger and non_burger")
    positive_label = class_to_idx["burger"]

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        clear_dir(synthetic_epoch_dir)
        generate_epoch_fake_data(
            cutout_dir=args.cutout_dir,
            background_dir=args.background_dir,
            synthetic_dir=str(synthetic_epoch_dir),
            fake_per_epoch=args.fake_per_epoch,
            image_size=args.image_size,
            epoch=epoch,
            seed=args.seed + epoch,
            clear=False,
        )

        train_loader = build_train_loader(train_real_ds, synthetic_epoch_dir, args.image_size, args.batch_size, args.num_workers)
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, y_true, y_pred = evaluate(model, DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers), criterion, device)
        metrics, cm = compute_metrics(y_true, y_pred, positive_label=positive_label)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
        }
        history.append(row)
        print(json.dumps(row, indent=2))

        if metrics["f1_score"] > best_val_f1:
            best_val_f1 = metrics["f1_score"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_name": args.model,
                "classes": classes,
                "image_size": args.image_size,
                "seed": args.seed,
                "best_val_f1": best_val_f1,
                "positive_class": "burger",
            }, best_path)

    with (out_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    save_curves(history, out_dir)
    print(f"Training complete. Best val F1: {best_val_f1:.4f}")


if __name__ == "__main__":
    main()
