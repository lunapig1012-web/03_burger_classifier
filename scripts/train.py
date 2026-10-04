import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from generate_fake_data import generate_epoch_fake_data
from utils import create_model, ensure_dir, get_device, list_images, set_seed


CLASSES = ["non_burger", "burger"]
CLASS_TO_IDX = {"non_burger": 0, "burger": 1}
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class ImagePathDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, int]], transform=None) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_path, label = self.samples[index]
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Failed to read image: {image_path}") from exc
        if self.transform:
            image = self.transform(image)
        return image, label


def count_images(folder: Path) -> int:
    return len(list_images(folder)) if folder.exists() else 0


def require_class_folder(root: Path, split: str, class_name: str) -> Path:
    folder = root / split / class_name
    if not folder.exists():
        raise FileNotFoundError(f"Required folder not found: {folder}")
    if count_images(folder) == 0:
        raise ValueError(f"Required folder has no images: {folder}")
    return folder


def optional_extra_folder(root: Optional[str], class_name: str) -> Optional[Path]:
    if not root:
        return None
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Optional data folder not found: {root_path}")

    class_folder = root_path / class_name
    if class_folder.exists():
        return class_folder
    if root_path.name == class_name:
        return root_path
    raise FileNotFoundError(f"Expected {root_path / class_name} or a direct {class_name} folder.")


def build_real_train_samples(data_dir: Path) -> list[tuple[Path, int]]:
    train_non = require_class_folder(data_dir, "train", "non_burger")
    train_burger = require_class_folder(data_dir, "train", "burger")
    train_samples = [(p, CLASS_TO_IDX["non_burger"]) for p in list_images(train_non)]
    train_samples += [(p, CLASS_TO_IDX["burger"]) for p in list_images(train_burger)]
    return train_samples


def build_samples(data_dir: Path, synthetic_dir: Optional[str], hard_negative_dir: Optional[str]) -> tuple[dict, dict]:
    train_non = require_class_folder(data_dir, "train", "non_burger")
    train_burger = require_class_folder(data_dir, "train", "burger")
    val_non = require_class_folder(data_dir, "val", "non_burger")
    val_burger = require_class_folder(data_dir, "val", "burger")
    test_non = require_class_folder(data_dir, "test", "non_burger")
    test_burger = require_class_folder(data_dir, "test", "burger")

    synthetic_burger = optional_extra_folder(synthetic_dir, "burger")
    hard_negative_non = optional_extra_folder(hard_negative_dir, "non_burger")

    train_samples = build_real_train_samples(data_dir)

    synthetic_count = 0
    if synthetic_burger:
        synthetic_images = list_images(synthetic_burger)
        if not synthetic_images:
            raise ValueError(f"Synthetic burger folder has no images: {synthetic_burger}")
        synthetic_count = len(synthetic_images)
        train_samples += [(p, CLASS_TO_IDX["burger"]) for p in synthetic_images]

    hard_negative_count = 0
    if hard_negative_non:
        hard_negative_images = list_images(hard_negative_non)
        if not hard_negative_images:
            raise ValueError(f"Hard negative non_burger folder has no images: {hard_negative_non}")
        hard_negative_count = len(hard_negative_images)
        train_samples += [(p, CLASS_TO_IDX["non_burger"]) for p in hard_negative_images]

    val_samples = [(p, CLASS_TO_IDX["non_burger"]) for p in list_images(val_non)]
    val_samples += [(p, CLASS_TO_IDX["burger"]) for p in list_images(val_burger)]
    test_samples = [(p, CLASS_TO_IDX["non_burger"]) for p in list_images(test_non)]
    test_samples += [(p, CLASS_TO_IDX["burger"]) for p in list_images(test_burger)]

    summary = {
        "real_train_burger_count": count_images(train_burger),
        "real_train_non_burger_count": count_images(train_non),
        "synthetic_burger_count": synthetic_count,
        "hard_negative_non_burger_count": hard_negative_count,
        "val_burger_count": count_images(val_burger),
        "val_non_burger_count": count_images(val_non),
        "test_burger_count": count_images(test_burger),
        "test_non_burger_count": count_images(test_non),
    }
    return {"train": train_samples, "val": val_samples, "test": test_samples}, summary


def build_dynamic_train_loader(
    data_dir: Path,
    synthetic_epoch_dir: Path,
    transform,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> tuple[DataLoader, int]:
    real_samples = build_real_train_samples(data_dir)
    synthetic_burger_dir = synthetic_epoch_dir / "burger"
    synthetic_images = list_images(synthetic_burger_dir)
    if not synthetic_images:
        raise ValueError(f"No generated synthetic images found in {synthetic_burger_dir}")
    samples = real_samples + [(p, CLASS_TO_IDX["burger"]) for p in synthetic_images]
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        ImagePathDataset(samples, transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=generator,
    )
    return loader, len(synthetic_images)


def build_transforms(image_size: int, augmentation: str) -> tuple[transforms.Compose, transforms.Compose]:
    eval_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    if augmentation == "none":
        return eval_tf, eval_tf
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(12),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return train_tf, eval_tf


def compute_metrics(y_true: list[int], y_pred: list[int]) -> tuple[dict, list[list[int]]]:
    cm = [[0, 0], [0, 0]]
    for true_label, pred_label in zip(y_true, y_pred):
        cm[true_label][pred_label] += 1

    total = len(y_true)
    correct = sum(1 for true_label, pred_label in zip(y_true, y_pred) if true_label == pred_label)
    tp = cm[CLASS_TO_IDX["burger"]][CLASS_TO_IDX["burger"]]
    fp = cm[CLASS_TO_IDX["non_burger"]][CLASS_TO_IDX["burger"]]
    fn = cm[CLASS_TO_IDX["burger"]][CLASS_TO_IDX["non_burger"]]

    accuracy = correct / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}, cm


def run_epoch(model, loader, criterion, device, optimizer=None) -> tuple[float, float, dict, list[list[int]]]:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss = 0.0
    total = 0
    y_true, y_pred = [], []

    with torch.set_grad_enabled(is_train):
        for images, labels in tqdm(loader, leave=False):
            images = images.to(device)
            labels = labels.to(device)
            if is_train:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if is_train:
                loss.backward()
                optimizer.step()

            preds = outputs.argmax(dim=1)
            total_loss += loss.item() * images.size(0)
            total += labels.size(0)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    metrics, cm = compute_metrics(y_true, y_pred)
    return total_loss / total, metrics["accuracy"], metrics, cm


def predict_dataset(model, dataset, device) -> tuple[list[int], list[int], list[tuple[torch.Tensor, str]], list[tuple[torch.Tensor, str]]]:
    model.eval()
    y_true, y_pred = [], []
    correct, wrong = [], []
    with torch.no_grad():
        for image, label in dataset:
            output = model(image.unsqueeze(0).to(device))
            pred = int(output.argmax(dim=1).item())
            y_true.append(label)
            y_pred.append(pred)
            caption = f"T:{CLASSES[label]} / P:{CLASSES[pred]}"
            if pred == label:
                correct.append((image, caption))
            else:
                wrong.append((image, caption))
    return y_true, y_pred, correct, wrong


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    return torch.clamp(tensor.cpu() * std + mean, 0, 1)


def save_examples(items: list[tuple[torch.Tensor, str]], output_path: Path, title: str, max_items: int = 16) -> None:
    if not items:
        print(f"[WARN] No images for {title}")
        return
    items = items[:max_items]
    cols = min(4, len(items))
    rows = (len(items) + cols - 1) // cols
    plt.figure(figsize=(cols * 3, rows * 3))
    for index, (image, caption) in enumerate(items, 1):
        plt.subplot(rows, cols, index)
        plt.imshow(denormalize(image).permute(1, 2, 0))
        plt.title(caption, fontsize=9)
        plt.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_confusion_matrix(cm: list[list[int]], output_path: Path) -> None:
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar()
    plt.xticks([0, 1], CLASSES, rotation=30, ha="right")
    plt.yticks([0, 1], CLASSES)
    max_value = max(max(row) for row in cm) if cm else 0
    threshold = max_value / 2
    for row in range(2):
        for col in range(2):
            color = "white" if cm[row][col] > threshold else "black"
            plt.text(col, row, str(cm[row][col]), ha="center", va="center", color=color)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_curves(history: list[dict], output_dir: Path) -> None:
    epochs = [row["epoch"] for row in history]

    plt.figure()
    plt.plot(epochs, [row["train_loss"] for row in history], label="train_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "train_loss.png", dpi=160)
    plt.close()

    plt.figure()
    plt.plot(epochs, [row["val_accuracy"] for row in history], label="val_accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "val_accuracy.png", dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train burger/non_burger experiments with fixed class order.")
    parser.add_argument("--data_dir", "--data-dir", default="data/base_real")
    parser.add_argument("--synthetic_dir", "--synthetic-dir", default=None)
    parser.add_argument("--dynamic_synthetic", "--dynamic-synthetic", action="store_true")
    parser.add_argument("--cutout_dir", "--cutout-dir", default="cutouts/burger")
    parser.add_argument("--background_dir", "--background-dir", default="backgrounds")
    parser.add_argument("--fake_per_epoch", "--fake-per-epoch", type=int, default=50)
    parser.add_argument("--hard_negative_dir", "--hard-negative-dir", default=None)
    parser.add_argument("--exp_name", "--exp-name", required=True)
    parser.add_argument("--augmentation", choices=["none", "basic"], default="none")
    parser.add_argument("--model", choices=["resnet18", "mobilenet_v3_small", "mobilenet_v3_large"], default="resnet18")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", "--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image_size", "--image-size", type=int, default=224)
    parser.add_argument("--num_workers", "--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no_pretrained", "--no-pretrained", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    output_dir = ensure_dir(Path("outputs") / args.exp_name)

    if args.dynamic_synthetic and args.hard_negative_dir:
        raise ValueError("Current Experiment C supports real + dynamic synthetic burger only. Do not pass --hard_negative_dir.")
    if args.dynamic_synthetic and args.synthetic_dir is not None:
        raise ValueError("Use --dynamic_synthetic with --cutout_dir and --background_dir, not --synthetic_dir.")

    samples, train_data_summary = build_samples(Path(args.data_dir), args.synthetic_dir, args.hard_negative_dir)
    train_tf, eval_tf = build_transforms(args.image_size, args.augmentation)

    train_ds = ImagePathDataset(samples["train"], train_tf)
    val_ds = ImagePathDataset(samples["val"], eval_tf)
    test_ds = ImagePathDataset(samples["test"], eval_tf)

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print("Class order: 0 = non_burger, 1 = burger")
    print(json.dumps(train_data_summary, indent=2))
    print(f"Using device: {device}")

    model = create_model(args.model, num_classes=2, pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_accuracy = -1.0
    best_epoch = 0
    history = []
    synthetic_epoch_dir = Path("data") / "synthetic" / "current_epoch"

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        if args.dynamic_synthetic:
            generate_epoch_fake_data(
                cutout_dir=args.cutout_dir,
                background_dir=args.background_dir,
                synthetic_dir=str(synthetic_epoch_dir),
                fake_per_epoch=args.fake_per_epoch,
                image_size=args.image_size,
                epoch=epoch,
                seed=args.seed + epoch,
                clear=True,
            )
            train_loader, synthetic_count = build_dynamic_train_loader(
                data_dir=Path(args.data_dir),
                synthetic_epoch_dir=synthetic_epoch_dir,
                transform=train_tf,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                seed=args.seed + epoch,
            )
            train_data_summary["synthetic_burger_count"] = synthetic_count

        train_loss, train_acc, train_metrics, _ = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc, val_metrics, _ = run_epoch(model, val_loader, criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "synthetic_burger_count": train_data_summary["synthetic_burger_count"],
        }
        history.append(row)
        print(json.dumps(row, indent=2))

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_epoch = epoch
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_name": args.model,
                "classes": CLASSES,
                "class_to_idx": CLASS_TO_IDX,
                "image_size": args.image_size,
                "best_epoch": best_epoch,
                "best_val_accuracy": best_val_accuracy,
                "augmentation": args.augmentation,
                "seed": args.seed,
            }, output_dir / "best_model.pth")

    try:
        checkpoint = torch.load(output_dir / "best_model.pth", map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(output_dir / "best_model.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    y_true, y_pred, correct_examples, wrong_examples = predict_dataset(model, test_ds, device)
    test_metrics, cm = compute_metrics(y_true, y_pred)

    metrics = {
        **test_metrics,
        "confusion_matrix": cm,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_accuracy,
        "model": args.model,
        "augmentation": args.augmentation,
        "dynamic_synthetic": args.dynamic_synthetic,
        "fake_per_epoch": args.fake_per_epoch if args.dynamic_synthetic else 0,
        "classes": CLASSES,
        "class_to_idx": CLASS_TO_IDX,
        "train_data_summary": train_data_summary,
    }

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with (output_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    save_curves(history, output_dir)
    save_confusion_matrix(cm, output_dir / "confusion_matrix.png")
    save_examples(correct_examples, output_dir / "correct_examples.png", "Correct examples")
    save_examples(wrong_examples, output_dir / "wrong_examples.png", "Wrong examples")

    print("Training complete.")
    print(json.dumps(metrics, indent=2))
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()