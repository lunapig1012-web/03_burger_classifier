import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from utils import create_model, ensure_dir, list_images, load_checkpoint, get_device, set_seed


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
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def build_test_samples(data_dir: Path) -> list[tuple[Path, int]]:
    test_dir = data_dir / "test"
    non_dir = test_dir / "non_burger"
    burger_dir = test_dir / "burger"
    if not non_dir.exists() or not burger_dir.exists():
        raise FileNotFoundError(f"Expected test folders: {non_dir} and {burger_dir}")
    non_images = list_images(non_dir)
    burger_images = list_images(burger_dir)
    if not non_images or not burger_images:
        raise ValueError("Test burger and non_burger folders must both contain images.")
    samples = [(p, CLASS_TO_IDX["non_burger"]) for p in non_images]
    samples += [(p, CLASS_TO_IDX["burger"]) for p in burger_images]
    return samples


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained burger classifier on the fixed test set.")
    parser.add_argument("--data_dir", "--data-dir", default="data/base_real")
    parser.add_argument("--model_path", "--model-path", required=True)
    parser.add_argument("--output_dir", "--output-dir", default=None)
    parser.add_argument("--batch_size", "--batch-size", type=int, default=32)
    parser.add_argument("--num_workers", "--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    model_path = Path(args.model_path)
    output_dir = ensure_dir(args.output_dir if args.output_dir else model_path.parent)
    checkpoint = load_checkpoint(model_path, device)
    image_size = checkpoint.get("image_size", 224)
    model_name = checkpoint["model_name"]

    if checkpoint.get("classes") != CLASSES:
        raise ValueError(f"Checkpoint class order must be {CLASSES}, got {checkpoint.get('classes')}")

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    dataset = ImagePathDataset(build_test_samples(Path(args.data_dir)), transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = create_model(model_name, num_classes=2, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader):
            outputs = model(images.to(device))
            preds = outputs.argmax(dim=1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(labels.tolist())

    metric_values, cm = compute_metrics(y_true, y_pred)
    metrics = {**metric_values, "confusion_matrix": cm, "classes": CLASSES, "class_to_idx": CLASS_TO_IDX}
    with (output_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    save_confusion_matrix(cm, output_dir / "confusion_matrix.png")

    print(json.dumps(metrics, indent=2))
    print(f"Saved evaluation outputs to {output_dir}")


if __name__ == "__main__":
    main()
