import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils import create_model, ensure_dir, get_device, list_images, load_checkpoint, set_seed


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


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    return torch.clamp(tensor.cpu() * std + mean, 0, 1)


def save_grid(items: list[tuple[torch.Tensor, str]], out_path: Path, title: str, max_items: int) -> None:
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
    plt.savefig(out_path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Save grids of correct and wrong predictions on the fixed test set.")
    parser.add_argument("--data_dir", "--data-dir", default="data/base_real")
    parser.add_argument("--model_path", "--model-path", required=True)
    parser.add_argument("--output_dir", "--output-dir", default=None)
    parser.add_argument("--max_images", "--max-images", type=int, default=16)
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

    model = create_model(model_name, num_classes=2, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    correct, wrong = [], []
    with torch.no_grad():
        for image, label in dataset:
            output = model(image.unsqueeze(0).to(device))
            pred = int(output.argmax(dim=1).item())
            caption = f"T:{CLASSES[label]} / P:{CLASSES[pred]}"
            if pred == label:
                correct.append((image, caption))
            else:
                wrong.append((image, caption))

    save_grid(correct, output_dir / "correct_examples.png", "Correct examples", args.max_images)
    save_grid(wrong, output_dir / "wrong_examples.png", "Wrong examples", args.max_images)
    print(f"Saved prediction visualizations to {output_dir}")


if __name__ == "__main__":
    main()
