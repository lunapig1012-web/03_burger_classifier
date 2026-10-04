import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from utils import ensure_dir


def locate_image(json_path: Path, data: dict, image_dir: Optional[Path]) -> Path:
    candidates = []
    image_path = data.get("imagePath")
    if image_path:
        candidates.append(json_path.parent / image_path)
        if image_dir is not None:
            candidates.append(image_dir / Path(image_path).name)
            candidates.extend(image_dir.glob(Path(image_path).stem + ".*"))
    if image_dir is not None:
        candidates.extend(image_dir.glob(json_path.stem + ".*"))

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find source image for {json_path}")


def polygon_mask(size: tuple[int, int], points: list[list[float]]) -> Image.Image:
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    polygon = [(float(x), float(y)) for x, y in points]
    draw.polygon(polygon, outline=255, fill=255)
    return mask


def build_alpha_mask(image_size: tuple[int, int], shapes: list[dict], label: str, feather_radius: float) -> Image.Image:
    combined = Image.new("L", image_size, 0)
    for shape in shapes:
        if shape.get("shape_type", "polygon") != "polygon":
            continue
        if shape.get("label") != label:
            continue
        points = shape.get("points", [])
        if len(points) < 3:
            continue
        poly = polygon_mask(image_size, points)
        combined = Image.fromarray(np.maximum(np.array(combined), np.array(poly)).astype(np.uint8))

    if combined.getbbox() is None:
        return combined

    if feather_radius > 0:
        combined = combined.filter(ImageFilter.GaussianBlur(radius=feather_radius))
    return combined


def save_cutout(json_path: Path, output_dir: Path, image_dir: Optional[Path], label: str, feather_radius: float) -> bool:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    image_path = locate_image(json_path, data, image_dir)
    image = Image.open(image_path).convert("RGBA")
    alpha = build_alpha_mask(image.size, data.get("shapes", []), label, feather_radius)
    bbox = alpha.getbbox()
    if bbox is None:
        print(f"[WARN] No '{label}' polygon found in {json_path}")
        return False

    cutout = image.copy()
    cutout.putalpha(alpha)
    cutout = cutout.crop(bbox)
    ensure_dir(output_dir)
    out_path = output_dir / f"{json_path.stem}.png"
    cutout.save(out_path)
    print(f"[OK] Saved {out_path}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Labelme polygon JSONs to transparent PNG cutouts.")
    parser.add_argument("--json_dir", required=True, help="Folder containing Labelme JSON files.")
    parser.add_argument("--image_dir", default=None, help="Folder containing original images.")
    parser.add_argument("--output_dir", default="cutouts/burger", help="Output cutout folder.")
    parser.add_argument("--label", default="burger", help="Polygon label to extract.")
    parser.add_argument("--feather_radius", type=float, default=2.0, help="Alpha blur radius for softer edges.")
    args = parser.parse_args()

    json_dir = Path(args.json_dir)
    if not json_dir.exists():
        raise FileNotFoundError(f"JSON folder not found: {json_dir}")

    image_dir = Path(args.image_dir) if args.image_dir else None
    output_dir = ensure_dir(args.output_dir)
    json_files = sorted(json_dir.glob("*.json"))
    if not json_files:
        raise ValueError(f"No .json files found in {json_dir}")

    created = 0
    for json_path in json_files:
        try:
            if save_cutout(json_path, output_dir, image_dir, args.label, args.feather_radius):
                created += 1
        except Exception as exc:
            print(f"[ERROR] {json_path}: {exc}")

    print(f"Done. Created {created} cutouts.")


if __name__ == "__main__":
    main()
