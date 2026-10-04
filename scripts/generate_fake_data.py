import argparse
import random
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter

from utils import clear_dir, ensure_dir, list_images, set_seed


def resize_with_ratio(image: Image.Image, target_w: int) -> Image.Image:
    if image.width <= 0 or image.height <= 0:
        raise ValueError("Invalid image size.")
    ratio = target_w / image.width
    target_h = max(1, int(image.height * ratio))
    return image.resize((target_w, target_h), Image.Resampling.LANCZOS)


def apply_alpha_feather(fg: Image.Image, blur_radius: float) -> Image.Image:
    rgba = fg.convert("RGBA")
    alpha = rgba.getchannel("A")
    alpha = alpha.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    rgba.putalpha(alpha)
    return rgba


def transform_cutout(cutout: Image.Image, max_rotate: float, min_scale: float, max_scale: float, blur_radius: float) -> Image.Image:
    fg = cutout.convert("RGBA")
    scale = random.uniform(min_scale, max_scale)
    target_w = max(12, int(fg.width * scale))
    fg = resize_with_ratio(fg, target_w)
    angle = random.uniform(-max_rotate, max_rotate)
    fg = fg.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
    fg = apply_alpha_feather(fg, blur_radius)
    bright = ImageEnhance.Brightness(fg.convert("RGB")).enhance(random.uniform(0.75, 1.25))
    contrast = ImageEnhance.Contrast(bright).enhance(random.uniform(0.85, 1.20))
    rgba = contrast.convert("RGBA")
    rgba.putalpha(fg.getchannel("A"))
    return rgba


def paste_randomly(background: Image.Image, foreground: Image.Image) -> Image.Image:
    bg = background.convert("RGB")
    fg = foreground.convert("RGBA")
    if fg.width >= bg.width or fg.height >= bg.height:
        scale = min((bg.width - 2) / max(fg.width, 1), (bg.height - 2) / max(fg.height, 1), 0.95)
        scale = max(scale, 0.1)
        fg = fg.resize((max(1, int(fg.width * scale)), max(1, int(fg.height * scale))), Image.Resampling.LANCZOS)
    max_x = max(0, bg.width - fg.width)
    max_y = max(0, bg.height - fg.height)
    x = random.randint(0, max_x)
    y = random.randint(0, max_y)
    bg.paste(fg, (x, y), fg)
    return bg


def generate_fake_burger(cutouts: list[Path], backgrounds: list[Path], image_size: int, max_rotate: float, min_scale: float, max_scale: float, blur_radius: float) -> Image.Image:
    cutout_path = random.choice(cutouts)
    background_path = random.choice(backgrounds)
    cutout = Image.open(cutout_path).convert("RGBA")
    background = Image.open(background_path).convert("RGB").resize((image_size, image_size), Image.Resampling.LANCZOS)
    transformed = transform_cutout(cutout, max_rotate, min_scale, max_scale, blur_radius)
    return paste_randomly(background, transformed)


def generate_epoch_fake_data(
    cutout_dir: str,
    background_dir: str,
    synthetic_dir: str,
    fake_per_epoch: int,
    image_size: int,
    epoch: int,
    seed: int = 42,
    max_rotate: float = 25.0,
    min_scale: float = 0.35,
    max_scale: float = 0.75,
    blur_radius: float = 2.0,
    clear: bool = True,
) -> Path:
    if fake_per_epoch <= 0:
        raise ValueError("fake_per_epoch must be positive")
    if epoch <= 0:
        raise ValueError("epoch must be positive")
    if not (0 < min_scale <= max_scale):
        raise ValueError("Require 0 < min_scale <= max_scale")

    set_seed(seed)
    synthetic_root = Path(synthetic_dir)
    if clear:
        clear_dir(synthetic_root)
    burger_out = ensure_dir(synthetic_root / "burger")
    cutouts = list_images(cutout_dir)
    backgrounds = list_images(background_dir)
    if not cutouts:
        all_files = sorted(Path(cutout_dir).glob("*")) if Path(cutout_dir).exists() else []
        json_count = sum(1 for p in all_files if p.suffix.lower() == ".json")
        if json_count:
            raise ValueError(
                f"No PNG/JPG cutout images found in {cutout_dir}. "
                "This folder contains Labelme JSON files. Run scripts/labelme_to_cutout.py first "
                "and write transparent PNGs to cutouts/burger or pass that PNG folder as --cutout_dir."
            )
        raise ValueError(f"No PNG/JPG cutout images found in {cutout_dir}")
    if not backgrounds:
        raise ValueError(f"No backgrounds found in {background_dir}")

    for i in range(1, fake_per_epoch + 1):
        fake = generate_fake_burger(
            cutouts=cutouts,
            backgrounds=backgrounds,
            image_size=image_size,
            max_rotate=max_rotate,
            min_scale=min_scale,
            max_scale=max_scale,
            blur_radius=blur_radius,
        )
        out_name = f"fake_epoch{epoch:03d}_{i:03d}.jpg"
        fake.save(burger_out / out_name, quality=95)
    return burger_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate one epoch of fake burger data into synthetic/current_epoch/burger.")
    parser.add_argument("--cutout_dir", default="cutouts/burger", help="Folder with transparent burger PNGs.")
    parser.add_argument("--background_dir", default="backgrounds", help="Folder with background images.")
    parser.add_argument("--synthetic_dir", default="synthetic/current_epoch", help="Output synthetic directory.")
    parser.add_argument("--fake_per_epoch", type=int, default=50, help="Number of fake burgers to generate.")
    parser.add_argument("--image_size", type=int, default=224, help="Output image size.")
    parser.add_argument("--epoch", type=int, default=1, help="Epoch number used in filenames.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--max_rotate", type=float, default=25.0, help="Max random rotation in degrees.")
    parser.add_argument("--min_scale", type=float, default=0.35, help="Min scale relative to background.")
    parser.add_argument("--max_scale", type=float, default=0.75, help="Max scale relative to background.")
    parser.add_argument("--blur_radius", type=float, default=2.0, help="Alpha feather radius.")
    parser.add_argument("--clear", action="store_true", help="Clear synthetic dir before generating.")
    args = parser.parse_args()

    out_dir = generate_epoch_fake_data(
        cutout_dir=args.cutout_dir,
        background_dir=args.background_dir,
        synthetic_dir=args.synthetic_dir,
        fake_per_epoch=args.fake_per_epoch,
        image_size=args.image_size,
        epoch=args.epoch,
        seed=args.seed,
        max_rotate=args.max_rotate,
        min_scale=args.min_scale,
        max_scale=args.max_scale,
        blur_radius=args.blur_radius,
        clear=args.clear,
    )
    print(f"Generated {args.fake_per_epoch} fake images at {out_dir}")


if __name__ == "__main__":
    main()
