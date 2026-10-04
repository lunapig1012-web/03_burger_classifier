import argparse
from pathlib import Path

from utils import IMAGE_EXTENSIONS


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch rename images to a beginner-friendly source ID format.")
    parser.add_argument("--folder", required=True, help="Image folder to rename, for example data/train/burger.")
    parser.add_argument("--prefix", required=True, help="Name prefix, for example burger or non_burger.")
    parser.add_argument("--start", type=int, default=1, help="Start number. Default: 1.")
    parser.add_argument("--digits", type=int, default=3, help="Number padding digits. Default: 3.")
    parser.add_argument("--tag", default="raw", help="Text after double underscore. Default: raw.")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without renaming files.")
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    images = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise ValueError(f"No image files found in {folder}")

    plans = []
    for index, image_path in enumerate(images, start=args.start):
        new_name = f"{args.prefix}{index:0{args.digits}d}__{args.tag}{image_path.suffix.lower()}"
        plans.append((image_path, image_path.with_name(new_name)))

    conflicts = [new for old, new in plans if new.exists() and old.name != new.name]
    if conflicts:
        conflict_text = "\n".join(str(p) for p in conflicts[:10])
        raise FileExistsError(f"Some target filenames already exist:\n{conflict_text}")

    for old, new in plans:
        print(f"{old.name} -> {new.name}")

    if args.dry_run:
        print("Dry run only. No files were renamed.")
        return

    temp_plans = []
    for old, new in plans:
        temp = old.with_name(f"__tmp_rename__{old.name}")
        old.rename(temp)
        temp_plans.append((temp, new))

    for temp, new in temp_plans:
        temp.rename(new)

    print(f"Renamed {len(plans)} file(s) in {folder}")


if __name__ == "__main__":
    main()
