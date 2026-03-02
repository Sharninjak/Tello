import argparse
from pathlib import Path

import cv2
import yaml


COLORS = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def normalize_split(split: str) -> str:
    mapping = {
        "valid": "val",
        "validation": "val",
        "test": "test",
        "train": "train",
        "val": "val",
    }
    key = split.strip().lower()
    if key not in mapping:
        raise ValueError(f"不支持的 split: {split}，可选 train/val/valid/test")
    return mapping[key]


def get_label_dir(images_dir: Path) -> Path:
    if images_dir.name == "images":
        return images_dir.parent / "labels"
    return Path(str(images_dir).replace("images", "labels"))


def resolve_images_dir(data_yaml: Path, split_value: str) -> Path:
    raw = split_value.replace("\\", "/").strip()
    candidates = []

    candidates.append((data_yaml.parent / raw).resolve())

    while raw.startswith("../"):
        raw = raw[3:]
    if raw:
        candidates.append((data_yaml.parent / raw).resolve())

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


def to_class_name_list(names):
    if isinstance(names, list):
        return names
    if isinstance(names, dict):
        keys = sorted(names.keys(), key=lambda x: int(x))
        return [names[k] for k in keys]
    raise ValueError("data.yaml 中 names 格式不正确，应为 list 或 dict")


def draw_boxes(img_path: Path, label_path: Path, output_path: Path, class_names: list[str]) -> None:
    img = cv2.imread(str(img_path))
    if img is None:
        return

    h, w = img.shape[:2]

    if label_path.exists():
        lines = label_path.read_text(encoding="utf-8").splitlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            try:
                class_id = int(float(parts[0]))
                x_c, y_c, bw, bh = map(float, parts[1:5])
            except ValueError:
                continue

            x1 = int((x_c - bw / 2) * w)
            y1 = int((y_c - bh / 2) * h)
            x2 = int((x_c + bw / 2) * w)
            y2 = int((y_c + bh / 2) * h)

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            label_name = class_names[class_id] if 0 <= class_id < len(class_names) else str(class_id)
            color = COLORS[class_id % len(COLORS)]

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            text_size = cv2.getTextSize(label_name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            text_top = max(0, y1 - text_size[1] - 6)
            cv2.rectangle(img, (x1, text_top), (x1 + text_size[0] + 4, y1), color, -1)
            cv2.putText(img, label_name, (x1 + 2, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imwrite(str(output_path), img)


def main() -> None:
    parser = argparse.ArgumentParser(description="可视化 YOLO 数据集的图片与标注框")
    parser.add_argument("--split", default="val", help="train/val/valid/test")
    parser.add_argument("--max-images", type=int, default=20)
    parser.add_argument("--data", default="img/cylinder.v1i.yolov8/data.yaml", help="data.yaml 路径")
    args = parser.parse_args()

    data_yaml = Path(args.data).resolve()
    if not data_yaml.exists():
        raise FileNotFoundError(f"未找到 data.yaml: {data_yaml}")

    data_config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    split_key = normalize_split(args.split)
    if split_key not in data_config:
        raise KeyError(f"data.yaml 中没有 '{split_key}' 字段")

    images_dir = resolve_images_dir(data_yaml, str(data_config[split_key]))
    labels_dir = get_label_dir(images_dir)
    output_dir = data_yaml.parent / f"{split_key}_visualized"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not images_dir.exists():
        raise FileNotFoundError(f"图片目录不存在: {images_dir}")

    class_names = to_class_name_list(data_config.get("names", []))
    image_files = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS and p.is_file()])

    if not image_files:
        raise RuntimeError(f"未在目录中找到图片: {images_dir}")

    total = min(args.max_images, len(image_files))
    for idx, img_path in enumerate(image_files[:total], start=1):
        label_path = labels_dir / f"{img_path.stem}.txt"
        output_path = output_dir / img_path.name
        draw_boxes(img_path, label_path, output_path, class_names)
        print(f"[{idx}/{total}] 已处理: {img_path.name}")

    print(f"完成！输出目录: {output_dir}")


if __name__ == "__main__":
    main()