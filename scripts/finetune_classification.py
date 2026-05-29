#!/usr/bin/env python3
"""Fine-tune Hugging Face image-classification models on plant disease datasets."""

import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForImageClassification


MODEL_REGISTRY = {
    "swin": "microsoft/swin-base-patch4-window12-384",
    "vit": "google/vit-base-patch16-224",
    "resnet": "microsoft/resnet-50",
    "convnext": "facebook/convnext-base-224-22k-1k",
}

DATASET_ALIASES = {
    "plantvillage": "PlantVillage",
    "fieldplant": "FieldPlant",
    "plantdoc": "PlantDoc",
    "croppedplantdoc": "CroppedPlantDoc",
    "cropped plantdoc": "CroppedPlantDoc",
    "cropped_plantdoc": "CroppedPlantDoc",
}

DEFAULT_DATASETS = ["PlantVillage", "FieldPlant", "PlantDoc", "CroppedPlantDoc"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune image-classification backbones on plant disease datasets."
    )
    parser.add_argument("--model", default="swin", help="One of: swin, vit, resnet, convnext, all")
    parser.add_argument("--dataset", default="PlantVillage", help="Dataset folder name or all")
    parser.add_argument("--data-root", default="datasets", help="Root folder containing image-folder datasets")
    parser.add_argument("--output-dir", default="results", help="Directory for result files")
    parser.add_argument("--epochs", type=int, default=10, help="Number of fine-tuning epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="AdamW learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_dataset_name(name):
    if name.lower() == "all":
        return "all"
    key = name.strip().lower()
    if key in DATASET_ALIASES:
        return DATASET_ALIASES[key]
    return name.replace(" ", "")


def selected_models(model_arg):
    model_arg = model_arg.lower()
    if model_arg == "all":
        return list(MODEL_REGISTRY.keys())
    if model_arg not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model '{model_arg}'. Choose from {list(MODEL_REGISTRY)} or all.")
    return [model_arg]


def selected_datasets(dataset_arg):
    dataset_name = normalize_dataset_name(dataset_arg)
    if dataset_name == "all":
        return DEFAULT_DATASETS
    return [dataset_name]


def validate_dataset(dataset_path):
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    class_dirs = sorted([p for p in dataset_path.iterdir() if p.is_dir()])
    if len(class_dirs) < 2:
        raise ValueError(f"Expected at least two class folders in {dataset_path}")
    return class_dirs


def pil_loader(path):
    with open(path, "rb") as f:
        image = Image.open(f)
        return image.convert("RGB")


def make_collate_fn(processor):
    def collate(batch):
        images, labels = zip(*batch)
        encoded = processor(images=list(images), return_tensors="pt")
        encoded["labels"] = torch.tensor(labels, dtype=torch.long)
        return encoded

    return collate


def split_dataset(dataset, seed):
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, test_size], generator=generator)


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="Training", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(1, len(loader))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    predictions = []
    labels = []
    for batch in tqdm(loader, desc="Evaluating", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(pixel_values=batch["pixel_values"])
        preds = outputs.logits.argmax(dim=1).cpu().numpy()
        predictions.extend(preds)
        labels.extend(batch["labels"].cpu().numpy())
    accuracy = accuracy_score(labels, predictions)
    macro_f1 = f1_score(labels, predictions, average="macro", zero_division=0)
    cm = confusion_matrix(labels, predictions)
    return accuracy, macro_f1, cm


def save_confusion_matrix(cm, class_names, csv_path, png_path):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true/pred"] + class_names)
        for class_name, row in zip(class_names, cm):
            writer.writerow([class_name] + row.tolist())

    size = max(6, len(class_names) * 0.45)
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(png_path, dpi=200)
    plt.close(fig)


def run_experiment(model_key, dataset_name, args, device):
    dataset_path = Path(args.data_root) / dataset_name
    validate_dataset(dataset_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Model: {model_key} | Dataset: {dataset_name} ===")
    print(f"Dataset path: {dataset_path}")

    processor = AutoImageProcessor.from_pretrained(MODEL_REGISTRY[model_key])
    dataset = ImageFolder(dataset_path, loader=pil_loader)
    class_names = dataset.classes
    label2id = {label: idx for idx, label in enumerate(class_names)}
    id2label = {idx: label for label, idx in label2id.items()}
    print(f"Classes ({len(class_names)}): {class_names}")

    train_subset, test_subset = split_dataset(dataset, args.seed)
    collate_fn = make_collate_fn(processor)
    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    model = AutoModelForImageClassification.from_pretrained(
        MODEL_REGISTRY[model_key],
        num_labels=len(class_names),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    history = []
    for epoch in range(args.epochs):
        loss = train_epoch(model, train_loader, optimizer, device)
        accuracy, macro_f1, _ = evaluate(model, test_loader, device)
        history.append({"epoch": epoch + 1, "loss": loss, "accuracy": accuracy, "macro_f1": macro_f1})
        print(
            f"Epoch {epoch + 1}/{args.epochs}: "
            f"loss={loss:.4f}, accuracy={accuracy:.4f}, macro_f1={macro_f1:.4f}"
        )

    accuracy, macro_f1, cm = evaluate(model, test_loader, device)
    safe_dataset = dataset_name.replace(" ", "")
    prefix = f"{model_key}_{safe_dataset}"
    results_path = output_dir / f"{prefix}_results.txt"
    cm_csv_path = output_dir / f"{prefix}_confusion_matrix.csv"
    cm_png_path = output_dir / f"{prefix}_confusion_matrix.png"
    mapping_path = output_dir / f"{prefix}_label_mapping.json"
    config_path = output_dir / f"{prefix}_config.json"

    save_confusion_matrix(cm, class_names, cm_csv_path, cm_png_path)
    mapping_path.write_text(json.dumps({"label2id": label2id, "id2label": id2label}, indent=2), encoding="utf-8")
    config = {
        "model_key": model_key,
        "model_name": MODEL_REGISTRY[model_key],
        "dataset": dataset_name,
        "dataset_path": str(dataset_path),
        "num_classes": len(class_names),
        "train_size": len(train_subset),
        "test_size": len(test_subset),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "device": str(device),
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    with open(results_path, "w", encoding="utf-8") as f:
        f.write(f"Model: {model_key}\n")
        f.write(f"Checkpoint: {MODEL_REGISTRY[model_key]}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Classes: {len(class_names)}\n")
        f.write(f"Train/Test split: {len(train_subset)}/{len(test_subset)}\n")
        f.write(f"Accuracy: {accuracy:.6f}\n")
        f.write(f"Macro F1-score: {macro_f1:.6f}\n")
        f.write(f"Confusion matrix CSV: {cm_csv_path}\n")
        f.write(f"Confusion matrix PNG: {cm_png_path}\n")
        f.write("\nEpoch history:\n")
        for item in history:
            f.write(json.dumps(item) + "\n")

    return {
        "model": model_key,
        "model_name": MODEL_REGISTRY[model_key],
        "dataset": dataset_name,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "num_classes": len(class_names),
        "train_size": len(train_subset),
        "test_size": len(test_subset),
        "results_file": str(results_path),
    }


def write_summary(rows, output_dir):
    summary_path = Path(output_dir) / "benchmark_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "model",
            "model_name",
            "dataset",
            "accuracy",
            "macro_f1",
            "num_classes",
            "train_size",
            "test_size",
            "results_file",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved summary: {summary_path}")


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Selected device: {device}")

    rows = []
    for model_key in selected_models(args.model):
        for dataset_name in selected_datasets(args.dataset):
            rows.append(run_experiment(model_key, dataset_name, args, device))
    write_summary(rows, args.output_dir)


if __name__ == "__main__":
    main()
