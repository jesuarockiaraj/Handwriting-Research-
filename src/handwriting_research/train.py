from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch
from sklearn.metrics import accuracy_score
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split

from .data import HandwritingDataset
from .model import IntegratedHandwritingModel, ModelConfig


@dataclass
class TrainConfig:
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 10
    val_split: float = 0.2
    seed: int = 42


def multitask_loss(outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], criterion: nn.Module):
    return (
        criterion(outputs["emotion"], batch["emotion"])
        + criterion(outputs["personality"], batch["personality"])
        + criterion(outputs["state"], batch["state"])
    ) / 3.0


def evaluate(model, dataloader, device):
    model.eval()
    all_preds = {"emotion": [], "personality": [], "state": []}
    all_targets = {"emotion": [], "personality": [], "state": []}

    with torch.no_grad():
        for batch in dataloader:
            image = batch["image"].to(device)
            handcrafted = batch["handcrafted"].to(device)
            outputs = model(image, handcrafted)

            for task in all_preds:
                preds = outputs[task].argmax(dim=1).cpu().tolist()
                targets = batch[task].tolist()
                all_preds[task].extend(preds)
                all_targets[task].extend(targets)

    return {task: accuracy_score(all_targets[task], all_preds[task]) for task in all_preds}


def train(csv_path: str, image_root: str, output_model: str, cfg: TrainConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)

    dataset = HandwritingDataset(csv_path=csv_path, image_root=image_root)
    val_size = int(len(dataset) * cfg.val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)

    model_cfg = ModelConfig(
        emotion_classes=len(dataset.label_maps["emotion"]),
        personality_classes=len(dataset.label_maps["personality"]),
        state_classes=len(dataset.label_maps["state"]),
    )
    model = IntegratedHandwritingModel(model_cfg).to(device)

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_score = 0.0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            image = batch["image"].to(device)
            handcrafted = batch["handcrafted"].to(device)
            labels = {k: batch[k].to(device) for k in ["emotion", "personality", "state"]}

            outputs = model(image, handcrafted)
            loss = multitask_loss(outputs, labels, criterion)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scores = evaluate(model, val_loader, device)
        avg_score = sum(scores.values()) / len(scores)
        print(
            f"Epoch {epoch}/{cfg.epochs} | loss={running_loss / max(len(train_loader), 1):.4f} "
            f"| emotion={scores['emotion']:.3f} personality={scores['personality']:.3f} state={scores['state']:.3f}"
        )

        if avg_score > best_score:
            best_score = avg_score
            Path(output_model).parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": model_cfg.__dict__,
                    "label_maps": dataset.label_maps,
                },
                output_model,
            )

    print(f"Saved best model to {output_model} with avg validation accuracy={best_score:.3f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train integrated handwriting analysis model")
    parser.add_argument("--csv", required=True, help="Path to metadata CSV")
    parser.add_argument("--image-root", default=".", help="Root directory for images")
    parser.add_argument("--output-model", default="artifacts/handwriting_model.pt")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = TrainConfig(batch_size=args.batch_size, lr=args.lr, epochs=args.epochs)
    train(args.csv, args.image_root, args.output_model, cfg)
