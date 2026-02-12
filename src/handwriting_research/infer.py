from __future__ import annotations

import argparse
import json

import torch
from PIL import Image
from torchvision import transforms

from .features import HandwritingFeatureExtractor
from .model import IntegratedHandwritingModel, ModelConfig


def invert_map(class_map):
    return {idx: label for label, idx in class_map.items()}


def predict(model_path: str, image_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)

    model_cfg = ModelConfig(**checkpoint["model_config"])
    model = IntegratedHandwritingModel(model_cfg).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([transforms.Resize((128, 512)), transforms.ToTensor()])
    x = transform(image).unsqueeze(0).to(device)

    extractor = HandwritingFeatureExtractor()
    handcrafted_values = list(extractor.extract(image).values())
    handcrafted = torch.tensor(handcrafted_values, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x, handcrafted)

    label_maps = checkpoint["label_maps"]
    inv_maps = {task: invert_map(label_maps[task]) for task in label_maps}

    out = {}
    for task, task_logits in logits.items():
        prob = torch.softmax(task_logits, dim=1)
        idx = prob.argmax(dim=1).item()
        out[task] = {
            "label": inv_maps[task][idx],
            "confidence": float(prob[0, idx].item()),
        }

    return out


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference for handwriting psychology model")
    parser.add_argument("--model", required=True, help="Path to trained model checkpoint")
    parser.add_argument("--image", required=True, help="Path to handwriting image")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prediction = predict(args.model, args.image)
    print(json.dumps(prediction, indent=2))
