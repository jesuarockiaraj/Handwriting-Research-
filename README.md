# Handwriting-Research-

Integrated source code for computational handwriting-based psychological marker analysis.

## What is implemented

This repository now contains a full baseline framework that follows the abstract's design:

- **Hybrid deep architecture** with a CNN visual encoder, BiGRU sequence modeling, and multi-head attention.
- **Multi-task prediction heads** for:
  - emotional valence/category,
  - personality class,
  - psychological state class.
- **Global + local feature extraction** module for interpretable handcrafted descriptors (ink density, contrast, slant proxy, line variability, edge density, and intensity histogram).
- **Training pipeline** with validation and checkpoint saving.
- **Inference pipeline** that returns class labels and confidence for each psychological target.

## Project structure

- `src/handwriting_research/model.py` – integrated CNN + RNN + attention model.
- `src/handwriting_research/features.py` – handcrafted feature extraction protocol.
- `src/handwriting_research/data.py` – CSV-based dataset loader for multitask labels.
- `src/handwriting_research/train.py` – end-to-end training script.
- `src/handwriting_research/infer.py` – single-image prediction script.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data format

Use a CSV file with the following columns:

- `image_path` (path relative to `--image-root`)
- `emotion`
- `personality`
- `state`

Example:

```csv
image_path,emotion,personality,state
samples/img_001.png,positive,stable,calm
samples/img_002.png,negative,neurotic,stressed
```

## Training

```bash
PYTHONPATH=src python -m handwriting_research.train \
  --csv data/labels.csv \
  --image-root . \
  --output-model artifacts/handwriting_model.pt \
  --epochs 20 \
  --batch-size 16
```

## Inference

```bash
PYTHONPATH=src python -m handwriting_research.infer \
  --model artifacts/handwriting_model.pt \
  --image data/example.png
```

The output is JSON with per-task labels and confidence scores.
