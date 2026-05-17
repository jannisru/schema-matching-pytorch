# schema-matching-pytorch

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)

A neural schema matching system that automatically identifies corresponding columns across different database schemas — using a Siamese sentence-transformer network trained on synthetic table pairs.

## Problem

Given two tables with different column naming conventions, predict which columns represent the same semantic concept:

| Table A | Table B | Match? |
|---|---|---|
| `salary` | `income` | ✓ |
| `hire_date` | `start_date` | ✓ |
| `department` | `total_amount` | ✗ |

## How it works

Each column is represented as a text string combining its name, data type, null rate, unique ratio, and sample values:

```
column: salary | dtype: float64 | null_rate: 0.00 | unique_ratio: 1.00 | min: 8500 | max: 62000 | mean: 35125 | values: 60000, 45000, ...
```

A Siamese-style neural network encodes both columns with a pre-trained sentence transformer, then classifies the pair:

```
column_a → SentenceTransformer → emb_a ─┐
                                          ├─ concat → Linear(768,256) → ReLU → Dropout
column_b → SentenceTransformer → emb_b ─┘          → Linear(256,128) → ReLU → Dropout
                                                     → Linear(128,1)  → logit → match probability
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# Train and evaluate
python main.py

# Custom config
python main.py --config path/to/config.yaml

# Skip training, load saved model
python main.py --load-model
```

## Configuration

All hyperparameters in `config.yaml`:

```yaml
model:
  encoder: all-MiniLM-L6-v2   # any SentenceTransformer model
  dropout: 0.3

training:
  epochs: 10
  lr: 0.001
  batch_size: 8
  patience: 3                  # early stopping

data:
  val_split: 0.2
  random_seed: 42
```

## Dataset

10 synthetic table pairs across business domains (Customers, Orders, Products, Employees, Payments, Shipments, Suppliers, Reviews, Invoices, Contracts). 250 labeled column pairs — 50 positive matches, 200 negatives.

## Training details

- Loss: `BCEWithLogitsLoss` with `pos_weight` computed from class ratio (4.0 for default dataset)
- Optimizer: Adam with `ReduceLROnPlateau` scheduler
- Early stopping: restores best checkpoint when val loss stagnates
- Device: auto-selects CUDA → MPS → CPU

## Evaluation output

After training, `main.py` reports:

- Best classification threshold (F1-optimized on validation set)
- Accuracy, F1, PR-AUC for train and validation sets
- Precision-Recall curve saved to `pr_curve.png`
- False positives and false negatives (named column pairs)
- Confusion matrix (TN / FP / FN / TP)

## Project structure

```
config.yaml          all hyperparameters
main.py              entry point: train + evaluate
src/
  preprocessing.py   column_to_text(): feature extraction per column
  dataset.py         ColumnMatchingDataset, build_full_dataset()
  model.py           ColumnMatcher (SentenceTransformer + MLP)
  train.py           training loop with early stopping
  evaluate.py        metrics, threshold optimization, PR curve
  utils.py           string similarity baseline
data/
  raw/               synthetic table pairs
  labels/            labeled column pairs (0/1)
```

## Stack

Python · PyTorch · sentence-transformers · scikit-learn
