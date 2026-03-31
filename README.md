# Schema Matching with PyTorch

A neural schema matching system that automatically identifies corresponding columns across different database schemas. Given two tables with different column naming conventions, the model predicts which columns represent the same semantic concept.

## How it works

Each column is represented as a text string combining its name, data type, null rate, unique ratio, and sample values:

```
column: salary | dtype: float64 | null_rate: 0.00 | unique_ratio: 1.00 | min: 8500.00 | max: 62000.00 | mean: 35125.00 | values: 60000, 45000, ...
```

A Siamese-style neural network encodes both columns using a pre-trained sentence transformer (`all-MiniLM-L6-v2`), concatenates the embeddings, and passes them through a classifier to predict match probability.

```
column_a → SentenceTransformer → emb_a ─┐
                                          ├─ cat → Linear(768,256) → ReLU → Dropout
column_b → SentenceTransformer → emb_b ─┘          → Linear(256,128) → ReLU → Dropout
                                                        → Linear(128,1) → logit
```

## Project structure

```
schema-matching-pytorch/
├── config.yaml                  # All hyperparameters
├── main.py                      # Entry point: train + evaluate
├── data/
│   ├── raw/                     # Synthetic table pairs (*_raw_a.csv, *_raw_b.csv)
│   └── labels/
│       └── column_matches.csv   # Labeled column pairs (label 0/1)
└── src/
    ├── preprocessing.py         # column_to_text(): feature extraction per column
    ├── dataset.py               # ColumnMatchingDataset, build_full_dataset()
    ├── model.py                 # ColumnMatcher (SentenceTransformer + MLP)
    ├── train.py                 # train_model(): training loop with early stopping
    ├── evaluate.py              # Metrics, threshold optimization, PR curve
    └── utils.py                 # String similarity baseline
```

## Dataset

Ten synthetic table pairs across different business domains:

| Domain | Table A | Table B |
|---|---|---|
| Customers | `customer_id`, `full_name`, `email`, `signup_date`, `city` | `client_id`, `name`, `email_address`, `registration_date`, `location` |
| Orders | `order_id`, `customer_id`, `order_date`, `amount`, `status` | `purchase_id`, `client_id`, `purchase_date`, `total_amount`, `state` |
| Products | `product_id`, `product_name`, `price`, `category`, `stock` | `item_id`, `name`, `cost`, `type`, `inventory` |
| Employees | `employee_id`, `full_name`, `department`, `salary`, `hire_date` | `staff_id`, `name`, `division`, `income`, `start_date` |
| Payments | `payment_id`, `order_id`, `payment_date`, `amount`, `method` | `transaction_id`, `purchase_id`, `date`, `total`, `channel` |
| Shipments | `shipment_id`, `order_id`, `ship_date`, `delivery_date`, `carrier` | `delivery_id`, `purchase_id`, `dispatch_date`, `arrival_date`, `service` |
| Suppliers | `supplier_id`, `company_name`, `contact_email`, `country`, `phone_number` | `vendor_id`, `org_name`, `email`, `region`, `phone` |
| Reviews | `review_id`, `product_id`, `rating`, `review_text`, `review_date` | `feedback_id`, `item_id`, `score`, `comment`, `created_at` |
| Invoices | `invoice_id`, `customer_id`, `issue_date`, `due_date`, `total_amount` | `bill_id`, `client_id`, `created_date`, `deadline`, `amount_due` |
| Contracts | `contract_id`, `client_name`, `start_date`, `end_date`, `value` | `agreement_id`, `party_name`, `valid_from`, `valid_until`, `contract_value` |

250 labeled pairs in total: 50 positive matches, 200 negative (non-matching) pairs.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

**Train and evaluate:**
```bash
python main.py
```

**Use a custom config:**
```bash
python main.py --config path/to/config.yaml
```

**Skip training, load a saved model:**
```bash
python main.py --load-model
```

## Configuration

All hyperparameters live in `config.yaml`:

```yaml
data:
  dir: data/raw
  labels: data/labels/column_matches.csv
  val_split: 0.2
  random_seed: 42

model:
  encoder: all-MiniLM-L6-v2   # any SentenceTransformer model
  dropout: 0.3

training:
  epochs: 10
  lr: 0.001
  batch_size: 8
  patience: 3                  # early stopping patience

output:
  model_path: model.pt
  pr_curve: pr_curve.png
```

## Training details

- **Loss:** `BCEWithLogitsLoss` with `pos_weight` computed from the class ratio (4.0 for the default dataset)
- **Optimizer:** Adam with `ReduceLROnPlateau` scheduler (factor=0.5, patience=2)
- **Early stopping:** Stops training when validation loss does not improve for `patience` epochs; restores the best checkpoint
- **Device:** Automatically selects CUDA, then MPS (Apple Silicon), then CPU

## Evaluation output

After training, `main.py` reports:

- Best classification threshold (optimized for F1 on the validation set)
- Accuracy, F1, and PR-AUC for train and validation sets
- Precision-Recall curve saved to `pr_curve.png`
- Detailed list of false positives and false negatives (named column pairs)
- Confusion matrix summary (TN / FP / FN / TP)
