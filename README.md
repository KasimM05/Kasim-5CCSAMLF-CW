# CW1 Regression Challenge

This repository contains a tabular regression workflow for Coursework 1.

The goal is to predict `outcome` from the provided training data and generate a one-column submission file for the held-out test set.

## Project Structure

```text
.
|-- compare_models.py
|-- build_submission.py
|-- requirements.txt
|-- README.md
|-- src/
|   |-- __init__.py
|   |-- config.py
|   |-- data_io.py
|   `-- pipeline_factory.py
|-- data/
|   |-- CW1_train.csv
|   `-- CW1_test.csv
|-- notebooks/
|   `-- eda.ipynb
|-- submissions/
`-- report/
    |-- main.tex
    `-- figures/
```

## Environment Setup

1. Create and activate a virtual environment (recommended).

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

## Run the script to generate submission: 

This is the CW submission path.

```bash
python build_submission.py
```

Output:

- `submissions/CW1_submission_K24060083.csv`
- single column: `yhat`

## Run model comparison (for report evidence)

Use this to generate a cross-validated comparison table across multiple model families.

```bash
python compare_models.py
```

Output:

- `submissions/model_comparison.csv`
- columns include mean/std for 5-fold CV `R2` and `RMSE`

(Run commands from the project root)
    
