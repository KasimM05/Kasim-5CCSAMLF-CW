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
|-- .gitignore
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

Python version: `3.11` (tested with `3.11.9`).

1. Clone the repo, then create and activate a virtual environment (recommended).

```bash
git clone https://github.com/KasimM05/Kasim-5CCSAMLF-CW.git
cd Kasim-5CCSAMLF-CW
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies.

```bash
python -m pip install --upgrade pip
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

## Run model comparison

Use this to generate a cross-validated comparison table across multiple model families.

```bash
python compare_models.py
```

Output:

- `submissions/model_comparison.csv`
- columns include mean/std for 5-fold CV `R2` and `RMSE`

(Run commands from the project root)
    
