# CW1 Regression Challenge

This repository contains a full tabular regression workflow for Coursework 1.

The goal is to predict `outcome` from the provided training data and generate a one-column submission file for the held-out test set.

## Project Structure

```text
.
|-- build_submission.py
|-- CW1_eval_script.py
|-- requirements.txt
|-- README.md
|-- data/
|   |-- CW1_train.csv
|   `-- CW1_test.csv
|-- submissions/
|   |-- CW1_submission_K24060083.csv
|   `-- tuning_summary.csv
|-- notebooks/
|   `-- eda.ipynb
`-- report/
    |-- report.tex
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

## How To Run

Run the training + tuning + submission pipeline:

```bash
python build_submission.py
```

## What The Script Does

`build_submission.py` performs the full end-to-end process:

1. Loads data from `data/CW1_train.csv` and `data/CW1_test.csv`.
2. Builds a leakage-safe sklearn `Pipeline`:
   - numeric median imputation
   - categorical most-frequent imputation
   - categorical ordinal encoding
   - `HistGradientBoostingRegressor`
3. Computes baseline 5-fold CV R2.
4. Runs stage-1 broad hyperparameter search (`RandomizedSearchCV`).
5. Runs stage-2 tighter search around stage-1 best parameters.
6. Selects the better stage automatically.
7. Runs a robustness check with a different CV random seed.
8. Fits the selected pipeline on all training rows.
9. Predicts test-set outcomes and saves submission.
10. Saves a tuning summary table.

## Output Files

After each run, the submission file is generated and can be found in:

- `submissions/CW1_submission.csv`
  - Single-column CSV: `yhat`
