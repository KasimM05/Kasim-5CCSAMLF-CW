# CW1 Regression Challenge

This repository contains a tabular regression workflow for Coursework 1.

The goal is to predict `outcome` from the provided training data and generate a one-column submission file for the held-out test set.

## Project Structure

```text
.
|-- train_and_submit.py
|-- CW1_eval_script.py
|-- requirements.txt
|-- README.md
|-- data/
|   |-- CW1_train.csv
|   `-- CW1_test.csv
|-- submissions/
|   `-- CW1_submission_K24060083.csv
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

Run the training, tuning, and submission pipeline:

```bash
python train_and_submit.py
```

## What The Script Does

`train_and_submit.py` performs the full end-to-end process:

1. Loads data from `data/CW1_train.csv` and `data/CW1_test.csv`.
2. Builds a leakage-safe sklearn pipeline:
   - numeric median imputation
   - categorical most-frequent imputation
   - categorical ordinal encoding
   - `HistGradientBoostingRegressor`
3. Computes baseline 5-fold CV R2.
4. Runs stage-1 broad hyperparameter search (`RandomizedSearchCV`).
5. Runs stage-2 tighter search around stage-1 best parameters.
6. Selects the better stage automatically.
7. Fits the selected pipeline on all training rows.
8. Predicts test-set outcomes and saves submission.

## Output File

After each run:

- `submissions/CW1_submission_K24060083.csv`
  - Single-column CSV: `yhat`
  - Overwritten on each run

## Notes On Evaluation

- The provided test file does not include true outcomes.
- Because of that, local performance is estimated using cross-validation R2 on training data.
- Final performance is determined externally when the submission is evaluated with hidden outcomes.
