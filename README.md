# CW1 Regression Challenge

This repository contains a full tabular regression workflow for Coursework 1.

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
python train_and_submit.py
```

## What The Script Does

`train_and_submit.py` performs the full end-to-end process:

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

After each run, files are written to `submissions/`:

- `submissions/CW1_submission_K24060083.csv`
  - Single-column CSV: `yhat`
  - Overwritten on each run

- `submissions/tuning_summary.csv`
  - Summary of baseline, stage-1, stage-2, and robustness-check CV metrics
  - Overwritten on each run

## Notes On Evaluation

- The provided test file does not include true outcomes.
- Because of that, local "accuracy" is estimated using cross-validation R2 on training data.
- Final leaderboard/performance is determined externally when submission is evaluated with hidden outcomes.

## Quick Troubleshooting

- If you get `FileNotFoundError`, confirm both files exist:
  - `data/CW1_train.csv`
  - `data/CW1_test.csv`
- If training is slow, set `ENABLE_TUNING = False` in `train_and_submit.py` to skip search and use locked final parameters.
