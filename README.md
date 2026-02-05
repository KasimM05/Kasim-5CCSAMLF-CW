# CW1 Regression Challenge

This repository contains my solution for Coursework 1, a data science regression challenge.

The objective is to train a regression model to predict the variable `outcome` from a tabular dataset and to generate predictions for a held-out test set. Model performance is evaluated using out-of-sample R².

## Repository Structure

├── train_and_submit.py # Main training and submission script
├── eval_script.py # Provided baseline script (unchanged)
├── data/
│ ├── CW1_train.csv
│ └── CW1_test.csv
├── report/
│ ├── report.tex
│ └── figures/
├── requirements.txt


## How to Run

1. Install dependencies:
```bash
pip install -r requirements.txt
```
2 Generate predictions:

python train_and_submit.py

This will produce a submission file:

CW1_submission_KXXXXXXX.csv

Method Overview

Exploratory data analysis (EDA)

Preprocessing using a unified pipeline:

Median imputation for numerical features

One-hot encoding for categorical features

Model selection via K-fold cross-validation

Hyperparameter tuning

Final training on the full dataset and test prediction generation

The file eval_script.py is included as reference for submission format but is not used directly in the final pipeline.

Author
Kasim Morsel, k24060083
