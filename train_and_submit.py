import pandas as pd
from math import prod
from pathlib import Path

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


RANDOM_STATE = 42
N_SPLITS = 5
SCORING = "r2"

# Locked params from the best run so far (used when tuning is skipped).
LOCKED_FINAL_PARAMS = {
    "learning_rate": 0.039,
    "max_iter": 300,
    "max_depth": 3,
    "max_leaf_nodes": 255,
    "min_samples_leaf": 10,
    "l2_regularization": 1.0,
}

# Keep enabled while iterating; set to False for fast submission generation.
ENABLE_TUNING = True


def resolve_data_path(filename: str) -> Path:
    candidates = [Path("data") / filename, Path(filename)]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find {filename}. Checked: {candidates}")


def build_ordinal_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ordinal",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, make_column_selector(dtype_include=["number"])),
            (
                "cat",
                categorical_pipeline,
                make_column_selector(dtype_include=["object", "category", "bool"]),
            ),
        ]
    )


def unique_sorted(values):
    return sorted(set(values))


def make_tighter_distributions(best_params: dict) -> dict:
    lr = float(best_params["model__learning_rate"])
    iters = int(best_params["model__max_iter"])
    depth = best_params["model__max_depth"]
    leaves = int(best_params["model__max_leaf_nodes"])
    min_leaf = int(best_params["model__min_samples_leaf"])
    l2 = float(best_params["model__l2_regularization"])

    lr_choices = unique_sorted(
        [
            max(0.005, round(lr * 0.7, 4)),
            round(lr, 4),
            min(0.3, round(lr * 1.3, 4)),
        ]
    )

    iter_choices = unique_sorted(
        [
            max(100, int(round(iters * 0.75))),
            iters,
            min(2000, int(round(iters * 1.25))),
        ]
    )

    if depth is None:
        depth_choices = [None, 3, 4, 5]
    else:
        d = int(depth)
        depth_choices = unique_sorted([max(2, d - 1), d, d + 1])

    leaf_choices = unique_sorted(
        [
            max(15, int(round(leaves * 0.7))),
            leaves,
            min(255, int(round(leaves * 1.3))),
        ]
    )

    min_leaf_choices = unique_sorted(
        [
            max(5, int(round(min_leaf * 0.7))),
            min_leaf,
            min(200, int(round(min_leaf * 1.4))),
        ]
    )

    l2_choices = unique_sorted(
        [
            max(0.0, round(l2 * 0.1, 6)),
            round(l2, 6),
            round(max(1e-6, l2 * 10.0), 6),
        ]
    )

    return {
        "model__learning_rate": lr_choices,
        "model__max_iter": iter_choices,
        "model__max_depth": depth_choices,
        "model__max_leaf_nodes": leaf_choices,
        "model__min_samples_leaf": min_leaf_choices,
        "model__l2_regularization": l2_choices,
    }


def make_pipeline_with_model(model: HistGradientBoostingRegressor) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_ordinal_preprocessor()),
            ("model", model),
        ]
    )


def eval_cv(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, cv: KFold) -> tuple[float, float, list[float]]:
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring=SCORING)
    return float(scores.mean()), float(scores.std()), [float(s) for s in scores]


def main() -> None:
    train_path = resolve_data_path("CW1_train.csv")
    test_path = resolve_data_path("CW1_test.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    target_col = "outcome"
    if target_col not in train_df.columns:
        raise ValueError(f"Training data must contain target column: {target_col}")

    X = train_df.drop(columns=[target_col])
    y = train_df[target_col]

    missing_in_test = [c for c in X.columns if c not in test_df.columns]
    if missing_in_test:
        raise ValueError(f"Test data is missing required columns: {missing_in_test}")

    X_test = test_df[X.columns].copy()

    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    baseline_pipeline = make_pipeline_with_model(
        HistGradientBoostingRegressor(random_state=RANDOM_STATE)
    )
    baseline_mean, baseline_std, baseline_scores = eval_cv(baseline_pipeline, X, y, cv)
    print("Baseline HistGradientBoosting")
    print("Fold R^2:", [round(s, 5) for s in baseline_scores])
    print(f"Mean R^2: {baseline_mean:.5f} (+/- {baseline_std:.5f})")
    print("-" * 40)

    results_rows = [
        {
            "stage": "baseline",
            "cv_mean_r2": baseline_mean,
            "cv_std_r2": baseline_std,
            "params": "default",
        }
    ]

    selected_pipeline = None
    selected_mean = None
    selected_stage = None

    if ENABLE_TUNING:
        stage1_distributions = {
            "model__learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1, 0.15],
            "model__max_iter": [200, 300, 500, 800, 1200],
            "model__max_depth": [None, 3, 5, 8, 12],
            "model__max_leaf_nodes": [15, 31, 63, 127, 255],
            "model__min_samples_leaf": [10, 20, 30, 50, 80],
            "model__l2_regularization": [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0],
        }

        stage1 = RandomizedSearchCV(
            estimator=baseline_pipeline,
            param_distributions=stage1_distributions,
            n_iter=35,
            scoring=SCORING,
            cv=cv,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            refit=True,
            verbose=1,
        )
        stage1.fit(X, y)

        print("Stage 1 best params:")
        print(stage1.best_params_)
        print(f"Stage 1 best CV Mean R^2: {stage1.best_score_:.5f}")
        print("-" * 40)

        results_rows.append(
            {
                "stage": "stage1",
                "cv_mean_r2": float(stage1.best_score_),
                "cv_std_r2": None,
                "params": str(stage1.best_params_),
            }
        )

        stage2_distributions = make_tighter_distributions(stage1.best_params_)
        total_stage2_combos = prod(len(v) for v in stage2_distributions.values())
        stage2_n_iter = min(24, total_stage2_combos)

        stage2 = RandomizedSearchCV(
            estimator=baseline_pipeline,
            param_distributions=stage2_distributions,
            n_iter=stage2_n_iter,
            scoring=SCORING,
            cv=cv,
            n_jobs=-1,
            random_state=RANDOM_STATE + 1,
            refit=True,
            verbose=1,
        )
        stage2.fit(X, y)

        print("Stage 2 best params:")
        print(stage2.best_params_)
        print(f"Stage 2 best CV Mean R^2: {stage2.best_score_:.5f}")
        print("-" * 40)

        results_rows.append(
            {
                "stage": "stage2",
                "cv_mean_r2": float(stage2.best_score_),
                "cv_std_r2": None,
                "params": str(stage2.best_params_),
            }
        )

        if stage2.best_score_ > stage1.best_score_:
            selected_search = stage2
            selected_stage = "stage2"
        else:
            selected_search = stage1
            selected_stage = "stage1"

        selected_pipeline = selected_search.best_estimator_
        selected_mean = float(selected_search.best_score_)

    else:
        print("Tuning is disabled. Using locked final parameters.")
        final_model = HistGradientBoostingRegressor(
            random_state=RANDOM_STATE, **LOCKED_FINAL_PARAMS
        )
        selected_pipeline = make_pipeline_with_model(final_model)
        selected_mean, selected_std, _ = eval_cv(selected_pipeline, X, y, cv)
        selected_stage = "locked_final"

        results_rows.append(
            {
                "stage": selected_stage,
                "cv_mean_r2": selected_mean,
                "cv_std_r2": selected_std,
                "params": str(LOCKED_FINAL_PARAMS),
            }
        )

    print(f"Selected model from: {selected_stage}")
    print(f"Selected CV Mean R^2: {selected_mean:.5f}")

    # Robustness check with a different CV split seed.
    robust_cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=7)
    robust_mean, robust_std, robust_scores = eval_cv(selected_pipeline, X, y, robust_cv)
    print("Robustness check (new CV split seed = 7)")
    print("Fold R^2:", [round(s, 5) for s in robust_scores])
    print(f"Mean R^2: {robust_mean:.5f} (+/- {robust_std:.5f})")
    print("-" * 40)

    results_rows.append(
        {
            "stage": "robustness_check_seed7",
            "cv_mean_r2": robust_mean,
            "cv_std_r2": robust_std,
            "params": "selected_pipeline",
        }
    )

    selected_pipeline.fit(X, y)
    yhat_test = selected_pipeline.predict(X_test)

    output_dir = Path("submissions")
    output_dir.mkdir(parents=True, exist_ok=True)

    submission = pd.DataFrame({"yhat": yhat_test})
    out_path = output_dir / "CW1_submission_K24060083.csv"
    submission.to_csv(out_path, index=False)
    print(f"Saved submission: {out_path}")

    summary_path = output_dir / "tuning_summary.csv"
    pd.DataFrame(results_rows).to_csv(summary_path, index=False)
    print(f"Saved tuning summary: {summary_path}")


if __name__ == "__main__":
    main()
