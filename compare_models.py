import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline

from src.config import RANDOM_STATE
from src.data_io import load_train_test
from src.pipeline_factory import build_preprocessor


def make_pipeline(model) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ]
    )


def main() -> None:
    X, y, _ = load_train_test()

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"r2": "r2", "rmse": "neg_root_mean_squared_error"}

    candidates = {
        "ridge": make_pipeline(Ridge(alpha=1.0)),
        "random_forest": make_pipeline(
            RandomForestRegressor(
                n_estimators=400,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        ),
        "hist_gradient_boosting": make_pipeline(
            HistGradientBoostingRegressor(random_state=RANDOM_STATE)
        ),
    }

    rows = []
    for name, estimator in candidates.items():
        result = cross_validate(
            estimator=estimator,
            X=X,
            y=y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
        )
        rows.append(
            {
                "model": name,
                "cv_r2_mean": float(result["test_r2"].mean()),
                "cv_r2_std": float(result["test_r2"].std()),
                "cv_rmse_mean": float((-result["test_rmse"]).mean()),
                "cv_rmse_std": float((-result["test_rmse"]).std()),
            }
        )

    comparison = (
        pd.DataFrame(rows)
        .sort_values(["cv_r2_mean", "cv_rmse_mean"], ascending=[False, True])
        .reset_index(drop=True)
    )

    out_path = "submissions/model_comparison.csv"
    comparison.to_csv(out_path, index=False)
    print(f"Saved comparison table: {out_path}")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
