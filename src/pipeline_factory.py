from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from src.config import RANDOM_STATE


def build_preprocessor() -> ColumnTransformer:
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


def make_histgb_pipeline(model_params: dict | None = None) -> Pipeline:
    model_kwargs = {"random_state": RANDOM_STATE}
    if model_params:
        model_kwargs.update(model_params)

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", HistGradientBoostingRegressor(**model_kwargs)),
        ]
    )
