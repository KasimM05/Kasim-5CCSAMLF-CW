import pandas as pd

from src.config import HGB_FIXED_PARAMS, SUBMISSION_DIR, SUBMISSION_NAME
from src.data_io import load_train_test
from src.pipeline_factory import make_histgb_pipeline


def main() -> None:
    X, y, X_test = load_train_test()

    pipeline = make_histgb_pipeline(HGB_FIXED_PARAMS)
    pipeline.fit(X, y)

    yhat_test = pipeline.predict(X_test)

    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SUBMISSION_DIR / SUBMISSION_NAME
    pd.DataFrame({"yhat": yhat_test}).to_csv(out_path, index=False)
    print(f"Saved submission: {out_path}")


if __name__ == "__main__":
    main()
