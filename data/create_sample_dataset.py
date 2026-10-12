"""Generate a sample breast cancer CSV from sklearn datasets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def main() -> None:
    output_path = Path(__file__).resolve().parent / "breast_cancer.csv"
    try:
        from sklearn.datasets import load_breast_cancer

        dataset = load_breast_cancer(as_frame=True)
        dataframe = dataset.frame.copy()
    except Exception as exc:  # pragma: no cover - environment fallback
        raise RuntimeError(
            "scikit-learn is required to generate breast_cancer.csv. "
            "Install dependencies from requirements.txt and re-run this script."
        ) from exc

    dataframe.to_csv(output_path, index=False)
    print(f"Saved dataset to {output_path}")


if __name__ == "__main__":
    main()
