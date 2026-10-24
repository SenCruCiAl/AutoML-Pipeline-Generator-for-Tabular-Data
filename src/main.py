"""CLI entrypoint for AutoML Pipeline Generator for Tabular Data."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from data_loader import DataLoader  # noqa: E402
from evaluator import ModelEvaluator  # noqa: E402
from model_generator import get_model_configs  # noqa: E402
from utils import ensure_directories, setup_logging  # noqa: E402

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="AutoML Pipeline Generator for Tabular Data")
    parser.add_argument("--data", required=True, help="Path to CSV dataset")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument("--test-size", type=float, default=0.2, help="Train-test split ratio")
    parser.add_argument("--cv", type=int, default=5, help="Cross-validation folds")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument("--results-dir", default="results", help="Directory to store result artifacts")
    parser.add_argument("--models-dir", default="models", help="Directory to store trained models")
    return parser.parse_args()


def save_comparison_plot(results_df, output_path: Path) -> None:
    """Save a CV-vs-test F1 comparison chart (CV is what selection used)."""
    x = range(len(results_df))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([i - 0.2 for i in x], results_df["cv_mean_f1"], width=0.4, yerr=results_df["cv_std_f1"],
           color="0.35", label="CV F1 (train split, used for selection)")
    ax.bar([i + 0.2 for i in x], results_df["f1_score"], width=0.4, color="#c0562a", label="Test F1 (held out)")
    ax.set_xticks(list(x), results_df["model"], rotation=20)
    ax.set_ylabel("F1 score")
    lo = min(results_df["cv_mean_f1"].min(), results_df["f1_score"].min())
    ax.set_ylim(max(0.0, lo - 0.05), 1.0)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def main() -> None:
    """Run complete AutoML workflow."""
    args = parse_args()
    setup_logging(args.log_level)

    ensure_directories(args.results_dir, args.models_dir)

    data_loader = DataLoader(
        data_path=args.data,
        target_column=args.target,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    dataset = data_loader.prepare_data()
    model_configs = get_model_configs(random_state=args.random_state)

    evaluator = ModelEvaluator(cv_folds=args.cv, random_state=args.random_state)
    results_df, best_model = evaluator.evaluate_models(
        model_configs=model_configs,
        preprocessor=dataset.preprocessor,
        X_train=dataset.X_train,
        y_train=dataset.y_train,
        X_test=dataset.X_test,
        y_test=dataset.y_test,
    )

    results_path = Path(args.results_dir) / "results.csv"
    plot_path = Path(args.results_dir) / "model_comparison.png"
    model_path = Path(args.models_dir) / "best_model.pkl"

    results_df.to_csv(results_path, index=False)
    save_comparison_plot(results_df, plot_path)
    joblib.dump(best_model, model_path)

    winner = results_df.iloc[0]
    LOGGER.info("Selected %s by CV F1 %.3f; held-out test F1 %.3f", winner["model"], winner["cv_mean_f1"],
                winner["f1_score"])
    LOGGER.info("Saved ranked model results to %s", results_path)
    LOGGER.info("Saved comparison chart to %s", plot_path)
    LOGGER.info("Saved best model to %s", model_path)


if __name__ == "__main__":
    main()
