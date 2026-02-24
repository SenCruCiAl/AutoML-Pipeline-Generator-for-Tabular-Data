"""CLI entrypoint for AutoML Pipeline Generator for Tabular Data."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import DataLoader
from evaluator import ModelEvaluator
from model_generator import get_model_configs
from utils import ensure_directories, setup_logging

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
    """Save model comparison bar plot based on F1 score."""
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(data=results_df, x="model", y="f1_score", palette="viridis")
    ax.set_title("Model Comparison by F1 Score")
    ax.set_xlabel("Model")
    ax.set_ylabel("F1 Score")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


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

    evaluator = ModelEvaluator(cv_folds=args.cv)
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

    LOGGER.info("Saved ranked model results to %s", results_path)
    LOGGER.info("Saved comparison chart to %s", plot_path)
    LOGGER.info("Saved best model to %s", model_path)


if __name__ == "__main__":
    main()
