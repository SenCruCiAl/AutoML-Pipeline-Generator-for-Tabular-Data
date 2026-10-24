# AutoML Pipeline Generator for Tabular Data

![Python](https://img.shields.io/badge/python-3.11+-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-GridSearchCV-F7931E?logo=scikitlearn&logoColor=white)
[![CI](https://github.com/SenCruCiAl/AutoML-Pipeline-Generator-for-Tabular-Data/actions/workflows/ci.yml/badge.svg)](https://github.com/SenCruCiAl/AutoML-Pipeline-Generator-for-Tabular-Data/actions)

A scikit-learn pipeline that automates model selection for tabular classification (binary or multiclass): preprocessing inside each CV fold, grid search over five model families, selection by cross-validated F1, and a held-out test score the selection never saw.

**Project page:** https://sencrucial.github.io/AutoML-Pipeline-Generator-for-Tabular-Data/ (results on five datasets)

## Project Overview

This project takes a CSV dataset and target column, then:

1. Loads and validates data.
2. Builds a robust preprocessing pipeline:
   - Missing value imputation
   - Categorical feature encoding
   - Numeric standardization
3. Generates multiple candidate ML models.
4. Runs `GridSearchCV` for each candidate.
5. Evaluates all models with key metrics.
6. Ranks models and saves results to disk.
7. Selects the best model by **mean cross-validated F1 on the training split** (macro F1 for multiclass), then reports its held-out test metrics. The test set is never used for selection; doing so biases the reported score upward (Cawley & Talbot, JMLR 11, 2010).

## Architecture

```text
├── data/
│   ├── breast_cancer.csv
│   └── create_sample_dataset.py
├── src/
│   ├── data_loader.py
│   ├── model_generator.py
│   ├── evaluator.py
│   ├── utils.py
│   └── main.py
├── results/
├── models/
├── scripts/run_benchmarks.py
├── tests/
├── requirements.txt
├── docs/                 # project page + results.json
├── README.md
└── .gitignore
```

### Module Responsibilities

- `data_loader.py`: Data ingestion, validation, train/test split, and preprocessing pipeline creation.
- `model_generator.py`: Candidate classifier and parameter-grid definitions.
- `evaluator.py`: Grid-search training, cross-validation, and metric computation.
- `utils.py`: Logging setup and filesystem helpers.
- `main.py`: CLI entrypoint orchestrating end-to-end workflow.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## How to Run

From the repository root:

```bash
python src/main.py --data data/breast_cancer.csv --target target
```

### CLI Options

- `--data`: Path to CSV file.
- `--target`: Target column name.
- `--test-size`: Test split ratio (default: `0.2`).
- `--cv`: Number of cross-validation folds (default: `5`).
- `--random-state`: Random seed (default: `42`).
- `--results-dir`: Directory for outputs (default: `results`).
- `--models-dir`: Directory for model artifacts (default: `models`).
- `--log-level`: Logging verbosity (default: `INFO`).

## Results

`python scripts/run_benchmarks.py`, 5-fold CV, 80/20 split, seed 42. F1 for binary targets, macro F1 for multiclass.

| Dataset | Rows | Classes | Selected by CV | CV F1 (± sd) | Test F1 | Test-set pick would be |
|---|---|---|---|---|---|---|
| breast_cancer | 569 | 2 | LogisticRegression | 0.986 ± 0.004 | 0.979 | SVM |
| wine | 178 | 3 | SVM | 0.993 ± 0.015 | 0.971 | RandomForest |
| iris | 150 | 3 | SVM | 0.975 ± 0.034 | 0.967 | LogisticRegression |
| digits | 1797 | 10 | SVM | 0.982 ± 0.004 | 0.978 | same |
| mixed | 600 | 2 | SVM | 0.728 ± 0.052 | 0.682 | same |

On 3 of 5 datasets, ranking by test F1 (what the first version did) would have chosen a different model, and on all 5 the winner's CV score is above its test score, the selection bias that makes a separate test score necessary. `mixed` is synthetic: numeric and categorical columns, 71 missing cells and string labels.

### Fixed in this version

- **Multiclass targets crashed**: scoring was hard-coded to binary F1 and the logistic-regression grid included the binary-only `liblinear` solver. Labels are now integer-encoded; multiclass uses macro F1.
- **String labels / missing targets** now work (`LabelEncoder`, rows with a missing target dropped).
- **Selection used the test set**; it now uses CV F1 (`results.csv` has a `selected` column).
- **Deprecated API**: `SVC(probability=True)` (deprecated in scikit-learn 1.9) replaced with `CalibratedClassifierCV(SVC(), ensemble=False)`.
- Unit tests added (`tests/`), CI runs them plus the end-to-end smoke test.

## Example Output

After successful execution:

- `results/results.csv`: Ranked model comparison table.
- `results/model_comparison.png`: CV F1 vs test F1 for every model.
- `models/best_model.pkl`: Persisted best model.

Example result table columns:

- `model`
- `selected`
- `best_params`
- `accuracy`
- `precision`
- `recall`
- `f1_score`
- `roc_auc`
- `cv_mean_f1`
- `cv_std_f1`

## Future Improvements

- Add support for regression workflows.
- Add experiment tracking (e.g., MLflow).
- Introduce parallelized model search strategies (RandomizedSearchCV/Optuna).
- Export automated HTML reports.
- Nested cross-validation for a lower-variance estimate on small datasets.

## Notes

- A sample breast-cancer CSV is included at `data/breast_cancer.csv`.
- If you need to regenerate it directly from sklearn, run:

```bash
python data/create_sample_dataset.py
```
