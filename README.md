# Korean Review Sentiment Classifier

This repository contains a Korean movie-review sentiment classification project
focused on a final scikit-learn stacking model and MCC-based evaluation.

## Project Summary

The task is binary sentiment classification:

- input: Korean review text
- labels: `POSITIVE` and `NEGATIVE`
- metric: Matthews correlation coefficient, MCC

The final notebook uses a classical machine learning pipeline:

- character and word n-gram vectorization
- TF-IDF transformation
- LinearSVC, RidgeClassifier, LogisticRegression, and ComplementNB components
- stacking classifier with cross-validation

## Repository Layout

```text
notebooks/
  best_cv_binary_count_stacking_mcc07667.ipynb
```

## Data Policy

The original train and test CSV files are not included. The review text contains
user-written content, URLs, and occasional email addresses, so the public
repository keeps only the final model notebook and repository metadata.

To rerun the notebooks, place compatible files under a local `data/` directory:

```text
data/public_train.csv
data/public_test.csv
```

Expected columns:

```text
public_train.csv: row_id, text, label
public_test.csv: row_id, text
```

## Notes

This project intentionally uses conventional ML models rather than pretrained
transformer models. That makes it useful for reviewing vectorization choices
and ensemble design under a constrained modeling setup.
