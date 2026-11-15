"""Compare TF-IDF classifiers, then evaluate them on an untouched final test set."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODEL_DIR = ROOT / "model"
SEED = 42
LABELS = ["ham", "spam"]
VECTORIZER = {"ngram_range": (1, 2), "min_df": 2, "sublinear_tf": True, "lowercase": True}


def make_models() -> dict:
    def vectorizer():
        return TfidfVectorizer(**VECTORIZER)

    return {
        "Logistic Regression": Pipeline([
            ("tfidf", vectorizer()),
            ("classifier", LogisticRegression(max_iter=2000, random_state=SEED)),
        ]),
        "Multinomial Naive Bayes": Pipeline([
            ("tfidf", vectorizer()),
            ("classifier", MultinomialNB(alpha=1.0)),
        ]),
        "Linear SVM (sigmoid-calibrated)": CalibratedClassifierCV(
            estimator=Pipeline([
                ("tfidf", vectorizer()),
                ("classifier", LinearSVC(random_state=SEED)),
            ]),
            method="sigmoid",
            cv=3,
            ensemble=False,
        ),
    }


def metrics(y_true, y_pred) -> dict:
    matrix = confusion_matrix(y_true, y_pred, labels=LABELS)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=["spam"], average="binary", pos_label="spam", zero_division=0
    )
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "spam_precision": float(precision),
        "spam_recall": float(recall),
        "spam_f1": float(f1),
        "false_positives": int(matrix[0, 1]),
        "false_negatives": int(matrix[1, 0]),
        "confusion_matrix_ham_spam": matrix.tolist(),
        "class_order": LABELS,
    }


def score_split(estimator, frame: pd.DataFrame) -> dict:
    prediction = estimator.predict(frame.text.astype(str))
    return metrics(frame.label.to_numpy(), prediction)


def main() -> None:
    paths = {name: DATA / f"unified_{name}.csv" for name in ("train", "validation", "test")}
    frames = {name: pd.read_csv(path, keep_default_na=False) for name, path in paths.items()}
    train, valid, test = frames["train"], frames["validation"], frames["test"]
    if any(frame.empty for frame in frames.values()):
        raise ValueError("One or more prepared dataset splits are empty; run prepare_dataset.py first.")
    if set(train.label) != {"ham", "spam"}:
        raise ValueError("Training split must contain both ham and spam labels.")

    # Model selection uses validation only; threshold remains fixed at 0.50.
    validation_results = {}
    for name, estimator in make_models().items():
        estimator.fit(train.text.astype(str), train.label.astype(str))
        validation_results[name] = score_split(estimator, valid)
    chosen_name = max(
        validation_results,
        key=lambda name: (
            validation_results[name]["spam_recall"],
            validation_results[name]["spam_precision"],
            validation_results[name]["spam_f1"],
        ),
    )

    # Refit each fixed candidate on train+validation for the final comparison.
    # No test message or test score is used in selection, thresholding, or fitting.
    train_valid = pd.concat([train, valid], ignore_index=True)
    test_results = {}
    final_estimators = make_models()
    for name, estimator in final_estimators.items():
        estimator.fit(train_valid.text.astype(str), train_valid.label.astype(str))
        test_results[name] = score_split(estimator, test)

    selected = final_estimators[chosen_name]
    MODEL_DIR.mkdir(exist_ok=True)
    model_path = MODEL_DIR / "nepal_scamshield_unified.pkl"
    joblib.dump(selected, model_path)
    by_source = {}
    for source, subset in test.groupby("source_dataset", sort=True):
        by_source[source] = score_split(selected, subset)
    by_language = {}
    for (source, language), subset in test.groupby(["source_dataset", "language"], sort=True):
        by_language[f"{source} | {language}"] = score_split(selected, subset)

    report = {
        "protocol": {
            "random_seed": SEED,
            "selection_split": "validation",
            "selection_rule": "highest spam recall at probability threshold 0.50; ties broken by spam precision then spam F1",
            "final_test_used_for_training_or_tuning": False,
            "test_split_includes": "official Bengali source test rows plus seeded stratified UCI holdout",
            "tfidf": VECTORIZER,
            "linear_svm_calibration": "sigmoid calibration with 3-fold CV inside training data",
            "selected_model": chosen_name,
            "saved_model": str(model_path.relative_to(ROOT)),
        },
        "comparison_validation": validation_results,
        "comparison_final_test": test_results,
        "selected_model_final_test_by_source": by_source,
        "selected_model_final_test_by_source_and_language": by_language,
        "nepal_synthetic_set_used_for_training_or_tuning": False,
    }
    (DATA / "model_evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
