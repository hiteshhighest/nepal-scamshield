"""Evaluate the unified saved model on the separate synthetic Nepal examples."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model" / "nepal_scamshield_unified.pkl"
DATA_PATH = ROOT / "data" / "nepal_synthetic_test.csv"
REPORT_PATH = ROOT / "data" / "nepal_synthetic_evaluation.json"


def main() -> None:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Saved unified model not found: {MODEL_PATH}; run train_models.py first.")
    data = pd.read_csv(DATA_PATH, keep_default_na=False)
    if not {"message", "label"}.issubset(data.columns):
        raise ValueError("Synthetic dataset must contain 'message' and 'label' columns.")
    if not set(data.label.unique()).issubset({"ham", "spam"}):
        raise ValueError("Labels must be 'ham' or 'spam'.")

    model = joblib.load(MODEL_PATH)
    messages = data.message.astype(str)
    actual = data.label.astype(str).to_numpy()
    predictions = model.predict(messages)
    spam_idx = list(model.classes_).index("spam")
    spam_scores = model.predict_proba(messages)[:, spam_idx]
    precision, recall, f1, _ = precision_recall_fscore_support(
        actual, predictions, labels=["ham", "spam"], average="binary", pos_label="spam", zero_division=0
    )
    cm = confusion_matrix(actual, predictions, labels=["ham", "spam"])
    report = {
        "n": int(len(data)),
        "metrics": {
            "accuracy": float(accuracy_score(actual, predictions)),
            "spam_precision": float(precision),
            "spam_recall": float(recall),
            "spam_f1": float(f1),
            "confusion_matrix_ham_spam": cm.tolist(),
            "class_order": ["ham", "spam"],
        },
        "missed_spam_count": int(((actual == "spam") & (predictions == "ham")).sum()),
        "spam_support": int((actual == "spam").sum()),
        "missed_spam": [
            {"message": str(messages.iloc[i]), "spam_score": float(spam_scores[i])}
            for i in range(len(data)) if actual[i] == "spam" and predictions[i] == "ham"
        ],
        "training_or_model_selection_use": False,
        "representative_of_nepal_prevalence": False,
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("NEPAL SCAMSHIELD - SEPARATE SYNTHETIC EXPLORATORY TEST")
    print("=" * 58)
    print(f"Examples: {report['n']}")
    print(f"Accuracy: {accuracy_score(actual, predictions):.4f}")
    print(f"Spam precision: {precision:.4f}")
    print(f"Spam recall: {recall:.4f}")
    print(f"Spam F1-score: {f1:.4f}")
    print("Confusion matrix (actual ham/spam x predicted ham/spam):")
    print(cm)
    print("\nIncorrectly classified messages:")
    errors = 0
    for i, row in data.iterrows():
        if row.label != predictions[i]:
            errors += 1
            print(f"\nActual: {row.label} | Predicted: {predictions[i]} | Spam score: {spam_scores[i]:.4f}")
            print(f"Message: {row.message}")
    if not errors:
        print("None")
    print(f"\nMissed spam: {report['missed_spam_count']} of {report['spam_support']}")
    for item in report["missed_spam"]:
        print(f"Spam score: {item['spam_score']:.4f} | Message: {item['message']}")
    print("\nThis set is manually written and does not estimate real-world or Nepal-wide performance.")


if __name__ == "__main__":
    main()
