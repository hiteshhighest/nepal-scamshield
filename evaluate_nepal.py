"""Evaluate the saved model on the separate, synthetic Nepal test set."""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model" / "nepal_scamshield_model.pkl"
DATA_PATH = ROOT / "data" / "nepal_synthetic_test.csv"


def main() -> None:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Saved model not found: {MODEL_PATH}")
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Synthetic evaluation data not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    if not {"message", "label"}.issubset(df.columns):
        raise ValueError("Dataset must contain 'message' and 'label' columns.")
    if not set(df["label"].dropna().unique()).issubset({"ham", "spam"}):
        raise ValueError("Labels must be 'ham' or 'spam'.")

    model = joblib.load(MODEL_PATH)
    messages = df["message"].fillna("").astype(str)
    actual = df["label"].astype(str)
    predicted = model.predict(messages)
    scores = model.predict_proba(messages)
    spam_index = list(model.named_steps["classifier"].classes_).index("spam")
    spam_scores = scores[:, spam_index]
    precision, recall, f1, _ = precision_recall_fscore_support(
        actual, predicted, labels=["ham", "spam"], average="binary",
        pos_label="spam", zero_division=0
    )

    print("NEPAL SCAMSHIELD — SYNTHETIC EVALUATION")
    print("=" * 48)
    print(f"Examples: {len(df)} manually written synthetic messages")
    print(f"Accuracy: {accuracy_score(actual, predicted):.4f} ({accuracy_score(actual, predicted):.2%})")
    print(f"Spam precision: {precision:.4f} ({precision:.2%})")
    print(f"Spam recall: {recall:.4f} ({recall:.2%})")
    print(f"Spam F1-score: {f1:.4f} ({f1:.2%})")
    print("\nClassification report:")
    print(classification_report(actual, predicted, labels=["ham", "spam"], target_names=["Ham", "Spam"], zero_division=0))
    print("Confusion matrix (rows: actual ham/spam; columns: predicted ham/spam):")
    print(confusion_matrix(actual, predicted, labels=["ham", "spam"]))

    print("\nIncorrectly classified messages:")
    errors = 0
    for message, truth, guess, score in zip(messages, actual, predicted, spam_scores):
        if truth != guess:
            errors += 1
            print(f"\nActual: {truth} | Predicted: {guess} | Spam score: {score:.4f} ({score:.2%})")
            print(f"Message: {message}")
    if not errors:
        print("None")

    missed = (actual.to_numpy() == "spam") & (predicted == "ham")
    print("\nMissed spam messages and model-estimated spam scores:")
    if not missed.any():
        print("None")
    for message, score in zip(messages[missed], spam_scores[missed]):
        print(f"Spam score: {score:.4f} ({score:.2%}) | Message: {message}")


if __name__ == "__main__":
    main()
