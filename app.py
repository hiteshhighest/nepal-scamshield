from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model" / "nepal_scamshield_model.pkl"

st.set_page_config(page_title="Nepal ScamShield", page_icon="🛡️", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except Exception as exc:
    st.error("Could not load the trained model. Confirm the model file and compatible dependencies are installed.")
    st.exception(exc)
    st.stop()

vectoriser = model.named_steps["tfidf"]
classifier = model.named_steps["classifier"]
SPAM_INDEX = list(classifier.classes_).index("spam")


def spam_score(message: str) -> float:
    return float(model.predict_proba([message])[0][SPAM_INDEX])


def feature_contributions(message: str, limit: int = 5) -> list[dict]:
    """Rank present TF-IDF features by their positive linear spam contribution."""
    row = vectoriser.transform([message]).tocoo()
    if not row.nnz:
        return []
    spam_direction = 1 if classifier.classes_[1] == "spam" else -1
    coefficients = classifier.coef_[0]
    names = vectoriser.get_feature_names_out()
    values = [(names[index], float(value * coefficients[index] * spam_direction))
              for index, value in zip(row.col, row.data)]
    values = sorted((item for item in values if item[1] > 0), key=lambda item: item[1], reverse=True)
    return [{"Feature": term, "Contribution": round(value, 4)} for term, value in values[:limit]]


def check_indicators(message: str) -> list[str]:
    rules = {
        "Suspicious URL": r"(?:https?://|www\.|bit\.ly|tinyurl|t\.co)\S*",
        "Urgency-related language": r"\b(?:urgent|immediately|act now|right away|expire today|turuntai)\b",
        "Financial request": r"\b(?:pay|payment|transfer|fee|deposit|send money|npr|rupees|paisa)\b",
        "Password, PIN, or OTP request": r"\b(?:password|passcode|pin|otp|one.time.password)\b",
        "Reward or prize language": r"\b(?:won|winner|prize|reward|free gift|cash reward)\b",
        "Account verification language": r"\b(?:verify|verification|confirm your account|account suspended|update your details)\b",
    }
    return [label for label, pattern in rules.items() if re.search(pattern, message, re.IGNORECASE)]


st.title("🛡️ Nepal ScamShield")
st.write("An educational prototype for exploring suspicious SMS patterns in Nepal-focused scenarios.")
st.caption("A model score is an estimate, not a verdict. Verify messages through official channels.")

analyser, dashboard = st.tabs(["Message analyser", "Performance"])

with analyser:
    st.subheader("Analyse a message")
    message = st.text_area("Paste an SMS message", placeholder="Example: Your account needs verification...", height=140)
    threshold = st.slider("Spam classification threshold", 0.10, 0.90, 0.50, 0.05,
                          help="Lower values flag more messages; this can also increase false alarms.")
    if st.button("Analyse message", type="primary"):
        if not message.strip():
            st.warning("Enter a message to analyse.")
        else:
            score = spam_score(message)
            is_spam = score >= threshold
            st.divider()
            result_col, score_col = st.columns([2, 1])
            with result_col:
                if is_spam:
                    st.error("Potential spam message")
                else:
                    st.success("Likely legitimate message")
            with score_col:
                st.metric("Model-estimated spam score", f"{score:.1%}")
            st.progress(score)

            st.subheader("Model feature contributions")
            contributions = feature_contributions(message)
            if contributions:
                st.dataframe(pd.DataFrame(contributions), use_container_width=True, hide_index=True)
            else:
                st.info("No recognised features with a positive spam contribution were found.")
            st.caption("These are contributions from the model's input features, not guaranteed reasoning or proof of fraud.")

            st.subheader("Rule-based suspicious-message indicators")
            indicators = check_indicators(message)
            if indicators:
                for indicator in indicators:
                    st.warning(indicator)
            else:
                st.info("None of the basic indicator rules matched.")
            st.caption("These supplementary text rules are separate from the machine-learning prediction and can be wrong.")

with dashboard:
    st.subheader("Original held-out benchmark")
    st.caption("Reported on the original held-out set of 1,115 messages at threshold 0.50.")
    cols = st.columns(4)
    for col, label, value in zip(cols, ["Accuracy", "Spam precision", "Spam recall", "Spam F1-score"],
                                 ["97.04%", "100.00%", "77.85%", "87.55%"]):
        col.metric(label, value)
    st.write("Confusion matrix — rows are actual classes; columns are predicted classes.")
    st.dataframe(pd.DataFrame([[966, 0], [33, 116]], index=["Actual ham", "Actual spam"],
                              columns=["Predicted ham", "Predicted spam"]), use_container_width=True)

    st.divider()
    st.subheader("Nepal synthetic evaluation")
    st.caption("Separate evaluation on 16 manually written synthetic examples; this set is not representative of Nepalese scam prevalence.")
    cols = st.columns(4)
    for col, label, value in zip(cols, ["Accuracy", "Spam precision", "Spam recall", "Spam F1-score"],
                                 ["62.50%", "100.00%", "25.00%", "40.00%"]):
        col.metric(label, value)
    st.dataframe(pd.DataFrame([[8, 0], [6, 2]], index=["Actual ham", "Actual spam"],
                              columns=["Predicted ham", "Predicted spam"]), use_container_width=True)
    st.info("Six of eight synthetic spam examples were missed. This small, manually authored test is useful for illustrating failure cases, not estimating real-world performance.")

st.divider()
st.caption("Nepal ScamShield is an educational prototype. This project is a prototype and is not financial advice. Do not share passwords, PINs, or OTPs.")
