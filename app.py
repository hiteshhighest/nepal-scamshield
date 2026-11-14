
import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------
# PAGE CONFIGURATION
# -----------------------------

st.set_page_config(
    page_title="Nepal ScamShield",
    page_icon="🛡️",
    layout="wide"
)

# -----------------------------
# LOAD MODEL
# -----------------------------

@st.cache_resource
def load_model():
    return joblib.load("model/nepal_scamshield_model.pkl")

try:
    model = load_model()
except Exception as error:
    st.error("Could not load the trained model.")
    st.code(str(error))
    st.stop()

vectoriser = model.named_steps["tfidf"]
classifier = model.named_steps["classifier"]

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------

def get_spam_score(message):
    probabilities = model.predict_proba([message])[0]
    spam_index = list(classifier.classes_).index("spam")
    return float(probabilities[spam_index])


def explain_message(message):
    vector = vectoriser.transform([message])
    feature_names = vectoriser.get_feature_names_out()
    coefficients = classifier.coef_[0]

    # Positive contributions indicate spam.
    direction = (
        1 if classifier.classes_[1] == "spam" else -1
    )

    contributions = vector.toarray()[0] * coefficients * direction

    important = []

    for index in np.argsort(contributions)[::-1]:
        if contributions[index] <= 0:
            break

        important.append({
            "Term": feature_names[index],
            "Contribution": round(float(contributions[index]), 4)
        })

        if len(important) == 5:
            break

    return important


def check_indicators(message):
    indicators = []

    if re.search(
        r"https?://|www\.|bit\.ly|tinyurl",
        message,
        re.IGNORECASE
    ):
        indicators.append(
            "Contains a web link. Check its destination carefully."
        )

    patterns = {
        "Urgency": r"\burgent\b|\bimmediately\b|\bact now\b",
        "Financial request": r"\bpay\b|\btransfer\b|\bpayment\b",
        "Sensitive information": r"\botp\b|\bpin\b|\bpassword\b",
        "Reward or prize": r"\bwon\b|\bprize\b|\breward\b|\bfree\b",
        "Account verification": r"\bverify\b|\baccount suspended\b"
    }

    for name, pattern in patterns.items():
        if re.search(pattern, message, re.IGNORECASE):
            indicators.append(name)

    return indicators


# -----------------------------
# HEADER
# -----------------------------

st.title("🛡️ Nepal ScamShield")

st.write(
    "An AI-powered SMS classification prototype "
    "designed to explore suspicious messages and "
    "scam-awareness scenarios relevant to Nepal."
)

st.divider()

# -----------------------------
# MAIN CLASSIFIER
# -----------------------------

st.subheader("Analyse a message")

message = st.text_area(
    "Paste an SMS message below",
    placeholder="Example: Congratulations! You have won a prize...",
    height=150
)

threshold = st.slider(
    "Spam classification threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.50,
    step=0.05,
    help="A lower threshold flags more messages as spam, "
         "but may also flag more legitimate messages."
)

if st.button("Analyse message", type="primary"):
    if not message.strip():
        st.warning("Please enter a message first.")

    else:
        score = get_spam_score(message)
        prediction = "spam" if score >= threshold else "ham"

        st.divider()
        st.subheader("Analysis result")

        col1, col2 = st.columns(2)

        with col1:
            if prediction == "spam":
                st.error("Potential spam message")
            else:
                st.success("Likely legitimate message")

        with col2:
            st.metric(
                "Model-estimated spam score",
                f"{score * 100:.2f}%"
            )

        st.progress(score)

        # Model explanation
        st.subheader("What influenced the prediction?")

        explanations = explain_message(message)

        if explanations:
            st.dataframe(
                pd.DataFrame(explanations),
                use_container_width=True,
                hide_index=True
            )
            st.caption(
                "These terms contributed positively to the model's "
                "spam prediction. They are not proof of fraud."
            )
        else:
            st.info(
                "The model found no strong spam-associated terms "
                "among the message's recognised features."
            )

        # Additional indicators
        st.subheader("Suspicious-message indicators")

        indicators = check_indicators(message)

        if indicators:
            for indicator in indicators:
                st.warning(indicator)
        else:
            st.info("No indicators from our basic rule checks were found.")

        st.caption(
            "These additional indicators are simple rules, "
            "not independent proof that a message is fraudulent."
        )

# -----------------------------
# PERFORMANCE DASHBOARD
# -----------------------------

st.divider()
st.subheader("Model performance")

st.caption(
    "Results from the held-out test set of 1,115 messages. "
    "Classification threshold: 0.50."
)

c1, c2, c3, c4 = st.columns(4)

c1.metric("Accuracy", "97.04%")
c2.metric("Spam precision", "100.00%")
c3.metric("Spam recall", "77.85%")
c4.metric("Spam F1-score", "87.55%")

st.write("### Confusion matrix")

confusion = pd.DataFrame(
    [[966, 0], [33, 116]],
    index=["Actual Ham", "Actual Spam"],
    columns=["Predicted Ham", "Predicted Spam"]
)

st.dataframe(
    confusion,
    use_container_width=True
)

st.info(
    "The model missed 33 spam messages in this test set. "
    "This is an important limitation, and the model should "
    "not be treated as a guaranteed scam detector."
)

st.divider()

st.caption(
    "Nepal ScamShield is an experimental educational prototype. "
    "It cannot guarantee that a message is safe or fraudulent. "
    "Do not share private messages, passwords, PINs or OTPs."
)
