from html import escape
from pathlib import Path
import json
import re

import joblib
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model" / "nepal_scamshield_unified.pkl"
EVALUATION_PATH = ROOT / "data" / "model_evaluation.json"
AUDIT_PATH = ROOT / "data" / "dataset_audit.json"
SYNTHETIC_EVALUATION_PATH = ROOT / "data" / "nepal_synthetic_evaluation.json"

PAGES = ["Overview", "Message Analyser", "Model Performance", "Nepal Evaluation", "About"]
NAV_ICONS = {
    "Overview": "space_dashboard",
    "Message Analyser": "sms",
    "Model Performance": "query_stats",
    "Nepal Evaluation": "travel_explore",
    "About": "info",
}

st.set_page_config(
    page_title="Nepal ScamShield",
    page_icon="\U0001F6E1\uFE0F",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      :root {
        --canvas: #0B1220;
        --surface: #121D2E;
        --surface-raised: #1A2940;
        --surface-soft: #101A2A;
        --stroke: #29384D;
        --stroke-strong: #40536C;
        --ink: #F1F5F9;
        --muted: #A6B2C2;
        --muted-2: #8493A8;
        --accent-blue: #6BA8FF;
        --accent-blue-bright: #A8CDFF;
        --accent-blue-soft: rgba(107,168,255,.12);
        --rose: #FB7185;
        --amber: #F6C969;
      }
      html, body, [class*="css"] { font-family: "Aptos", "Segoe UI Variable", "Segoe UI", sans-serif; }
      .stApp { background: var(--canvas); color: var(--ink); }
      [data-testid="stHeader"] { background: rgba(11,18,32,.82); }
      .block-container { max-width: 1240px; padding-top: 1.35rem; padding-bottom: 2.5rem; }
      [data-testid="stToolbar"] { right: 1rem; }

      [data-testid="stSidebar"] {
        background: #0E1726; border-right: 1px solid #253347;
      }
      [data-testid="stSidebar"] > div:first-child { padding-top: 1.15rem; }
      [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: var(--muted); }
      .sidebar-brand { display:flex; align-items:center; gap:.75rem; padding:.25rem .35rem 1.2rem; border-bottom:1px solid #29384D; margin-bottom:1.15rem; }
      .sidebar-logo { width:38px; height:38px; display:grid; place-items:center; border-radius:12px; background:linear-gradient(145deg,#183253,#6BA8FF); box-shadow:0 7px 20px rgba(107,168,255,.13); }
      .sidebar-logo svg { width:22px; height:22px; }
      .sidebar-name { color:var(--ink); font-weight:750; font-size:.92rem; letter-spacing:-.02em; }
      .sidebar-sub { color:var(--muted-2); font-size:.72rem; margin-top:.1rem; }
      .sidebar-section { color:#718198; font-size:.67rem; font-weight:750; letter-spacing:.12em; text-transform:uppercase; padding:.35rem .7rem .5rem; }
      [data-testid="stSidebar"] [data-testid="stRadio"] label {
        min-height:42px; padding:.42rem .62rem; border:1px solid transparent; border-radius:10px;
        transition:background .18s ease, border-color .18s ease, transform .18s ease;
      }
      [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background:#17263B; border-color:#34475E; transform:translateX(2px);
      }
      [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
        background:rgba(107,168,255,.10); border-color:rgba(107,168,255,.22);
      }
      [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p { color:#DFEDFF; font-weight:700; }
      .sidebar-foot { margin-top:1.5rem; padding:.85rem .7rem; border:1px solid #29384D; border-radius:12px; background:#111C2C; }
      .sidebar-foot-title { color:#D7E2EF; font-size:.76rem; font-weight:700; }
      .sidebar-foot-copy { color:#8999AE; font-size:.7rem; line-height:1.55; margin-top:.3rem; }

      .console-intro { padding:.4rem 0 1.15rem; margin:.2rem 0 1.15rem; border-bottom:1px solid #26364B; }
      .console-topline { display:flex; align-items:center; justify-content:space-between; gap:1rem; }
      .console-kicker { display:flex; align-items:center; gap:.55rem; color:#9AB0CA; font-size:.68rem; font-weight:750; letter-spacing:.12em; text-transform:uppercase; }
      .console-kicker:before { content:""; width:7px; height:7px; border-radius:50%; background:#63C9A5; box-shadow:0 0 0 4px rgba(99,201,165,.12); }
      .console-status { display:flex; align-items:center; gap:.55rem; padding:.42rem .68rem; border:1px solid #29415D; border-radius:8px; background:#101E30; color:#C8D8EA; font-size:.72rem; font-weight:650; }
      .console-status span { color:#63C9A5; font-size:.7rem; }
      .console-intro h1 { margin:.8rem 0 .25rem; color:#F1F5F9; font-size:clamp(1.85rem,3vw,2.55rem); font-weight:760; letter-spacing:-.045em; line-height:1.12; }
      .console-intro p { max-width:760px; margin:0; color:#A6B2C2; font-size:.9rem; line-height:1.65; }
      .console-meta { margin-top:.72rem; color:#7F91A8; font-size:.7rem; letter-spacing:.015em; }
      .console-meta strong { color:#B4C7DD; font-weight:650; }
      .overview-snapshot { padding:.85rem 0 .35rem; margin-bottom:1.1rem; border-bottom:1px solid #202F43; }
      .snapshot-title { display:flex; align-items:center; justify-content:space-between; gap:.8rem; margin-bottom:.65rem; color:#A9BBD0; font-size:.69rem; font-weight:700; letter-spacing:.09em; text-transform:uppercase; }
      .snapshot-scope { color:#778AA2; font-size:.66rem; font-weight:500; letter-spacing:0; text-transform:none; }
      .chart-heading { margin:.35rem 0 .15rem; color:#DFE9F5; font-size:.88rem; font-weight:650; }

      [data-testid="stTabs"] [data-baseweb="tab-list"] { gap:.55rem; border-bottom:1px solid var(--stroke); }
      [data-testid="stTabs"] button[role="tab"] { color:var(--muted); font-weight:650; padding:.72rem 1rem; transition:color .18s ease,transform .18s ease; }
      [data-testid="stTabs"] button[role="tab"]:hover { color:var(--accent-blue-bright); transform:translateY(-1px); }
      [data-testid="stTabs"] button[role="tab"][aria-selected="true"] { color:#DFEDFF; }
      [data-testid="stTabs"] [data-baseweb="tab-highlight"] { height:3px; border-radius:3px; background:var(--accent-blue); transition:left .25s ease,width .25s ease; }
      [data-testid="stTabContent"] { animation:fade-in .3s ease-out both; }

      h1,h2,h3,h4 { color:var(--ink); letter-spacing:-.035em; }
      h2 { font-weight:760; }
      h3 { font-weight:720; }
      .page-kicker { color:#9CC4F4; font-size:.67rem; font-weight:800; letter-spacing:.14em; text-transform:uppercase; margin:.15rem 0 .2rem; }
      .page-desc { color:var(--muted); font-size:.9rem; line-height:1.6; margin-bottom:1.15rem; }
      .section-label { margin:.3rem 0 .28rem; color:#9CC4F4; font-size:.68rem; font-weight:800; letter-spacing:.13em; text-transform:uppercase; }
      .muted-copy { color:var(--muted); font-size:.86rem; line-height:1.6; }

      [data-testid="stVerticalBlockBorderWrapper"] {
        border:1px solid var(--stroke); border-radius:16px; background:var(--surface);
        box-shadow:0 4px 15px rgba(0,0,0,.08); animation:fade-rise .42s cubic-bezier(.2,.7,.2,1) both;
        transition:transform .2s ease,border-color .2s ease,box-shadow .2s ease;
      }
      [data-testid="stVerticalBlockBorderWrapper"]:hover { transform:translateY(-2px); border-color:#3A5369; box-shadow:0 12px 28px rgba(0,0,0,.18); }
      [data-testid="stMetric"] { border:1px solid var(--stroke); border-radius:14px; padding:.95rem 1rem; background:linear-gradient(145deg,#18263A,#121D2E); animation:fade-rise .4s ease-out both; transition:transform .18s ease,border-color .18s ease; }
      [data-testid="stMetric"]:hover { transform:translateY(-2px); border-color:#3A5369; }
      [data-testid="stMetricLabel"] { color:#A6B2C2; font-weight:600; }
      [data-testid="stMetricValue"] { color:#F1F5F9; font-weight:760; }
      [data-testid="stTextArea"] textarea { color:#F1F5F9; background:#0E1726; border:1px solid #33445A; border-radius:12px; transition:border-color .18s ease,box-shadow .18s ease,background .18s ease; }
      [data-testid="stTextArea"] textarea:focus { border-color:#6BA8FF; background:#111D2F; box-shadow:0 0 0 3px rgba(107,168,255,.14); }
      [data-testid="stTextArea"] textarea::placeholder { color:#76879D; }
      [data-testid="stSlider"] [role="slider"] { background:var(--accent-blue); }
      [data-testid="stSlider"] [data-baseweb="slider"] div[role="presentation"] { background-color:#26374C; }
      .stButton > button, [data-testid="stFormSubmitButton"] button { border-radius:10px; font-weight:700; transition:transform .16s ease,box-shadow .16s ease,border-color .16s ease,background .16s ease; }
      [data-testid="stFormSubmitButton"] button[kind="primary"] { color:#09251E; border:1px solid #6BA8FF; background:linear-gradient(135deg,#A8CDFF,#6BA8FF); box-shadow:0 7px 17px rgba(107,168,255,.13); }
      [data-testid="stFormSubmitButton"] button[kind="primary"]:hover { border-color:#D0E3FF; background:linear-gradient(135deg,#D0E3FF,#6BA8FF); box-shadow:0 9px 22px rgba(107,168,255,.19); transform:translateY(-1px); }
      .stButton > button:hover { border-color:#3B526A; color:#E7F2F7; transform:translateY(-1px); }
      .stButton > button:active,[data-testid="stFormSubmitButton"] button:active { transform:translateY(1px) scale(.99); }
      [data-testid="stProgressBar"] > div { background:#26364B; border-radius:999px; }
      [data-testid="stProgressBar"] > div > div { background:linear-gradient(90deg,#4D82C6,#A8CDFF); border-radius:999px; transition:width .65s cubic-bezier(.2,.7,.2,1); }
      [data-testid="stAlert"] { border-radius:12px; animation:result-in .3s ease-out both; }
      [data-testid="stDataFrame"] { border:1px solid var(--stroke); border-radius:12px; overflow:hidden; }
      [data-testid="stExpander"] { border:1px solid var(--stroke); border-radius:12px; background:rgba(18,29,46,.78); transition:border-color .18s ease,box-shadow .18s ease; }
      [data-testid="stExpander"]:hover { border-color:#3A5369; box-shadow:0 5px 18px rgba(0,0,0,.12); }
      [data-testid="stSpinner"] p { color:#B6C4D6; }
      [data-testid="stStatusWidget"] { border:1px solid var(--stroke); border-radius:12px; background:var(--surface); }
      a { color:#A8CDFF !important; text-decoration-color:rgba(168,205,255,.4) !important; }
      a:hover { color:#A8CDFF !important; }
      hr { border-color:var(--stroke); }
      .feature-sample { display:flex; flex-direction:column; gap:.15rem; padding:.2rem 0 .45rem; }
      .feature-sample strong { color:#DCE7F2; font-size:.75rem; }
      .feature-sample span { color:#8091A7; font-size:.69rem; line-height:1.45; }
      .highlighted-message { padding:1rem 1.1rem; border:1px solid #2A3C52; border-radius:12px; color:#C9D6E5; background:#0E1726; white-space:pre-wrap; line-height:1.75; overflow-wrap:anywhere; }
      .highlighted-message mark { padding:.08rem .16rem; border:1px solid rgba(107,168,255,.25); border-radius:4px; color:#E3F0FF; background:rgba(107,168,255,.16); }
      .callout { padding:.8rem 1rem; border:1px solid rgba(107,168,255,.2); border-radius:12px; color:#C5D4E3; background:rgba(107,168,255,.07); font-size:.82rem; line-height:1.55; }
      .disclaimer { padding:.85rem 1rem; border:1px solid #354258; border-radius:12px; background:#101A2A; color:#AAB8C9; font-size:.76rem; line-height:1.55; }

      @keyframes fade-in { from { opacity:0; } to { opacity:1; } }
      @keyframes fade-rise { from { opacity:0; transform:translateY(8px); } to { opacity:1; transform:translateY(0); } }
      @keyframes result-in { from { opacity:0; transform:translateY(6px) scale(.99); } to { opacity:1; transform:translateY(0) scale(1); } }
      @media (max-width: 760px) {
        .block-container { padding:1rem .9rem 2rem; }
        .console-topline { align-items:flex-start; flex-direction:column; gap:.6rem; }
        .console-intro h1 { font-size:1.9rem; }
        .console-meta { line-height:1.6; }
        .snapshot-title { align-items:flex-start; flex-direction:column; gap:.15rem; }
        [data-testid="stMetric"] { padding:.78rem; }
      }
      @media (max-width: 860px) {
        [data-testid="stHorizontalBlock"] { flex-wrap:wrap; gap:.7rem; }
        [data-testid="stHorizontalBlock"] > [data-testid="column"] { min-width:calc(50% - .4rem) !important; flex:1 1 calc(50% - .4rem) !important; }
      }
      @media (max-width: 520px) {
        [data-testid="stHorizontalBlock"] > [data-testid="column"] { min-width:100% !important; flex:1 1 100% !important; }
      }
      @media (prefers-reduced-motion: reduce) {
        *,*::before,*::after { animation-duration:.01ms !important; animation-iteration-count:1 !important; scroll-behavior:auto !important; transition-duration:.01ms !important; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def load_json(path: Path):
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


try:
    model = load_model()
except Exception as exc:
    st.error("The saved model could not be loaded. Check the model file and installed dependencies.")
    st.exception(exc)
    st.stop()


def model_score(message: str) -> float:
    class_names = list(model.classes_)
    return float(model.predict_proba([message])[0][class_names.index("spam")])


def feature_contributions(message: str, limit: int = 7) -> list[dict]:
    """Rank present TF-IDF features by positive contribution to the spam class."""
    if hasattr(model, "named_steps"):
        estimators = [model]
    else:
        estimators = [getattr(item, "estimator", None) for item in getattr(model, "calibrated_classifiers_", [])]
        estimators = [item for item in estimators if item is not None and hasattr(item, "named_steps")]
    if not estimators:
        return []

    pipeline = estimators[0]
    vectoriser = pipeline.named_steps.get("tfidf")
    classifier = pipeline.named_steps.get("classifier")
    if vectoriser is None or classifier is None:
        return []
    classes = list(classifier.classes_)
    spam_index = classes.index("spam")
    if hasattr(classifier, "coef_"):
        coefficients = classifier.coef_[0] * (1 if spam_index == 1 else -1)
    elif hasattr(classifier, "feature_log_prob_") and len(classes) == 2:
        coefficients = classifier.feature_log_prob_[spam_index] - classifier.feature_log_prob_[1 - spam_index]
    else:
        return []

    row = vectoriser.transform([message]).tocoo()
    if not row.nnz:
        return []
    names = vectoriser.get_feature_names_out()
    ranked = sorted(
        (
            (str(names[index]), float(value * coefficients[index]))
            for index, value in zip(row.col, row.data)
            if value * coefficients[index] > 0
        ),
        key=lambda pair: pair[1],
        reverse=True,
    )
    return [{"Feature": term, "Contribution": round(value, 4)} for term, value in ranked[:limit]]


def highlighted_message(message: str, contributions: list[dict]) -> str:
    """Safely highlight matching contributing terms in a message excerpt."""
    candidates = []
    for item in contributions:
        term = item["Feature"]
        pattern = re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", re.IGNORECASE)
        for match in pattern.finditer(message):
            candidates.append((match.start(), match.end(), float(item["Contribution"])))
    candidates.sort(key=lambda item: (-item[2], -(item[1] - item[0]), item[0]))
    selected = []
    for start, end, _ in candidates:
        if all(end <= old_start or start >= old_end for old_start, old_end in selected):
            selected.append((start, end))
    selected.sort()

    pieces = []
    cursor = 0
    for start, end in selected:
        pieces.append(escape(message[cursor:start]))
        pieces.append(f"<mark>{escape(message[start:end])}</mark>")
        cursor = end
    pieces.append(escape(message[cursor:]))
    return "".join(pieces)


def check_indicators(message: str) -> list[str]:
    patterns = {
        "Suspicious URL": r"(?:https?://|www\.|bit\.ly|tinyurl|t\.co)\S*",
        "Urgency language": r"\b(?:urgent|immediately|act now|right away|expire today|turuntai)\b",
        "Financial request": r"\b(?:pay|payment|transfer|fee|deposit|send money|npr|rupees|paisa)\b",
        "Credential or code request": r"\b(?:password|passcode|pin|otp|one.time.password)\b",
        "Reward or prize language": r"\b(?:won|winner|prize|reward|free gift|cash reward)\b",
        "Account verification language": r"\b(?:verify|verification|confirm your account|account suspended|update your details)\b",
    }
    return [name for name, pattern in patterns.items() if re.search(pattern, message, re.IGNORECASE)]


def render_metrics(metrics: dict) -> None:
    columns = st.columns(4, gap="small")
    fields = [
        ("Accuracy", "accuracy"),
        ("Spam precision", "spam_precision"),
        ("Spam recall", "spam_recall"),
        ("Spam F1", "spam_f1"),
    ]
    for column, (label, key) in zip(columns, fields):
        column.metric(label, f"{metrics[key]:.2%}")


def confusion_frame(metrics: dict) -> pd.DataFrame:
    return pd.DataFrame(
        metrics["confusion_matrix_ham_spam"],
        index=["Actual ham", "Actual spam"],
        columns=["Predicted ham", "Predicted spam"],
    )


def page_heading(kicker: str, title: str, description: str) -> None:
    st.markdown(f'<div class="page-kicker">{escape(kicker)}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="page-desc">{escape(description)}</div>', unsafe_allow_html=True)


def render_landing():
    st.markdown(
        """
        <section class="console-intro">
          <div class="console-topline">
            <div class="console-kicker">Message threat analysis · Nepal awareness</div>
            <div class="console-status"><span aria-hidden="true">●</span> Model loaded</div>
          </div>
          <h1>Nepal ScamShield</h1>
          <p>Review a message's estimated spam score and separate warning indicators before deciding what to do.</p>
          <div class="console-meta"><strong>MODEL:</strong> calibrated Linear SVM &nbsp;·&nbsp; <strong>TRAINING SCOPE:</strong> English and Bengali-focused SMS corpora</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_overview_snapshot():
    report = load_json(EVALUATION_PATH)
    if not report:
        return
    selected = report["protocol"]["selected_model"]
    metrics = report["comparison_final_test"][selected]
    st.markdown(
        f'<div class="snapshot-title"><span>Held-out corpus snapshot</span>'
        f'<span class="snapshot-scope">{metrics["n"]:,} messages · English and Bengali-focused benchmark</span></div>',
        unsafe_allow_html=True,
    )
    columns = st.columns(4, gap="small")
    for column, (label, value) in zip(columns, [
        ("Spam recall", f"{metrics['spam_recall']:.1%}"),
        ("Spam precision", f"{metrics['spam_precision']:.1%}"),
        ("Spam F1", f"{metrics['spam_f1']:.1%}"),
        ("Missed spam", str(metrics["false_negatives"])),
    ]):
        column.metric(label, value)
    st.caption("Benchmark metrics cover the prepared English and Bengali-focused corpora. Nepal-focused examples are evaluated separately.")


SAMPLE_MESSAGES = {
    "Wallet verification": "Your wallet account needs verification today. Visit https://account-check.example to continue.",
    "Prize fee request": "Congratulations! Claim your reward by paying a processing fee today.",
    "Class conversation": "Can you send me the notes from today's physics class?",
}


def set_sample_message(message: str) -> None:
    st.session_state["scamshield_message"] = message


def render_analyser(compact: bool = False) -> None:
    if compact:
        st.markdown('<div class="section-label">Quick message check</div>', unsafe_allow_html=True)
        st.subheader("Analyse a message")
        st.markdown('<div class="muted-copy">Explore the sample messages or paste your own. Avoid entering private information, account numbers or one-time codes.</div>', unsafe_allow_html=True)
    else:
        page_heading(
            "Message analyser",
            "Review a message",
            "Use the model estimate and independent warning indicators as complementary signals when reviewing a message.",
        )

    st.markdown('<div class="section-label">Try a demo example</div>', unsafe_allow_html=True)
    sample_columns = st.columns(3, gap="small")
    for column, (label, message) in zip(sample_columns, SAMPLE_MESSAGES.items()):
        with column:
            with st.container(border=True):
                st.markdown(f'<div class="feature-sample"><strong>{escape(label)}</strong><span>{escape(message)}</span></div>', unsafe_allow_html=True)
                st.button(
                    "Use example",
                    key=f"sample_{label.lower().replace(' ', '_')}",
                    width="stretch",
                    on_click=set_sample_message,
                    args=(message,),
                )
    st.caption("These short examples are for trying the interface only; they are not additional evaluation data.")

    with st.container(border=True):
        with st.form("message_analysis", clear_on_submit=False):
            message = st.text_area(
                "SMS message",
                key="scamshield_message",
                placeholder="Paste or type a message to inspect...",
                height=165,
                label_visibility="collapsed",
            )
            slider_column, button_column = st.columns([1.55, .8], vertical_alignment="bottom", gap="large")
            with slider_column:
                threshold = st.slider(
                    "Spam classification threshold",
                    min_value=0.10,
                    max_value=0.90,
                    value=0.50,
                    step=0.05,
                    help="A lower threshold flags more messages and can increase false positives.",
                )
            with button_column:
                analyse = st.form_submit_button("Analyse message", type="primary", width="stretch")

    if not analyse:
        return
    if not message.strip():
        st.warning("Enter a message or choose one of the examples above.")
        return

    try:
        with st.spinner("Scanning message patterns..."):
            score = model_score(message)
            contributions = feature_contributions(message)
            indicators = check_indicators(message)
    except Exception as exc:
        st.error("The message could not be analysed. Please try again.")
        with st.expander("Technical details"):
            st.exception(exc)
        return

    st.markdown('<div class="section-label">Analysis result</div>', unsafe_allow_html=True)
    result_col, detail_col = st.columns([1.03, 1.5], gap="medium")
    with result_col:
        with st.container(border=True):
            st.markdown('<div class="section-label">Model prediction</div>', unsafe_allow_html=True)
            if score >= threshold:
                st.error("Potential spam message")
            else:
                st.success("Likely legitimate message")
            st.metric("Estimated spam probability", f"{score:.1%}")
            st.progress(score, text=f"Spam estimate: {score:.0%}")
            st.caption(f"Classification threshold: {threshold:.0%}. Spam probability is estimated by the saved model.")
    with detail_col:
        with st.container(border=True):
            st.markdown('<div class="section-label">Independent rule checks</div>', unsafe_allow_html=True)
            if indicators:
                st.warning("Possible message patterns detected")
                for indicator in indicators:
                    st.markdown(f"- {escape(indicator)}")
            else:
                st.success("No configured warning patterns detected")
            st.caption("Configured URL and keyword indicators are shown separately from the machine-learning prediction.")

    st.markdown('<div class="section-label">Why the model leaned this way</div>', unsafe_allow_html=True)
    st.subheader("Text features associated with the estimate")
    if contributions:
        st.markdown(
            f'<div class="highlighted-message">{highlighted_message(message, contributions)}</div>',
            unsafe_allow_html=True,
        )
        feature_col, explanation_col = st.columns([1, 1.3], gap="large")
        with feature_col:
            st.dataframe(pd.DataFrame(contributions), width="stretch", hide_index=True)
        with explanation_col:
            st.markdown(
                '<div class="callout"><strong>How to read this:</strong> highlighted words or phrases are present TF-IDF features contributing toward the spam score. They show model associations, not proof of fraud.</div>',
                unsafe_allow_html=True,
            )
    else:
        st.info("No recognised input features with a positive spam contribution were found.")


def render_model_performance():
    page_heading(
        "Model performance",
        "How the model performed",
        "Metrics are computed on the prepared corpus splits and loaded from the saved evaluation report. The final test is kept separate from selection.",
    )
    results = load_json(EVALUATION_PATH)
    if results is None:
        st.warning("Evaluation results are unavailable. Run `python train_models.py` to generate the report.")
        return

    selected = results["protocol"]["selected_model"]
    selected_metrics = results["comparison_final_test"][selected]
    left, right = st.columns([1.5, 1], gap="large")
    with left:
        st.markdown('<div class="section-label">Selected model</div>', unsafe_allow_html=True)
        st.subheader(selected)
        st.markdown("Chosen on validation by spam recall. Final-test results did not select or tune the model.")
    with right:
        st.markdown(f'<div class="callout">Final test: <strong>{selected_metrics["n"]:,}</strong> messages<br>False positives: <strong>{selected_metrics["false_positives"]}</strong> &nbsp; Missed spam: <strong>{selected_metrics["false_negatives"]}</strong></div>', unsafe_allow_html=True)
    render_metrics(selected_metrics)

    with st.container(border=True):
        st.markdown('<div class="section-label">Final test comparison</div>', unsafe_allow_html=True)
        st.subheader("TF-IDF classifier comparison")
        st.caption("All candidates use the same fixed 0.50 probability threshold. Confusion matrix rows are actual ham/spam; columns are predicted ham/spam.")
        comparison = results["comparison_final_test"]
        final_table = pd.DataFrame([
            {
                "Model": name,
                "Accuracy": metrics["accuracy"],
                "Spam precision": metrics["spam_precision"],
                "Spam recall": metrics["spam_recall"],
                "Spam F1": metrics["spam_f1"],
                "False positives": metrics["false_positives"],
                "Missed spam": metrics["false_negatives"],
            }
            for name, metrics in comparison.items()
        ]).set_index("Model")
        st.dataframe(final_table.style.format({
            "Accuracy": "{:.2%}", "Spam precision": "{:.2%}",
            "Spam recall": "{:.2%}", "Spam F1": "{:.2%}",
        }), width="stretch")
        st.markdown('<div class="chart-heading">Spam precision and recall by model</div>', unsafe_allow_html=True)
        st.caption("Final-test scores at the fixed 0.50 threshold. The chart summarizes the table above.")
        st.bar_chart(
            final_table[["Spam precision", "Spam recall"]],
            horizontal=True,
            color=["#6BA8FF", "#43C9C1"],
            x_label="Score",
            y_label="Model",
            height=220,
            sort=False,
        )
        with st.expander("Final-test confusion matrices"):
            matrix_columns = st.columns(len(comparison), gap="small")
            for column, (name, metrics) in zip(matrix_columns, comparison.items()):
                with column:
                    st.markdown(f"**{name}**")
                    st.dataframe(confusion_frame(metrics), width="stretch")

    with st.expander("Validation results used to choose the model"):
        validation = results["comparison_validation"]
        val_table = pd.DataFrame([
            {"Model": name, "Accuracy": metrics["accuracy"], "Spam precision": metrics["spam_precision"],
             "Spam recall": metrics["spam_recall"], "Spam F1": metrics["spam_f1"]}
            for name, metrics in validation.items()
        ]).set_index("Model")
        st.dataframe(val_table.style.format("{:.2%}"), width="stretch")
        st.caption("Selection rule: highest spam recall, with ties broken by spam precision and then spam F1.")

    st.markdown('<div class="section-label">Coverage</div>', unsafe_allow_html=True)
    st.subheader("Performance by source and language")
    st.caption("Bengali, Banglish and CodeMix slices do not establish performance on Nepali.")
    for source, metrics in results["selected_model_final_test_by_source"].items():
        with st.expander(f"{source} - {metrics['n']:,} examples"):
            render_metrics(metrics)
            st.dataframe(confusion_frame(metrics), width="stretch")

    rows = []
    for group, metrics in results["selected_model_final_test_by_source_and_language"].items():
        rows.append({
            "Source | language": group,
            "Examples": metrics["n"],
            "Accuracy": metrics["accuracy"],
            "Spam precision": metrics["spam_precision"],
            "Spam recall": metrics["spam_recall"],
            "Spam F1": metrics["spam_f1"],
        })
    language_table = pd.DataFrame(rows).set_index("Source | language")
    st.dataframe(language_table.style.format({
        "Accuracy": "{:.2%}", "Spam precision": "{:.2%}",
        "Spam recall": "{:.2%}", "Spam F1": "{:.2%}",
    }), width="stretch")


def render_nepal_evaluation():
    page_heading(
        "Nepal evaluation",
        "Nepal-focused evaluation",
        "Sixteen manually written examples, held separate from training and model selection.",
    )
    report = load_json(SYNTHETIC_EVALUATION_PATH)
    if report is None:
        st.warning("The synthetic evaluation report is unavailable. Run `python evaluate_nepal.py`.")
        return

    with st.container(border=True):
        head_col, count_col = st.columns([1.5, 1], gap="large")
        with head_col:
            st.markdown('<div class="section-label">Synthetic evaluation</div>', unsafe_allow_html=True)
            st.subheader("Nepal-focused examples")
            st.markdown("This score summarizes the authored examples in this set and is reported separately from the held-out corpus benchmark.")
        with count_col:
            st.metric("Manually written examples", report["n"])
            st.caption("Manually written evaluation examples")
        render_metrics(report["metrics"])
        st.dataframe(confusion_frame(report["metrics"]), width="stretch")
        st.markdown(f"**Missed spam:** {report['missed_spam_count']} of {report['spam_support']}")

    if report["missed_spam"]:
        with st.expander("Missed spam messages and model scores"):
            for item in report["missed_spam"]:
                st.markdown(f"- Spam score **{item['spam_score']:.1%}** - {escape(item['message'])}")
    else:
        st.info("All 8 synthetic spam examples were classified as spam in this evaluation. The result is reported for these 16 messages.")

    st.markdown(
        '<div class="callout">The classifier is trained on English and Bengali-focused corpora. This separate evaluation uses Nepal-focused synthetic examples.</div>',
        unsafe_allow_html=True,
    )


def render_about():
    page_heading(
        "About ScamShield",
        "How ScamShield works",
        "A small text-classification and awareness tool. It combines a saved machine-learning model with simple, separately reported message-pattern checks.",
    )
    first, second = st.columns(2, gap="large")
    with first:
        with st.container(border=True):
            st.markdown('<div class="section-label">Model</div>', unsafe_allow_html=True)
            st.subheader("TF-IDF classifiers")
            st.markdown("The saved model is a sigmoid-calibrated Linear SVM trained on TF-IDF word unigrams and bigrams. Logistic Regression and Multinomial Naive Bayes were compared on the same prepared splits. The app displays the model's estimated spam probability.")
        with st.container(border=True):
            st.markdown('<div class="section-label">Method</div>', unsafe_allow_html=True)
            st.subheader("Separate data partitions")
            st.markdown("Model choice used validation spam recall at a fixed 0.50 threshold. Final-test results are report-only. Nepal synthetic examples are excluded from training and selection.")
    with second:
        with st.container(border=True):
            st.markdown('<div class="section-label">Data sources</div>', unsafe_allow_html=True)
            st.markdown("- [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection), CC BY 4.0; English `ham`/`spam` messages.\n- [Bengali SMS Smishing Dataset](https://huggingface.co/datasets/shariul-islam/bengali-sms-smishing-dataset), whose card declares MIT; Bengali, English, Banglish and CodeMix categories.\n- The Bengali data includes translated or synthetic content according to its research description. Bangla is not Nepali.")
        with st.container(border=True):
            st.markdown('<div class="section-label">Known limitations</div>', unsafe_allow_html=True)
            st.markdown("The source datasets have different label meanings and language distributions. The training corpus is English and Bengali-focused and contains no Nepali-language examples. Exact duplicates were removed; template-level overlap was not assessed. The Mendeley Financial Scams dataset was not included because its data file could not be reliably accessed.")

    audit = load_json(AUDIT_PATH)
    if audit:
        with st.expander("Dataset preparation summary"):
            st.caption(
                f"{sum(audit['input_rows_by_dataset'].values()):,} source rows; "
                f"{audit['duplicate_rows_removed']:,} exact duplicate rows removed from the source data."
            )
            split_rows = []
            for split_name in ("train", "validation", "test"):
                counts = audit["label_counts"][split_name]
                split_rows.append({
                    "Split": split_name.capitalize(),
                    "Messages": audit["source_split_counts"][split_name],
                    "Ham": counts.get("ham", 0),
                    "Spam": counts.get("spam", 0),
                })
            st.dataframe(pd.DataFrame(split_rows), hide_index=True, width="stretch")
            overlaps = audit["cross_split_exact_normalized_text_overlaps"]
            if all(value == 0 for value in overlaps.values()):
                st.caption(
                    "Exact normalized-text overlap: 0 across train/validation, train/test, "
                    "and validation/test. This does not detect near-duplicate templates or related campaigns."
                )
            else:
                st.warning(f"Exact normalized-text overlaps found: {overlaps}")


def render_disclaimer():
    st.markdown(
        '<div class="disclaimer"><strong>Safety note:</strong> Use the score to prioritize review. Verify sensitive requests through official channels before taking action. This application provides message-awareness guidance and is not financial advice.</div>',
        unsafe_allow_html=True,
    )


def main():
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
              <div class="sidebar-logo" aria-hidden="true">
                <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M12 2.7 20 5.8v5.7c0 5.1-3.2 8.4-8 10.3-4.8-1.9-8-5.2-8-10.3V5.8l8-3.1Z" stroke="white" stroke-width="1.6" stroke-linejoin="round"/>
                  <path d="m8.1 12.2 2.5 2.5 5.4-5.5" stroke="#D0E3FF" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
              </div>
              <div><div class="sidebar-name">Nepal ScamShield</div><div class="sidebar-sub">SMS awareness workspace</div></div>
            </div>
            <div class="sidebar-section">Workspace</div>
            """,
            unsafe_allow_html=True,
        )
        page = st.radio(
            "Navigation",
            PAGES,
            format_func=lambda item: f":material/{NAV_ICONS[item]}:  {item}",
            label_visibility="collapsed",
            key="active_page",
        )
        st.markdown(
            '<div class="sidebar-foot"><div class="sidebar-foot-title">SMS awareness</div><div class="sidebar-foot-copy">Use the score to guide review. Verify requests through official channels.</div></div>',
            unsafe_allow_html=True,
        )

    if page == "Overview":
        render_landing()
        render_overview_snapshot()
        render_analyser(compact=True)
    elif page == "Message Analyser":
        render_analyser()
    elif page == "Model Performance":
        render_model_performance()
    elif page == "Nepal Evaluation":
        render_nepal_evaluation()
    else:
        render_about()

    st.divider()
    render_disclaimer()


main()
