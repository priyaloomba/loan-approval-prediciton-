"""Loan Approval Predictor - a Streamlit app powered by an SVM classifier."""

import pandas as pd
import streamlit as st
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦", layout="wide")

DATA_PATH = "loan_approval.csv"
FEATURES = ["income", "credit_score", "loan_amount", "years_working", "employment"]

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Plus Jakarta Sans', sans-serif; }
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; }
.block-container { padding-top: 1.5rem; max-width: 1150px; }

.hero {
  background: linear-gradient(120deg, #0B2540 0%, #0E4D64 60%, #0E7C7B 100%);
  border-radius: 20px; padding: 2.4rem 2.6rem; color: #fff; margin-bottom: 1.4rem;
}
.hero h1 { font-size: 2.3rem; font-weight: 800; margin: 0 0 .4rem 0; color: #fff; line-height: 1.15; }
.hero p { font-size: 1.02rem; margin: 0; max-width: 620px; color: #D5E6EE; line-height: 1.6; }
.chips { margin-top: 1.1rem; display: flex; gap: .5rem; flex-wrap: wrap; }
.chip { background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.25);
  padding: .25rem .8rem; border-radius: 999px; font-size: .82rem; font-weight: 600; }

.kpi { background: #fff; border: 1px solid #DCE5EE; border-radius: 14px; padding: 1rem 1.2rem; }
.kpi .v { font-size: 1.7rem; font-weight: 800; color: #0B2540; }
.kpi .l { font-size: .85rem; color: #5B6F84; font-weight: 500; }

.result { border-radius: 16px; padding: 1.5rem 1.7rem; border: 1px solid; }
.result.ok { background: #E8F6F1; border-color: #9AD6C2; }
.result.no { background: #FCEEEC; border-color: #EDB3AD; }
.result .t { font-size: 1.6rem; font-weight: 800; }
.result.ok .t { color: #0A6B56; } .result.no .t { color: #B03A2E; }
.meter { height: 10px; background: #fff; border-radius: 99px; overflow: hidden; margin: .8rem 0 .3rem; }
.meter > div { height: 100%; border-radius: 99px; }
.small { font-size: .85rem; color: #5B6F84; }
div[data-testid="stForm"] { background: #fff; border: 1px solid #DCE5EE; border-radius: 16px; padding: 1.3rem; }
.stButton > button, .stFormSubmitButton > button { border-radius: 10px; font-weight: 700; }
</style>
"""


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)

    # Clean income values such as "13,80,000"
    df["income"] = pd.to_numeric(df["income"].astype(str).str.replace(",", "", regex=False), errors="coerce")

    # Credit scores above 900 are invalid
    df.loc[df["credit_score"] > 900, "credit_score"] = None

    # Normalise and encode employment type
    df["employment"] = df["employment"].astype(str).str.strip().str.capitalize()
    df["employment"] = df["employment"].replace({"Self employed": "Self-employed"})
    df["employment"] = df["employment"].map({"Salaried": 1, "Self-employed": 0})

    df["loan_status"] = df["loan_status"].astype(str).str.strip().str.capitalize()
    df = df.drop_duplicates().dropna(subset=["loan_status"])
    return df


@st.cache_resource
def train_model():
    df = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df["loan_status"], test_size=0.2, random_state=42, stratify=df["loan_status"]
    )
    # Imputation and scaling are fitted on training data only
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("svm", CalibratedClassifierCV(SVC(kernel="rbf", C=1), method="sigmoid", cv=5, ensemble=False)),
    ])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    labels = ["Approved", "Rejected"]
    cm = pd.DataFrame(
        confusion_matrix(y_test, preds, labels=labels),
        index=[f"Actual {l}" for l in labels],
        columns=[f"Predicted {l}" for l in labels],
    )
    return pipe, accuracy_score(y_test, preds), cm


def kpi(value, label):
    st.markdown(f'<div class="kpi"><div class="v">{value}</div><div class="l">{label}</div></div>',
                unsafe_allow_html=True)


def predict_tab(model):
    left, right = st.columns([1.1, 1], gap="large")

    with left:
        with st.form("loan_form"):
            st.markdown("#### Applicant details")
            c1, c2 = st.columns(2)
            income = c1.number_input("Annual income (₹)", 0, 100_000_000, 800_000, 10_000)
            loan_amount = c2.number_input("Loan amount (₹)", 0, 100_000_000, 300_000, 10_000)
            credit_score = st.slider("Credit score", 300, 900, 760)
            c3, c4 = st.columns(2)
            years = c3.number_input("Years of experience", 0.0, 60.0, 6.0, 1.0)
            emp = c4.selectbox("Employment type", ["Salaried", "Self-employed"])
            submitted = st.form_submit_button("Check loan status", type="primary", width="stretch")

    with right:
        if not submitted:
            st.markdown(
                '<div class="kpi"><div class="v">Your result appears here</div>'
                '<div class="l">Fill in the applicant details and select <b>Check loan status</b>.</div></div>',
                unsafe_allow_html=True,
            )
            return

        applicant = pd.DataFrame([{
            "income": income, "credit_score": credit_score, "loan_amount": loan_amount,
            "years_working": years, "employment": 1 if emp == "Salaried" else 0,
        }])
        proba = model.predict_proba(applicant)[0]
        p = dict(zip(model.classes_, proba))
        approved = p["Approved"] >= 0.5
        conf = p["Approved"] if approved else p["Rejected"]
        cls, title, color = ("ok", "Likely approved", "#0E9F7E") if approved else ("no", "Likely rejected", "#D9534F")

        st.markdown(
            f'<div class="result {cls}"><div class="t">{title}</div>'
            f'<div class="small">Model confidence: <b>{conf:.0%}</b></div>'
            f'<div class="meter"><div style="width:{p["Approved"]:.0%};background:{color}"></div></div>'
            f'<div class="small">Approval probability: {p["Approved"]:.0%}</div></div>',
            unsafe_allow_html=True,
        )

        ratio = loan_amount / income if income else float("inf")
        m1, m2 = st.columns(2)
        with m1:
            kpi("—" if income == 0 else f"{ratio:.0%}", "Loan as share of income")
        with m2:
            kpi(credit_score, "Credit score")
        st.caption("Educational prediction only. It is not a bank decision or financial advice.")


def insights_tab(df, accuracy, cm):
    st.markdown("#### Dataset at a glance")
    c1, c2, c3 = st.columns(3)
    with c1: kpi(len(df), "Records after cleaning")
    with c2: kpi(f"{(df['loan_status'] == 'Approved').mean():.0%}", "Approval rate in data")
    with c3: kpi(f"{accuracy:.1%}", "Test accuracy")

    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("##### Approved vs rejected")
        st.bar_chart(df["loan_status"].value_counts(), color="#0E7C7B")
    with b:
        st.markdown("##### Average by outcome")
        st.dataframe(
            df.groupby("loan_status")[["income", "credit_score", "loan_amount", "years_working"]]
            .mean().round(0).astype(int),
            width="stretch",
        )
    st.markdown("##### Confusion matrix (test set)")
    st.dataframe(cm, width="stretch")
    with st.expander("Preview cleaned data"):
        st.dataframe(df.head(15), width="stretch")


def about_tab():
    st.markdown("""
#### How it works
1. **Clean** the CSV: fix comma-formatted income, drop invalid credit scores and duplicates, standardise employment labels.
2. **Split** the data 80/20 into training and test sets, stratified by outcome.
3. **Preprocess**: fill missing values with the training median, then scale features.
4. **Train** a Support Vector Machine with an RBF kernel.
5. **Predict** the outcome and probability for a new applicant.

#### Limitations
The dataset is small, so accuracy can vary and does not reflect real lending performance.
Real lenders weigh many more factors. Use this project to learn, not to make financial decisions.
""")


def main():
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        '<div class="hero"><h1>Loan Approval Predictor</h1>'
        '<p>Enter an applicant\'s income, credit score and work history to see how a machine-learning '
        'model would judge the loan application.</p>'
        '<div class="chips"><span class="chip">Support Vector Machine</span></div></div>',
        unsafe_allow_html=True,
    )

    try:
        model, accuracy, cm = train_model()
        df = load_data()
    except FileNotFoundError:
        st.error("Dataset not found. Keep `loan_approval.csv` in the same folder as `app.py`.")
        st.stop()
    except Exception as error:
        st.error(f"Could not train the model: {error}")
        st.stop()

    tab1, tab2, tab3 = st.tabs(["Predict", "Model insights", "About"])
    with tab1:
        predict_tab(model)
    with tab2:
        insights_tab(df, accuracy, cm)
    with tab3:
        about_tab()


if __name__ == "__main__":
    main()