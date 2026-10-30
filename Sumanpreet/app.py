import os
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score

st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="centered")

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "creditcard.csv")

# ---------- Styling ----------
st.markdown("""
<style>
.hero {background: linear-gradient(135deg, #1e3a8a, #3b82f6);
       padding: 1.8rem 2rem; border-radius: 16px; margin-bottom: 1.5rem;}
.hero h1 {margin: 0; font-size: 2rem; color: white;}
.hero p {margin: .4rem 0 0; color: #dbeafe;}
.result {padding: 1.2rem 1.5rem; border-radius: 14px; border: 1px solid; margin-top: 1rem;}
.result h2 {margin: 0;}
.ok {background: rgba(34,197,94,.12); border-color: rgba(34,197,94,.55);}
.bad {background: rgba(239,68,68,.12); border-color: rgba(239,68,68,.55);}
</style>
""", unsafe_allow_html=True)


# ---------- Train the model (runs once, then cached) ----------
@st.cache_resource
def train():
    data = pd.read_csv(CSV_PATH)
    X = data.drop("Class", axis=1)
    y = data["Class"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    scores = {
        "Accuracy": f"{accuracy_score(y_test, y_pred) * 100:.1f}%",
        "ROC-AUC": f"{roc_auc_score(y_test, y_prob):.4f}",
        "Precision (Fraud)": f"{precision_score(y_test, y_pred) * 100:.1f}%",
        "Recall (Fraud)": f"{recall_score(y_test, y_pred) * 100:.1f}%",
    }
    return data, X.columns, scaler, model, scores


data, features, scaler, model, scores = train()


def check(index):
    """Run the trained model on one transaction (a row of the dataset)."""
    row = data.iloc[[index]]
    prob = model.predict_proba(scaler.transform(row[features]))[0, 1]
    st.session_state["result"] = {
        "index": index,
        "amount": float(row["Amount"].iloc[0]),
        "time": int(row["Time"].iloc[0]),
        "actual": int(row["Class"].iloc[0]),
        "prob": float(prob),
    }


# ---------- Header ----------
st.markdown("""
<div class="hero">
  <h1>💳 Credit Card Fraud Detection System</h1>
  <p>A machine learning approach to securing financial transactions</p>
</div>
""", unsafe_allow_html=True)

st.write("Pick a transaction from the dataset. The trained model checks it and says "
         "whether it looks **Normal** or **Fraud**.")

# ---------- Choose a transaction ----------
b1, b2, b3 = st.columns(3)
if b1.button("🎲 Random transaction", use_container_width=True):
    check(int(data.sample(1).index[0]))
if b2.button("🟢 Random normal", use_container_width=True):
    check(int(data[data["Class"] == 0].sample(1).index[0]))
if b3.button("🔴 Random fraud", use_container_width=True):
    check(int(data[data["Class"] == 1].sample(1).index[0]))

c1, c2 = st.columns([3, 1])
number = c1.number_input("Or enter a transaction number (0 to %d)" % (len(data) - 1),
                         min_value=0, max_value=len(data) - 1, value=0, step=1)
c2.write("")
c2.write("")
if c2.button("Check", type="primary", use_container_width=True):
    check(int(number))

# ---------- Result ----------
res = st.session_state.get("result")
if res:
    is_fraud = res["prob"] >= 0.5
    css, icon, title = ("bad", "🚨", "Fraud Transaction Detected") if is_fraud \
        else ("ok", "✅", "Normal Transaction")
    st.markdown(f'<div class="result {css}"><h2>{icon} {title}</h2></div>',
                unsafe_allow_html=True)
    st.write("")
    m1, m2, m3 = st.columns(3)
    m1.metric("Transaction no.", res["index"])
    m2.metric("Amount", f"{res['amount']:.2f}")
    m3.metric("Fraud probability", f"{res['prob'] * 100:.1f}%")
    st.progress(res["prob"])
    actual = "Fraud" if res["actual"] == 1 else "Normal"
    if (res["actual"] == 1) == is_fraud:
        st.success(f"The model is correct. The real label in the dataset is **{actual}**.")
    else:
        st.warning(f"The model made a mistake. The real label in the dataset is **{actual}**. "
                   "No model is perfect, and this is why the report lists recall as a limitation.")

# ---------- Model performance ----------
st.divider()
st.subheader("Model performance (on 200 unseen test transactions)")
cols = st.columns(4)
for col, (name, value) in zip(cols, scores.items()):
    col.metric(name, value)

st.caption("Model: Logistic Regression · Data: 1,000 transactions "
           f"({(data['Class'] == 0).sum()} normal, {(data['Class'] == 1).sum()} fraud) · "
           "Internship Project, CT University, Ludhiana")