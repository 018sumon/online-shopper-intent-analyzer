import sys
import streamlit as st

st.write("Python version:", sys.version)
st.write("Python executable:", sys.executable)

import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
    confusion_matrix, ConfusionMatrixDisplay
)

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Shopper Intelligence",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA_FILE = "online_shoppers_intention.csv"
MODEL_FILE = "shopper_intent_model.pkl"

NAVY = "#172554"
BLUE = "#3B82F6"
PURPLE = "#8B5CF6"
TEAL = "#0D9488"
ORANGE = "#F59E0B"


# =====================================================
# PROFESSIONAL CSS
# =====================================================
st.markdown("""
<style>
.stApp {
    background: #F4F7FC;
    color: #172554;
}

[data-testid="stHeader"] {
    background: rgba(244,247,252,0.92);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#111827,#312E81);
}

[data-testid="stSidebar"] * {
    color: #F8FAFC;
}

[data-testid="stMetric"] {
    background: white;
    border: 1px solid #E1E8F5;
    border-radius: 15px;
    padding: 18px;
    box-shadow: 0 4px 15px rgba(23,37,84,0.05);
}

[data-testid="stMetricLabel"] {
    color: #64748B;
    font-weight: 600;
}

[data-testid="stMetricValue"] {
    color: #172554;
    font-weight: 750;
}

.hero {
    position: relative;
    overflow: hidden;
    padding: 34px;
    margin: 4px 0 26px 0;
    border-radius: 22px;
    color: white;
    background:
        radial-gradient(
            circle at 85% 20%,
            rgba(34,211,238,0.38),
            transparent 28%
        ),
        radial-gradient(
            circle at 72% 100%,
            rgba(167,139,250,0.55),
            transparent 35%
        ),
        linear-gradient(120deg,#101B3D,#3730A3 60%,#6D28D9);
    box-shadow: 0 12px 32px rgba(49,46,129,0.18);
}

.hero h1 {
    color: white !important;
    font-size: 32px;
    font-weight: 800;
    line-height: 1.2;
    margin: 8px 0;
}

.hero p {
    color: #E0E7FF !important;
    font-size: 15px;
    margin: 0;
}

.hero-icon {
    font-size: 38px;
}

.hero-pill {
    display: inline-block;
    margin-top: 18px;
    padding: 7px 12px;
    border-radius: 30px;
    background: rgba(255,255,255,0.13);
    border: 1px solid rgba(255,255,255,0.22);
    color: white;
    font-size: 12px;
}

.section-title {
    color: #172554;
    font-size: 21px;
    font-weight: 750;
    margin: 12px 0;
}

[data-testid="stForm"] {
    background: white;
    padding: 20px;
    border: 1px solid #E1E8F5;
    border-radius: 16px;
}

.stButton button,
.stFormSubmitButton button,
.stDownloadButton button {
    border-radius: 10px;
    font-weight: 650;
}

div[data-testid="stAlert"] {
    border-radius: 12px;
}

div[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}

.footer {
    text-align: center;
    padding: 24px 0 8px 0;
    color: #64748B;
    font-size: 12px;
}

hr {
    border-color: #E2E8F0;
}

@media (max-width: 700px) {
    .hero {
        padding: 23px;
    }
    .hero h1 {
        font-size: 25px;
    }
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# LOAD DATA AND MODEL
# =====================================================
@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


try:
    df = load_data()
    model = load_model()
except Exception as e:
    st.error(f"File loading failed: {e}")
    st.write(
        "Check that app.py, online_shoppers_intention.csv "
        "and shopper_intent_model.pkl are in the same folder."
    )
    st.stop()


FEATURE_COLUMNS = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend"
]

if "Revenue" not in df.columns:
    st.error("Dataset-এ Revenue column পাওয়া যায়নি।")
    st.stop()

if set(df.columns) != set(FEATURE_COLUMNS + ["Revenue"]):
    st.error("Dataset columns model-এর expected columns-এর সঙ্গে মিলছে না।")
    st.stop()


# =====================================================
# HELPER FUNCTIONS
# =====================================================
def hero(title, subtitle, icon="🛍️"):
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-icon">{icon}</div>
            <h1>{title}</h1>
            <p>{subtitle}</p>
            <div class="hero-pill">
                ✦ MACHINE LEARNING &nbsp; • &nbsp; DATA INSIGHTS
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def section(title):
    st.markdown(
        f'<div class="section-title">{title}</div>',
        unsafe_allow_html=True
    )


def style_chart(ax):
    ax.set_facecolor("white")
    ax.grid(axis="y", linestyle="--", alpha=0.2)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#E2E8F0")
    ax.spines["bottom"].set_color("#E2E8F0")
    ax.tick_params(colors="#475569")
    ax.xaxis.label.set_color("#475569")
    ax.yaxis.label.set_color("#475569")
    ax.title.set_color(NAVY)


def footer():
    st.markdown("""
    <div class="footer">
        🛍️ SHOPPER INTELLIGENCE
        &nbsp; • &nbsp; ONLINE SHOPPER INTENT ANALYZER
        <br>Built with Python, Streamlit and Machine Learning
    </div>
    """, unsafe_allow_html=True)


# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:15px 0 20px">
        <div style="font-size:43px">🛍️</div>
        <h2 style="color:white;margin:5px 0">Shopper AI</h2>
        <p style="color:#C7D2FE;font-size:12px">
            INTELLIGENCE DASHBOARD
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "🏠 Overview",
            "📊 Dataset Explorer",
            "🔮 Predict Purchase",
            "📈 Model Evaluation"
        ]
    )

    st.divider()

    st.markdown("**PROJECT DETAILS**")
    st.write("🤖 Gradient Boosting")
    st.write("📁 Tabular Dataset")
    st.write(f"🧾 {len(df):,} sessions")
    st.write("🎯 Target: Revenue")

    st.divider()
    st.caption("Predictions are estimates, not guarantees.")


# =====================================================
# 1. OVERVIEW
# =====================================================
if page == "🏠 Overview":

    hero(
        "Online Shopper Intent Analyzer",
        "Turn browsing behaviour into meaningful purchase-intention insights.",
        "🛒"
    )

    revenue = df["Revenue"].astype(bool)
    total = len(df)
    buyers = int(revenue.sum())
    non_buyers = total - buyers
    purchase_rate = buyers / total * 100 if total else 0

    section("📌 Business Overview")

    a, b, c, d = st.columns(4)

    a.metric("Total Sessions", f"{total:,}")
    b.metric("Purchase Sessions", f"{buyers:,}")
    c.metric("No Purchase", f"{non_buyers:,}")
    d.metric("Purchase Rate", f"{purchase_rate:.2f}%")

    st.divider()

    left, right = st.columns(2)

    with left:
        section("🛍️ Purchase Distribution")

        fig, ax = plt.subplots(figsize=(7, 4))

        bars = ax.bar(
            ["Purchase", "No Purchase"],
            [buyers, non_buyers],
            color=[TEAL, BLUE],
            width=0.58
        )

        ax.bar_label(bars, fmt="%,d", padding=4)
        ax.set_ylabel("Sessions")
        ax.set_title("Purchase vs No Purchase")
        style_chart(ax)

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with right:
        section("📅 Monthly Session Trends")

        month_order = [
            "Feb", "Mar", "May", "June", "Jul",
            "Aug", "Sep", "Oct", "Nov", "Dec"
        ]

        counts = df["Month"].value_counts()
        months = [m for m in month_order if m in counts.index]
        months += [m for m in counts.index if m not in months]

        fig, ax = plt.subplots(figsize=(7, 4))

        monthly_values = counts.reindex(months).values

        ax.plot(
            months,
            monthly_values,
            color=PURPLE,
            linewidth=2.8,
            marker="o"
        )
        ax.fill_between(
            range(len(months)),
            monthly_values,
            alpha=0.12,
            color=PURPLE
        )

        ax.set_ylabel("Sessions")
        ax.set_title("Sessions by Month")
        ax.tick_params(axis="x", rotation=35)
        style_chart(ax)

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    section("👥 Purchase Rate by Visitor Type")

    visitor_rates = (
        df.assign(Purchase=df["Revenue"].astype(int))
        .groupby("VisitorType")["Purchase"]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(9, 4))

    palette = [BLUE, PURPLE, TEAL, ORANGE]
    bars = ax.bar(
        visitor_rates.index.astype(str),
        visitor_rates.values,
        color=palette[:len(visitor_rates)]
    )

    ax.bar_label(
        bars,
        labels=[f"{v:.1f}%" for v in visitor_rates.values],
        padding=4
    )

    ax.set_ylabel("Purchase Rate (%)")
    ax.set_title("Conversion Rate by Visitor Type")
    style_chart(ax)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.info(
        "Insight: Purchase Rate shows the percentage of sessions "
        "where Revenue=True. It does not represent revenue amount or profit."
    )


# =====================================================
# 2. DATASET EXPLORER
# =====================================================
elif page == "📊 Dataset Explorer":

    hero(
        "Dataset Explorer",
        "Explore the records and understand your dataset.",
        "📊"
    )

    section("📌 Dataset Summary")

    a, b, c, d = st.columns(4)

    a.metric("Total Rows", f"{len(df):,}")
    b.metric("Total Columns", f"{len(df.columns)}")
    c.metric("Missing Cells", f"{int(df.isna().sum().sum()):,}")
    d.metric("Duplicate Rows", f"{int(df.duplicated().sum()):,}")

    st.divider()

    section("🔎 Dataset Preview")

    max_rows = min(100, len(df))

    n_rows = st.slider(
        "Number of rows to display",
        1,
        max_rows,
        min(10, max_rows)
    )

    st.dataframe(
        df.head(n_rows),
        use_container_width=True,
        hide_index=True
    )

    section("🧩 Column Information")

    info_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": [str(x) for x in df.dtypes],
        "Missing Values": df.isna().sum().values,
        "Unique Values": [
            df[col].nunique(dropna=True) for col in df.columns
        ]
    })

    st.dataframe(info_df, use_container_width=True, hide_index=True)

    section("📈 Statistical Summary")

    st.dataframe(
        df.describe(include="all").transpose(),
        use_container_width=True
    )

    section("⬇️ Download Dataset")

    st.download_button(
        "📥 Download Dataset CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="online_shoppers_intention.csv",
        mime="text/csv",
        use_container_width=True
    )


# =====================================================
# 3. PREDICT PURCHASE
# =====================================================
elif page == "🔮 Predict Purchase":

    hero(
        "Purchase Intention Predictor",
        "Enter session information to estimate purchase probability.",
        "🔮"
    )

    st.warning(
        "This is a machine learning estimate. "
        "It does not guarantee that a visitor will make a purchase."
    )

    with st.form("prediction_form"):

        section("📄 Page Activity")

        a, b, c = st.columns(3)

        with a:
            administrative = st.number_input(
                "Administrative pages", min_value=0, value=0
            )
            administrative_duration = st.number_input(
                "Administrative duration", min_value=0.0, value=0.0
            )

        with b:
            informational = st.number_input(
                "Informational pages", min_value=0, value=0
            )
            informational_duration = st.number_input(
                "Informational duration", min_value=0.0, value=0.0
            )

        with c:
            product_related = st.number_input(
                "Product-related pages", min_value=0, value=1
            )
            product_related_duration = st.number_input(
                "Product-related duration", min_value=0.0, value=0.0
            )

        section("🌐 Session Behaviour")

        a, b, c = st.columns(3)

        with a:
            bounce_rate = st.number_input(
                "Bounce rate",
                min_value=0.0, max_value=1.0,
                value=0.02, step=0.01, format="%.4f"
            )
            exit_rate = st.number_input(
                "Exit rate",
                min_value=0.0, max_value=1.0,
                value=0.03, step=0.01, format="%.4f"
            )

        with b:
            page_values = st.number_input(
                "Page values", min_value=0.0, value=0.0
            )
            special_day = st.number_input(
                "Special day",
                min_value=0.0, max_value=1.0,
                value=0.0, step=0.1
            )

        with c:
            month = st.selectbox(
                "Month",
                df["Month"].dropna().unique().tolist()
            )
            visitor_type = st.selectbox(
                "Visitor type",
                df["VisitorType"].dropna().unique().tolist()
            )

        section("💻 Visitor Technology")

        a, b, c = st.columns(3)

        with a:
            operating_system = st.selectbox(
                "Operating system",
                sorted(df["OperatingSystems"].dropna().unique().tolist())
            )
            browser = st.selectbox(
                "Browser",
                sorted(df["Browser"].dropna().unique().tolist())
            )

        with b:
            region = st.selectbox(
                "Region",
                sorted(df["Region"].dropna().unique().tolist())
            )
            traffic_type = st.selectbox(
                "Traffic type",
                sorted(df["TrafficType"].dropna().unique().tolist())
            )

        with c:
            weekend = st.selectbox(
                "Weekend",
                [False, True],
                format_func=lambda x: "Yes" if x else "No"
            )

        submitted = st.form_submit_button(
            "🚀 Predict Purchase",
            use_container_width=True
        )

    if submitted:

        values = {
            "Administrative": administrative,
            "Administrative_Duration": administrative_duration,
            "Informational": informational,
            "Informational_Duration": informational_duration,
            "ProductRelated": product_related,
            "ProductRelated_Duration": product_related_duration,
            "BounceRates": bounce_rate,
            "ExitRates": exit_rate,
            "PageValues": page_values,
            "SpecialDay": special_day,
            "Month": month,
            "OperatingSystems": operating_system,
            "Browser": browser,
            "Region": region,
            "TrafficType": traffic_type,
            "VisitorType": visitor_type,
            "Weekend": weekend
        }

        input_df = pd.DataFrame([values])[FEATURE_COLUMNS]

        try:
            prediction = int(model.predict(input_df)[0])
            probability = float(model.predict_proba(input_df)[0][1])

            st.divider()
            section("🎯 Prediction Result")

            a, b = st.columns(2)

            a.metric("Purchase Probability", f"{probability:.2%}")
            b.metric(
                "Predicted Outcome",
                "Purchase" if prediction == 1 else "No Purchase"
            )

            st.progress(max(0.0, min(1.0, probability)))

            if prediction == 1:
                st.success("🛒 Model prediction: Purchase")
            else:
                st.info("👀 Model prediction: No Purchase")

            st.caption(
                "Important: PageValues may not be known before a purchase. "
                "For real-time use, test the model without information "
                "that becomes available only after a purchase."
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")


# =====================================================
# 4. MODEL EVALUATION
# =====================================================
elif page == "📈 Model Evaluation":

    hero(
        "Model Evaluation Center",
        "Evaluate performance and understand classification errors.",
        "📈"
    )

    st.warning(
        "This evaluation reconstructs the test split. "
        "It must match the original training split and row order "
        "to reproduce the original test metrics."
    )

    try:
        X = df.drop(columns=["Revenue"]).copy()
        y = df["Revenue"].astype(int)

        _, X_test, _, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)

        section("🎯 Performance Metrics")

        a, b, c = st.columns(3)
        a.metric("Accuracy", f"{accuracy:.2%}")
        b.metric("Precision", f"{precision:.2%}")
        c.metric("Recall", f"{recall:.2%}")

        a, b = st.columns(2)
        a.metric("F1-score", f"{f1:.2%}")
        b.metric("ROC-AUC", f"{auc:.4f}")

        st.divider()

        section("🧮 Confusion Matrix")

        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

        fig, ax = plt.subplots(figsize=(7, 5))

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["No Purchase", "Purchase"]
        )

        disp.plot(
            ax=ax,
            values_format="d",
            colorbar=False,
            cmap="Blues"
        )

        ax.set_title("Actual vs Predicted", color=NAVY)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        tn, fp, fn, tp = cm.ravel()

        a, b = st.columns(2)
        a.metric("Correct No Purchase (TN)", f"{tn:,}")
        b.metric("Correct Purchase (TP)", f"{tp:,}")

        a, b = st.columns(2)
        a.metric("False Purchase (FP)", f"{fp:,}")
        b.metric("Missed Purchase (FN)", f"{fn:,}")

        st.divider()

        section("📉 ROC Curve")

        fpr, tpr, _ = roc_curve(y_test, y_prob)

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(
            fpr, tpr,
            color=PURPLE,
            linewidth=2.7,
            label=f"Gradient Boosting (AUC = {auc:.4f})"
        )

        ax.plot(
            [0, 1], [0, 1],
            color="#94A3B8",
            linestyle="--",
            label="Random Classifier"
        )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Receiver Operating Characteristic", color=NAVY)
        ax.legend(loc="lower right")
        style_chart(ax)

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.info(
            f"ROC-AUC: {auc:.4f}. A value closer to 1 generally "
            "indicates better ranking of purchase sessions."
        )

        st.divider()

        section("📘 Metric Guide")

        with st.expander("Accuracy"):
            st.write("Percentage of all test predictions that were correct.")

        with st.expander("Precision"):
            st.write(
                "Of sessions predicted as Purchase, how many actually purchased."
            )

        with st.expander("Recall"):
            st.write(
                "Of all actual purchase sessions, how many the model detected."
            )

        with st.expander("F1-score"):
            st.write("Harmonic mean of precision and recall.")

        with st.expander("ROC-AUC"):
            st.write(
                "Measures how well the model ranks purchase sessions "
                "above non-purchase sessions across thresholds."
            )

        metrics_df = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"],
            "Score": [accuracy, precision, recall, f1, auc]
        })

        section("⬇️ Export Evaluation")

        st.download_button(
            "📥 Download Evaluation Metrics",
            data=metrics_df.to_csv(index=False).encode("utf-8"),
            file_name="model_evaluation_metrics.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.caption(
            "Caution: PageValues may contain information unavailable "
            "before purchase, so real-world pre-purchase performance "
            "should be evaluated separately."
        )

    except Exception as e:
        st.error(f"Evaluation failed: {e}")
        st.info(
            "Check the model file, dataset columns, split settings "
            "and scikit-learn compatibility."
        )


# =====================================================
# FOOTER
# =====================================================
footer()
