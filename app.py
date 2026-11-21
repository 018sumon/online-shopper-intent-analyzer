
import streamlit as st
import pandas as pd
import numpy as np
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
BLUE = "#2563EB"
PURPLE = "#7C3AED"
GREEN = "#0D9488"
CYAN = "#0891B2"
ORANGE = "#D97706"

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

# =====================================================
# CUSTOM CSS
# =====================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #F3F6FB;
    color: #172554;
}

[data-testid="stHeader"] {
    background: rgba(243,246,251,0.95);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#101B36,#292F68);
}

[data-testid="stSidebar"] * {
    color: white;
}

.block-container {
    max-width: 1500px;
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

.hero {
    background:
        radial-gradient(circle at 85% 15%,rgba(34,211,238,.30),transparent 25%),
        radial-gradient(circle at 75% 100%,rgba(167,139,250,.35),transparent 35%),
        linear-gradient(115deg,#101B36,#273779 60%,#5B35A5);
    border-radius: 22px;
    padding: 32px;
    color: white;
    margin-bottom: 25px;
    box-shadow: 0 12px 30px rgba(23,37,84,.13);
}

.hero h1 {
    color: white !important;
    font-size: 30px;
    font-weight: 800;
    margin: 8px 0;
}

.hero p {
    color: #DCE7FF !important;
    font-size: 14px;
}

.hero-icon {
    font-size: 38px;
}

.kicker {
    color: #A5F3FC;
    font-size: 10px;
    letter-spacing: 2px;
    font-weight: 800;
}

.section-title {
    color: #172554;
    font-size: 19px;
    font-weight: 800;
    margin: 12px 0;
}

.subtle {
    color: #64748B;
    font-size: 12px;
}

[data-testid="stMetric"] {
    background: white;
    border: 1px solid #E5EAF3;
    border-radius: 16px;
    padding: 17px;
    box-shadow: 0 5px 18px rgba(23,37,84,.05);
}

[data-testid="stMetricLabel"] {
    color: #64748B;
    font-weight: 600;
}

[data-testid="stMetricValue"] {
    color: #172554;
    font-weight: 800;
}

[data-testid="stForm"] {
    background: white;
    border: 1px solid #E5EAF3;
    border-radius: 18px;
    padding: 20px;
}

.stButton button,
.stFormSubmitButton button,
.stDownloadButton button {
    border-radius: 10px;
    font-weight: 700;
    min-height: 42px;
}

div[data-testid="stAlert"] {
    border-radius: 12px;
}

.footer {
    text-align: center;
    color: #94A3B8;
    font-size: 11px;
    padding: 25px 0 5px;
}

hr {
    border-color: #E2E8F0;
}

@media(max-width:700px) {
    .hero {
        padding: 22px;
    }
    .hero h1 {
        font-size: 23px;
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
    st.info(
        "Keep app.py, online_shoppers_intention.csv "
        "and shopper_intent_model.pkl in the same folder."
    )
    st.stop()


# =====================================================
# DATA VALIDATION
# =====================================================
required_columns = FEATURE_COLUMNS + ["Revenue"]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    st.error(
        "Missing dataset columns: "
        + ", ".join(missing_columns)
    )
    st.stop()


def convert_target(value):
    if isinstance(value, (bool, np.bool_)):
        return int(value)

    if isinstance(value, (int, float, np.integer, np.floating)):
        return int(value != 0)

    return int(
        str(value).strip().lower()
        in ["true", "1", "yes"]
    )


df = df.copy()
df["Revenue"] = df["Revenue"].apply(convert_target)

total_sessions = len(df)
purchase_sessions = int(df["Revenue"].sum())
no_purchase_sessions = total_sessions - purchase_sessions

purchase_rate = (
    purchase_sessions / total_sessions * 100
    if total_sessions else 0
)


# =====================================================
# HELPER FUNCTIONS
# =====================================================
def hero(title, subtitle, icon="🛍️"):
    st.markdown(f"""
    <div class="hero">
        <div class="kicker">
            SHOPPER INTELLIGENCE • DATA ANALYTICS
        </div>
        <div class="hero-icon">{icon}</div>
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def section(title, subtitle=None):
    st.markdown(
        f'<div class="section-title">{title}</div>',
        unsafe_allow_html=True
    )

    if subtitle:
        st.markdown(
            f'<div class="subtle">{subtitle}</div>',
            unsafe_allow_html=True
        )


def chart_style(ax):
    ax.set_facecolor("white")
    ax.grid(axis="y", linestyle="--", alpha=0.2)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_color("#E2E8F0")
    ax.spines["bottom"].set_color("#E2E8F0")

    ax.tick_params(colors="#475569")
    ax.set_title(ax.get_title(), color=NAVY)


def footer():
    st.markdown("""
    <div class="footer">
        🛍️ SHOPPER INTELLIGENCE
        <br>
        ONLINE SHOPPER INTENT ANALYZER
        <br>
        Built with Python, Streamlit and Machine Learning
    </div>
    """, unsafe_allow_html=True)


def get_purchase_probability(estimator, X):
    if not hasattr(estimator, "predict_proba"):
        raise ValueError(
            "The saved model does not support predict_proba."
        )

    probabilities = estimator.predict_proba(X)

    classes = list(
        getattr(estimator, "classes_", [0, 1])
    )

    if 1 in classes:
        return probabilities[:, classes.index(1)]

    return probabilities[:, -1]


# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:

    st.markdown("""
    <div style="text-align:center;padding:18px 0">
        <div style="font-size:45px">🛍️</div>
        <div style="font-size:23px;font-weight:800">
            Shopper AI
        </div>
        <div style="font-size:10px;letter-spacing:2px;color:#C7D2FE">
            ANALYTICS DASHBOARD
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "🏠 Overview",
            "📊 Dataset Explorer",
            "🔮 Purchase Predictor",
            "📈 Model Evaluation"
        ]
    )

    st.divider()

    st.markdown("**PROJECT DETAILS**")
    st.write("🤖 Gradient Boosting")
    st.write(f"📁 {total_sessions:,} Sessions")
    st.write("🎯 Target: Revenue")
    st.write("🧩 17 Input Features")

    st.divider()

    st.caption(
        "Predictions are estimates, not guarantees."
    )


# =====================================================
# 1. OVERVIEW
# =====================================================
if page == "🏠 Overview":

    hero(
        "Online Shopper Intent Analyzer",
        "Turn visitor session behaviour into meaningful purchase insights.",
        "🛒"
    )

    section(
        "Business Overview",
        "Key indicators calculated directly from your dataset."
    )

    a, b, c, d = st.columns(4)

    a.metric("Total Sessions", f"{total_sessions:,}")
    b.metric("Purchase Sessions", f"{purchase_sessions:,}")
    c.metric("No Purchase", f"{no_purchase_sessions:,}")
    d.metric("Purchase Rate", f"{purchase_rate:.2f}%")

    st.write("")

    left, right = st.columns(2, gap="large")

    # PURCHASE DISTRIBUTION
    with left:

        section(
            "Purchase Distribution",
            "Purchase versus non-purchase sessions."
        )

        fig, ax = plt.subplots(figsize=(7, 4))

        bars = ax.bar(
            ["Purchase", "No Purchase"],
            [purchase_sessions, no_purchase_sessions],
            color=[GREEN, BLUE],
            width=0.55
        )

        ax.bar_label(bars, padding=5, fmt="%,d")
        ax.set_ylabel("Number of Sessions")
        ax.set_title("Purchase vs No Purchase")

        chart_style(ax)

        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # MONTHLY TRENDS
    with right:

        section(
            "Monthly Session Trends",
            "Number of sessions recorded in each month."
        )

        month_order = [
            "Feb", "Mar", "May", "June", "Jul",
            "Aug", "Sep", "Oct", "Nov", "Dec"
        ]

        counts = df["Month"].value_counts()

        months = [
            m for m in month_order
            if m in counts.index
        ]

        months += [
            m for m in counts.index
            if m not in months
        ]

        monthly_values = counts.reindex(months).values

        fig, ax = plt.subplots(figsize=(7, 4))

        ax.plot(
            months,
            monthly_values,
            color=BLUE,
            linewidth=2.8,
            marker="o"
        )

        ax.fill_between(
            range(len(months)),
            monthly_values,
            alpha=0.12,
            color=BLUE
        )

        ax.set_ylabel("Sessions")
        ax.set_title("Sessions by Month")
        ax.tick_params(axis="x", rotation=30)

        chart_style(ax)

        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.write("")

    left, right = st.columns(2, gap="large")

    # VISITOR TYPE
    with left:

        section(
            "Purchase Rate by Visitor Type",
            "Percentage of sessions with Revenue=True."
        )

        visitor_rates = (
            df.groupby("VisitorType")["Revenue"]
            .mean()
            .mul(100)
            .sort_values(ascending=False)
        )

        fig, ax = plt.subplots(figsize=(7, 4))

        bars = ax.bar(
            visitor_rates.index.astype(str),
            visitor_rates.values,
            color=[BLUE, PURPLE, GREEN][:len(visitor_rates)]
        )

        ax.bar_label(
            bars,
            labels=[
                f"{v:.1f}%"
                for v in visitor_rates.values
            ],
            padding=4
        )

        ax.set_ylabel("Purchase Rate (%)")
        ax.set_title("Visitor Type Analysis")

        chart_style(ax)

        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # SESSION BEHAVIOUR
    with right:

        section(
            "Session Behaviour",
            "Average Bounce Rate and Exit Rate."
        )

        behaviour = pd.Series({
            "Bounce Rate": df["BounceRates"].mean() * 100,
            "Exit Rate": df["ExitRates"].mean() * 100
        })

        fig, ax = plt.subplots(figsize=(7, 4))

        bars = ax.bar(
            behaviour.index,
            behaviour.values,
            color=[CYAN, ORANGE],
            width=0.5
        )

        ax.bar_label(
            bars,
            labels=[
                f"{v:.2f}%"
                for v in behaviour.values
            ],
            padding=4
        )

        ax.set_ylabel("Average Rate (%)")
        ax.set_title("Average Session Rates")

        chart_style(ax)

        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.info(
        "Dataset limitation: this dataset contains session-level "
        "Revenue outcomes. It does not contain actual cart additions, "
        "checkout stages, order values or monetary revenue amounts. "
        "Therefore, this dashboard uses real dataset indicators "
        "instead of invented cart-abandonment or revenue figures."
    )


# =====================================================
# 2. DATASET EXPLORER
# =====================================================
elif page == "📊 Dataset Explorer":

    hero(
        "Dataset Explorer",
        "Explore records and understand your dataset.",
        "📊"
    )

    section("Dataset Summary")

    a, b, c, d = st.columns(4)

    a.metric("Total Rows", f"{len(df):,}")
    b.metric("Total Columns", f"{len(df.columns)}")
    c.metric(
        "Missing Cells",
        f"{int(df.isna().sum().sum()):,}"
    )
    d.metric(
        "Duplicate Rows",
        f"{int(df.duplicated().sum()):,}"
    )

    st.divider()

    section("Dataset Preview")

    a, b = st.columns(2)

    with a:
        target_filter = st.selectbox(
            "Filter Revenue",
            ["All", "Purchase", "No Purchase"]
        )

    with b:
        row_count = st.slider(
            "Rows to display",
            5,
            min(200, len(df)),
            min(25, len(df))
        )

    filtered_df = df.copy()

    if target_filter == "Purchase":
        filtered_df = filtered_df[
            filtered_df["Revenue"] == 1
        ]

    elif target_filter == "No Purchase":
        filtered_df = filtered_df[
            filtered_df["Revenue"] == 0
        ]

    st.dataframe(
        filtered_df.head(row_count),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    section("Column Information")

    info_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": [
            str(x) for x in df.dtypes
        ],
        "Missing Values": [
            int(df[col].isna().sum())
            for col in df.columns
        ],
        "Unique Values": [
            int(df[col].nunique())
            for col in df.columns
        ]
    })

    st.dataframe(
        info_df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    section("Statistical Summary")

    st.dataframe(
        df.describe(include="all").transpose(),
        use_container_width=True
    )

    st.download_button(
        "⬇️ Download Dataset CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="online_shoppers_intention.csv",
        mime="text/csv"
    )


# =====================================================
# 3. PURCHASE PREDICTOR
# =====================================================
elif page == "🔮 Purchase Predictor":

    hero(
        "Purchase Intention Predictor",
        "Enter visitor session information to estimate purchase probability.",
        "🔮"
    )

    st.warning(
        "Complete all 17 inputs. No prediction will be made "
        "until the required information is entered."
    )

    with st.form("prediction_form"):

        # PAGE ACTIVITY
        section("A. Page Activity")

        a, b, c = st.columns(3)

        with a:
            administrative = st.number_input(
                "Administrative Pages",
                min_value=0,
                value=None,
                step=1,
                placeholder="Enter number"
            )

            administrative_duration = st.number_input(
                "Administrative Duration",
                min_value=0.0,
                value=None,
                placeholder="Enter duration"
            )

        with b:
            informational = st.number_input(
                "Informational Pages",
                min_value=0,
                value=None,
                step=1,
                placeholder="Enter number"
            )

            informational_duration = st.number_input(
                "Informational Duration",
                min_value=0.0,
                value=None,
                placeholder="Enter duration"
            )

        with c:
            product_related = st.number_input(
                "Product Related Pages",
                min_value=0,
                value=None,
                step=1,
                placeholder="Enter number"
            )

            product_related_duration = st.number_input(
                "Product Related Duration",
                min_value=0.0,
                value=None,
                placeholder="Enter duration"
            )

        st.divider()

        # SESSION BEHAVIOUR
        section("B. Session Behaviour")

        a, b, c = st.columns(3)

        with a:
            bounce_rate = st.number_input(
                "Bounce Rate (0–1)",
                min_value=0.0,
                max_value=1.0,
                value=None,
                format="%.4f",
                placeholder="Example: 0.02"
            )

        with b:
            exit_rate = st.number_input(
                "Exit Rate (0–1)",
                min_value=0.0,
                max_value=1.0,
                value=None,
                format="%.4f",
                placeholder="Example: 0.03"
            )

        with c:
            page_values = st.number_input(
                "Page Values",
                min_value=0.0,
                value=None,
                placeholder="Enter value"
            )

        special_day = st.number_input(
            "Special Day (0–1)",
            min_value=0.0,
            max_value=1.0,
            value=None,
            step=0.1,
            placeholder="Example: 0.0"
        )

        st.divider()

        # VISITOR INFORMATION
        section("C. Visitor Information")

        a, b, c = st.columns(3)

        with a:
            month = st.selectbox(
                "Month",
                ["Choose Month"] +
                sorted(
                    df["Month"].dropna().unique().tolist()
                )
            )

        with b:
            visitor_type = st.selectbox(
                "Visitor Type",
                ["Choose Visitor Type"] +
                sorted(
                    df["VisitorType"].dropna().unique().tolist()
                )
            )

        with c:
            weekend = st.selectbox(
                "Weekend",
                ["Choose Option", "Yes", "No"]
            )

        st.divider()

        # TECHNOLOGY
        section("D. Visitor Technology")

        a, b, c, d = st.columns(4)

        with a:
            operating_system = st.selectbox(
                "Operating System",
                ["Choose OS"] +
                sorted(
                    df["OperatingSystems"].dropna().unique().tolist()
                )
            )

        with b:
            browser = st.selectbox(
                "Browser",
                ["Choose Browser"] +
                sorted(
                    df["Browser"].dropna().unique().tolist()
                )
            )

        with c:
            region = st.selectbox(
                "Region",
                ["Choose Region"] +
                sorted(
                    df["Region"].dropna().unique().tolist()
                )
            )

        with d:
            traffic_type = st.selectbox(
                "Traffic Type",
                ["Choose Traffic"] +
                sorted(
                    df["TrafficType"].dropna().unique().tolist()
                )
            )

        submitted = st.form_submit_button(
            "🚀 Predict Purchase",
            use_container_width=True
        )

    # PROCESS FORM
    if submitted:

        numeric_inputs = {
            "Administrative Pages": administrative,
            "Administrative Duration": administrative_duration,
            "Informational Pages": informational,
            "Informational Duration": informational_duration,
            "Product Related Pages": product_related,
            "Product Related Duration": product_related_duration,
            "Bounce Rate": bounce_rate,
            "Exit Rate": exit_rate,
            "Page Values": page_values,
            "Special Day": special_day
        }

        missing = [
            name
            for name, value in numeric_inputs.items()
            if value is None
        ]

        choices = {
            "Month": (month, "Choose Month"),
            "Visitor Type": (visitor_type, "Choose Visitor Type"),
            "Weekend": (weekend, "Choose Option"),
            "Operating System": (operating_system, "Choose OS"),
            "Browser": (browser, "Choose Browser"),
            "Region": (region, "Choose Region"),
            "Traffic Type": (traffic_type, "Choose Traffic")
        }

        for name, (value, placeholder) in choices.items():
            if value == placeholder:
                missing.append(name)

        if missing:

            st.error(
                "Please complete: " + ", ".join(missing)
            )

        else:

            input_values = {
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
                "Weekend": weekend == "Yes"
            }

            input_df = pd.DataFrame(
                [input_values]
            )[FEATURE_COLUMNS]

            try:

                prediction = int(
                    model.predict(input_df)[0]
                )

                probability = float(
                    get_purchase_probability(model, input_df)[0]
                )

                st.divider()

                section("Prediction Result")

                a, b = st.columns(2)

                a.metric(
                    "Purchase Probability",
                    f"{probability:.2%}"
                )

                b.metric(
                    "Predicted Outcome",
                    "Purchase" if prediction == 1
                    else "No Purchase"
                )

                st.progress(
                    max(0.0, min(1.0, probability))
                )

                if prediction == 1:
                    st.success(
                        "🛒 Model Prediction: Purchase"
                    )
                else:
                    st.info(
                        "👀 Model Prediction: No Purchase"
                    )

                st.caption(
                    "This is an estimate from the saved model. "
                    "PageValues may contain information that is not "
                    "available early in a real visitor session."
                )

            except Exception as e:
                st.error(f"Prediction failed: {e}")


# =====================================================
# 4. MODEL EVALUATION
# =====================================================
elif page == "📈 Model Evaluation":

    hero(
        "Model Evaluation Center",
        "Evaluate the saved Gradient Boosting model.",
        "📈"
    )

    st.warning(
        "The evaluation split must match your original notebook: "
        "test_size=0.2, random_state=42, stratify=y, "
        "and the same row order and preprocessing."
    )

    try:

        X = df[FEATURE_COLUMNS].copy()
        y = df["Revenue"].astype(int)

        _, X_test, _, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        y_pred = model.predict(X_test)

        y_prob = get_purchase_probability(
            model,
            X_test
        )

        accuracy = accuracy_score(y_test, y_pred)

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )

        auc = roc_auc_score(
            y_test,
            y_prob
        )

        section("Model Performance Metrics")

        a, b, c, d, e = st.columns(5)

        a.metric("Accuracy", f"{accuracy:.2%}")
        b.metric("Precision", f"{precision:.2%}")
        c.metric("Recall", f"{recall:.2%}")
        d.metric("F1-score", f"{f1:.2%}")
        e.metric("ROC-AUC", f"{auc:.4f}")

        st.divider()

        left, right = st.columns(2, gap="large")

        # CONFUSION MATRIX
        with left:

            section("Confusion Matrix")

            cm = confusion_matrix(
                y_test,
                y_pred,
                labels=[0, 1]
            )

            fig, ax = plt.subplots(figsize=(6, 5))

            display = ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=[
                    "No Purchase",
                    "Purchase"
                ]
            )

            display.plot(
                ax=ax,
                values_format="d",
                colorbar=False,
                cmap="Blues"
            )

            ax.set_title(
                "Actual vs Predicted",
                color=NAVY
            )

            fig.tight_layout()

            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            tn, fp, fn, tp = cm.ravel()

            st.metric("True Negatives", f"{tn:,}")
            st.metric("False Positives", f"{fp:,}")
            st.metric("False Negatives", f"{fn:,}")
            st.metric("True Positives", f"{tp:,}")

        # ROC CURVE
        with right:

            section("ROC Curve")

            fpr, tpr, _ = roc_curve(
                y_test,
                y_prob
            )

            fig, ax = plt.subplots(figsize=(6, 5))

            ax.plot(
                fpr,
                tpr,
                color=PURPLE,
                linewidth=2.7,
                label=f"Gradient Boosting (AUC={auc:.4f})"
            )

            ax.plot(
                [0, 1],
                [0, 1],
                color="#94A3B8",
                linestyle="--",
                label="Random Classifier"
            )

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.05)

            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("Receiver Operating Characteristic")

            ax.legend(loc="lower right")

            chart_style(ax)

            fig.tight_layout()

            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            st.info(
                "ROC-AUC measures how well the model ranks "
                "purchase sessions above non-purchase sessions."
            )

        st.divider()

        section("Metric Guide")

        with st.expander("Accuracy"):
            st.write(
                "Percentage of all test predictions that were correct."
            )

        with st.expander("Precision"):
            st.write(
                "Of sessions predicted as Purchase, "
                "how many actually purchased."
            )

        with st.expander("Recall"):
            st.write(
                "Of all actual purchase sessions, "
                "how many the model detected."
            )

        with st.expander("F1-score"):
            st.write(
                "Harmonic mean of precision and recall."
            )

        with st.expander("ROC-AUC"):
            st.write(
                "Measures how well the model ranks "
                "the two classes across thresholds."
            )

        metrics_df = pd.DataFrame({
            "Metric": [
                "Accuracy",
                "Precision",
                "Recall",
                "F1-score",
                "ROC-AUC"
            ],
            "Score": [
                accuracy,
                precision,
                recall,
                f1,
                auc
            ]
        })

        st.download_button(
            "⬇️ Download Evaluation Metrics",
            data=metrics_df.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="model_evaluation_metrics.csv",
            mime="text/csv"
        )

        st.caption(
            "Important: PageValues may contain information "
            "unavailable before purchase. Test a separate model "
            "without PageValues for pre-purchase prediction."
        )

    except Exception as e:

        st.error(f"Evaluation failed: {e}")

        st.info(
            "Check your model file, dataset columns, "
            "training split and preprocessing."
        )


# =====================================================
# FOOTER
# =====================================================
footer()
