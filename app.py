from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor


# =============================================================================
# 0. PAGE CONFIGURATION
# =============================================================================
APP_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = APP_DIR / "data" / "CartIQ_Clean.csv"
RANDOM_STATE = 42

st.set_page_config(
    page_title="CartIQ | GCC E-commerce Analytics",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .stApp {background: linear-gradient(180deg, #f7f9fc 0%, #ffffff 40%);}
        [data-testid="stSidebar"] {background: #111827;}
        [data-testid="stSidebar"] * {color: #f9fafb;}
        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 14px;
            padding: 14px;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.05);
        }
        div[data-testid="stExpander"] {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
        }
        .hero {
            background: linear-gradient(115deg, #111827 0%, #1f4d78 58%, #0f766e 100%);
            border-radius: 20px;
            padding: 28px 30px;
            color: white;
            margin-bottom: 20px;
            box-shadow: 0 12px 30px rgba(15, 23, 42, .18);
        }
        .hero h1 {margin: 0 0 6px 0; font-size: 2.2rem;}
        .hero p {margin: 0; opacity: .90; font-size: 1.02rem;}
        .insight-card {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-left: 5px solid #0f766e;
            border-radius: 12px;
            padding: 16px 18px;
            margin: 8px 0;
        }
        .small-note {color: #64748b; font-size: .88rem;}
        .block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# 1. DATA GENERATION, LOADING, VALIDATION, AND CLEANING
# =============================================================================
PRODUCTS_BY_CATEGORY: Dict[str, List[str]] = {
    "Electronics": ["Wireless Earbuds", "Power Bank", "Phone Case", "Smart Watch"],
    "Fashion": ["Sneakers", "T-Shirt", "Handbag", "Sunglasses"],
    "Beauty": ["Face Serum", "Sunscreen", "Lip Tint", "Cleanser"],
    "Home & Living": ["Scented Candle", "Storage Box", "Bedsheet", "Desk Lamp"],
}

REQUIRED_BASE_COLUMNS = {
    "device_category",
    "user_region",
    "traffic_source",
    "payment_method_selected",
    "user_tier",
    "primary_product_category",
    "past_purchases_count",
    "items_in_cart",
    "average_item_price",
    "session_duration_secs",
    "checkout_status",
    "historical_clv",
}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


@st.cache_data(show_spinner=False)
def generate_cartiQ_data(n_rows: int = 2200, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Generate reproducible GCC e-commerce journey data with realistic relationships."""
    rng = np.random.default_rng(seed)

    devices = rng.choice(
        ["Mobile", "Desktop", "MOBILE", "desktop ", "Tablet"],
        size=n_rows,
        p=[0.46, 0.22, 0.14, 0.08, 0.10],
    )
    regions = rng.choice(
        [" Dubai", "Dubai", "Abu Dhabi", "Sharjah", " Riyadh", "Riyadh", "Jeddah"],
        size=n_rows,
        p=[0.08, 0.26, 0.16, 0.12, 0.07, 0.20, 0.11],
    )
    traffic = rng.choice(
        ["Google Shopping", "Instagram Ads", "Direct", "TikTok Shop", "Email"],
        size=n_rows,
        p=[0.25, 0.25, 0.20, 0.18, 0.12],
    )
    payment = rng.choice(
        ["Credit Card", "Cash on Delivery", "Apple Pay", "Tabby/Tamara"],
        size=n_rows,
        p=[0.35, 0.22, 0.22, 0.21],
    )
    tier = rng.choice(["Guest", "Logged-In Member"], size=n_rows, p=[0.55, 0.45])
    category = rng.choice(list(PRODUCTS_BY_CATEGORY), size=n_rows, p=[0.28, 0.26, 0.24, 0.22])

    past_purchases = rng.poisson(lam=np.where(tier == "Logged-In Member", 4.3, 1.6), size=n_rows)
    items_in_cart = np.clip(rng.poisson(lam=3.3, size=n_rows) + 1, 1, 11)
    avg_item_price = np.round(rng.lognormal(mean=4.55, sigma=0.72, size=n_rows), 2)
    avg_item_price = np.clip(avg_item_price, 12, 650)
    total_cart_value = np.round(items_in_cart * avg_item_price, 2)

    pages_viewed = np.clip(items_in_cart * 2 + rng.poisson(4, size=n_rows), 2, 40)
    items_added_count = np.clip(items_in_cart + rng.binomial(2, 0.35, size=n_rows), 1, 13)
    session_duration = np.round(
        80 + pages_viewed * rng.uniform(18, 42, size=n_rows) + rng.normal(0, 55, size=n_rows),
        1,
    )
    session_duration = np.clip(session_duration, 45, 2400)

    product_viewed = np.ones(n_rows, dtype=int)
    cart_added = (items_in_cart >= 1).astype(int)
    checkout_started_prob = _sigmoid(-1.4 + 0.006 * session_duration + 0.18 * items_in_cart)
    checkout_started = rng.binomial(1, checkout_started_prob)

    member = (tier == "Logged-In Member").astype(int)
    direct = (traffic == "Direct").astype(int)
    email = (traffic == "Email").astype(int)
    mobile = np.char.upper(np.char.strip(devices.astype(str))) == "MOBILE"
    cod = payment == "Cash on Delivery"
    bnpl = payment == "Tabby/Tamara"
    apple = payment == "Apple Pay"

    shipping_prob = _sigmoid(
        -0.75
        + 1.8 * checkout_started
        + 0.25 * member
        + 0.20 * direct
        + 0.12 * email
        - 0.10 * mobile
        - 0.00018 * total_cart_value
    )
    shipping_details_entered = rng.binomial(1, shipping_prob)

    payment_selected_prob = _sigmoid(
        -0.65
        + 2.0 * shipping_details_entered
        + 0.25 * apple
        + 0.20 * bnpl
        - 0.18 * cod
    )
    payment_stage_completed = rng.binomial(1, payment_selected_prob)

    # Target depends on pre-purchase behaviour and payment preference.
    conversion_logit = (
        -2.2
        + 0.0032 * session_duration
        + 0.11 * items_in_cart
        + 0.16 * np.log1p(past_purchases)
        + 0.40 * member
        + 0.28 * direct
        + 0.18 * email
        + 0.35 * bnpl
        + 0.28 * apple
        - 0.35 * cod
        - 0.00028 * total_cart_value
        + 0.65 * checkout_started
    )
    checkout_probability = _sigmoid(conversion_logit)
    checkout_status = rng.binomial(1, checkout_probability)

    # Downstream fields intentionally contain structural nulls.
    days_to_delivery = np.where(
        checkout_status == 1,
        np.round(np.clip(rng.normal(3.2, 1.0, n_rows), 1, 8), 1),
        np.nan,
    )
    delivery_rating = np.where(
        checkout_status == 1,
        rng.choice([2, 3, 4, 5], size=n_rows, p=[0.05, 0.17, 0.43, 0.35]),
        np.nan,
    )

    historical_clv = np.round(
        85 + past_purchases * rng.uniform(105, 190, size=n_rows) + member * 120 + rng.normal(0, 65, n_rows),
        2,
    )
    historical_clv = np.clip(historical_clv, 0, None)
    missing_clv_rows = rng.choice(n_rows, max(10, int(n_rows * 0.008)), replace=False)
    historical_clv[missing_clv_rows] = np.nan

    abandoned = checkout_status == 0
    coupon_discount = rng.choice([0, 5, 10, 15, 20], size=n_rows, p=[0.18, 0.15, 0.28, 0.24, 0.15])
    email_delay_hours = rng.choice([1, 4, 12, 24, 48], size=n_rows, p=[0.15, 0.26, 0.25, 0.21, 0.13])
    email_open_prob = _sigmoid(-0.5 + 0.0007 * np.nan_to_num(historical_clv, nan=250) - 0.018 * email_delay_hours)
    recovery_email_opened = np.where(abandoned, rng.binomial(1, email_open_prob), 0)
    recovery_prob = _sigmoid(
        -2.0
        + 1.1 * recovery_email_opened
        + 0.07 * coupon_discount
        + 0.00045 * np.nan_to_num(historical_clv, nan=250)
        - 0.016 * email_delay_hours
        + 0.25 * member
    )
    recovery_email_conversion = np.where(abandoned, rng.binomial(1, recovery_prob), 0)

    basket_items: List[str] = []
    for primary_cat, item_count in zip(category, items_in_cart):
        primary_pool = PRODUCTS_BY_CATEGORY[primary_cat]
        n_unique = int(np.clip(np.ceil(item_count / 2), 1, 5))
        selected = list(rng.choice(primary_pool, size=min(n_unique, len(primary_pool)), replace=False))

        # Add deliberately plausible cross-sell affinities.
        if "Phone Case" in selected and rng.random() < 0.60:
            selected.append("Power Bank")
        if "Face Serum" in selected and rng.random() < 0.58:
            selected.append("Sunscreen")
        if "Sneakers" in selected and rng.random() < 0.42:
            selected.append("Sunglasses")
        if "Scented Candle" in selected and rng.random() < 0.45:
            selected.append("Storage Box")

        if rng.random() < 0.25:
            other_categories = [c for c in PRODUCTS_BY_CATEGORY if c != primary_cat]
            other_cat = rng.choice(other_categories)
            selected.append(rng.choice(PRODUCTS_BY_CATEGORY[other_cat]))
        basket_items.append("|".join(dict.fromkeys(selected)))

    df = pd.DataFrame(
        {
            "session_id": [f"CIQ-{i + 1:05d}" for i in range(n_rows)],
            "device_category": devices,
            "user_region": regions,
            "traffic_source": traffic,
            "payment_method_selected": payment,
            "user_tier": tier,
            "primary_product_category": category,
            "past_purchases_count": past_purchases,
            "items_in_cart": items_in_cart,
            "average_item_price": avg_item_price,
            "session_duration_secs": session_duration,
            "pages_viewed": pages_viewed,
            "items_added_count": items_added_count,
            "total_cart_value_usd": total_cart_value,
            "product_viewed": product_viewed,
            "cart_added": cart_added,
            "checkout_started": checkout_started,
            "shipping_details_entered": shipping_details_entered,
            "payment_stage_completed": payment_stage_completed,
            "checkout_status": checkout_status,
            "days_to_delivery": days_to_delivery,
            "delivery_satisfaction_rating": delivery_rating,
            "historical_clv": historical_clv,
            "coupon_discount_pct_offered": coupon_discount,
            "recovery_email_delay_hours": email_delay_hours,
            "recovery_email_opened": recovery_email_opened,
            "recovery_email_conversion": recovery_email_conversion,
            "basket_items": basket_items,
        }
    )
    return df


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = (
        out.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )
    return out


def clean_pipeline(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return a cleaned dataframe and an auditable cleaning log."""
    cleaned = standardize_column_names(df)
    log_rows: List[Dict[str, object]] = []

    before_rows = len(cleaned)
    duplicate_count = int(cleaned.duplicated().sum())
    cleaned = cleaned.drop_duplicates().copy()
    log_rows.append(
        {
            "Step": "Remove exact duplicate rows",
            "Affected rows": duplicate_count,
            "Result": f"{before_rows:,} → {len(cleaned):,} rows",
        }
    )

    for col in [
        "device_category",
        "user_region",
        "traffic_source",
        "payment_method_selected",
        "user_tier",
        "primary_product_category",
    ]:
        if col in cleaned.columns:
            before_unique = cleaned[col].nunique(dropna=True)
            cleaned[col] = cleaned[col].astype("string").str.strip()
            if col == "device_category":
                cleaned[col] = cleaned[col].str.upper()
            after_unique = cleaned[col].nunique(dropna=True)
            log_rows.append(
                {
                    "Step": f"Standardize {col}",
                    "Affected rows": int(cleaned[col].notna().sum()),
                    "Result": f"{before_unique} → {after_unique} unique values",
                }
            )

    numeric_candidates = [
        "past_purchases_count",
        "items_in_cart",
        "average_item_price",
        "session_duration_secs",
        "checkout_status",
        "historical_clv",
        "pages_viewed",
        "items_added_count",
        "total_cart_value_usd",
        "shipping_details_entered",
        "payment_stage_completed",
        "recovery_email_conversion",
        "recovery_email_opened",
        "coupon_discount_pct_offered",
        "recovery_email_delay_hours",
    ]
    for col in numeric_candidates:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    if "historical_clv" in cleaned.columns:
        missing_before = int(cleaned["historical_clv"].isna().sum())
        median_clv = cleaned["historical_clv"].median()
        if pd.notna(median_clv):
            cleaned["historical_clv"] = cleaned["historical_clv"].fillna(median_clv)
        log_rows.append(
            {
                "Step": "Median-impute historical_clv",
                "Affected rows": missing_before,
                "Result": f"Median used: {median_clv:,.2f}" if pd.notna(median_clv) else "No valid median",
            }
        )

    # Derive fields when an uploaded file contains the base measures but not engineered ones.
    if {"items_in_cart", "average_item_price"}.issubset(cleaned.columns):
        if "total_cart_value_usd" not in cleaned.columns:
            cleaned["total_cart_value_usd"] = cleaned["items_in_cart"] * cleaned["average_item_price"]
            log_rows.append(
                {
                    "Step": "Engineer total_cart_value_usd",
                    "Affected rows": len(cleaned),
                    "Result": "items_in_cart × average_item_price",
                }
            )

    if "pages_viewed" not in cleaned.columns and "items_in_cart" in cleaned.columns:
        cleaned["pages_viewed"] = cleaned["items_in_cart"] * 2
    if "items_added_count" not in cleaned.columns and "items_in_cart" in cleaned.columns:
        cleaned["items_added_count"] = cleaned["items_in_cart"]

    return cleaned.reset_index(drop=True), pd.DataFrame(log_rows)


@st.cache_data(show_spinner=False)
def read_csv_bytes(file_bytes: bytes) -> pd.DataFrame:
    from io import BytesIO

    return pd.read_csv(BytesIO(file_bytes))


@st.cache_data(show_spinner=False)
def read_local_csv(path_str: str) -> pd.DataFrame:
    return pd.read_csv(path_str)


def load_source_data(uploaded_file) -> Tuple[pd.DataFrame, str]:
    if uploaded_file is not None:
        try:
            return read_csv_bytes(uploaded_file.getvalue()), f"Uploaded file: {uploaded_file.name}"
        except Exception as exc:
            st.error(f"The uploaded CSV could not be read: {exc}")
            st.stop()

    if DEFAULT_DATA_PATH.exists():
        try:
            return read_local_csv(str(DEFAULT_DATA_PATH)), "Bundled CartIQ_Clean.csv"
        except Exception as exc:
            st.warning(f"Bundled CSV could not be read ({exc}). Using generated fallback data.")

    return generate_cartiQ_data(), "Generated fallback dataset"


def missing_columns(df: pd.DataFrame, columns: Iterable[str]) -> List[str]:
    return sorted(set(columns).difference(df.columns))


def require_columns(df: pd.DataFrame, columns: Iterable[str], section_name: str) -> None:
    missing = missing_columns(df, columns)
    if missing:
        st.error(
            f"{section_name} cannot run because these columns are missing: "
            + ", ".join(f"`{c}`" for c in missing)
        )
        st.stop()


# =============================================================================
# 2. MODEL HELPERS
# =============================================================================
def safe_binary_metrics(y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    result = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }
    try:
        result["ROC-AUC"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        result["ROC-AUC"] = np.nan
    return result


def classification_models() -> Dict[str, object]:
    return {
        "K-Nearest Neighbours": Pipeline(
            [("scale", StandardScaler()), ("model", KNeighborsClassifier(n_neighbors=11))]
        ),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=12, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=8,
            min_samples_leaf=6,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE,
        ),
    }


@st.cache_data(show_spinner=False)
def train_classification_suite(
    data: pd.DataFrame,
    feature_columns: Tuple[str, ...],
    target_column: str,
) -> Tuple[pd.DataFrame, Dict[str, dict]]:
    model_df = data[list(feature_columns) + [target_column]].dropna().copy()
    X = model_df[list(feature_columns)]
    y = model_df[target_column].astype(int)

    if y.nunique() < 2:
        raise ValueError("The target contains only one class after filtering.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    comparison_rows = []
    artifacts: Dict[str, dict] = {}
    for name, model in classification_models().items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X_test)[:, 1]
        else:
            prob = pred.astype(float)

        metrics = safe_binary_metrics(y_test, pred, prob)
        comparison_rows.append({"Model": name, **metrics})
        artifacts[name] = {
            "model": model,
            "X_test": X_test,
            "y_test": y_test,
            "pred": pred,
            "prob": prob,
            "features": list(feature_columns),
        }

    comparison = pd.DataFrame(comparison_rows).sort_values("ROC-AUC", ascending=False)
    return comparison, artifacts


def extract_feature_importance(model: object, feature_names: List[str]) -> pd.DataFrame:
    inner_model = model.named_steps.get("model") if isinstance(model, Pipeline) else model
    if hasattr(inner_model, "feature_importances_"):
        values = inner_model.feature_importances_
    elif hasattr(inner_model, "coef_"):
        values = np.ravel(inner_model.coef_)
    else:
        return pd.DataFrame()
    return (
        pd.DataFrame({"Feature": feature_names, "Importance": values})
        .assign(Absolute=lambda d: d["Importance"].abs())
        .sort_values("Absolute", ascending=True)
    )


@st.cache_data(show_spinner=False)
def train_regression_suite(
    data: pd.DataFrame,
    feature_columns: Tuple[str, ...],
    target_column: str,
    alpha: float,
) -> Tuple[pd.DataFrame, Dict[str, dict]]:
    model_df = data[list(feature_columns) + [target_column]].dropna().copy()
    X = model_df[list(feature_columns)]
    y = model_df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
    )

    models = {
        "Linear Regression": Pipeline([("scale", StandardScaler()), ("model", LinearRegression())]),
        "Ridge Regression": Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=alpha))]),
        "Lasso Regression": Pipeline(
            [("scale", StandardScaler()), ("model", Lasso(alpha=alpha, max_iter=20000))]
        ),
        "Decision Tree Regression": DecisionTreeRegressor(
            max_depth=7, min_samples_leaf=10, random_state=RANDOM_STATE
        ),
        "Random Forest Regression": RandomForestRegressor(
            n_estimators=250,
            max_depth=10,
            min_samples_leaf=4,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    rows = []
    artifacts: Dict[str, dict] = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        rows.append(
            {
                "Model": name,
                "R²": r2_score(y_test, pred),
                "MAE": mean_absolute_error(y_test, pred),
                "RMSE": rmse,
            }
        )
        artifacts[name] = {
            "model": model,
            "X_test": X_test,
            "y_test": y_test,
            "pred": pred,
            "features": list(feature_columns),
        }

    comparison = pd.DataFrame(rows).sort_values("R²", ascending=False)
    return comparison, artifacts


def calculate_vif(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    numeric = df[columns].dropna().astype(float)
    if numeric.empty:
        return pd.DataFrame()
    scaled = StandardScaler().fit_transform(numeric)
    vif_rows = []
    for index, col in enumerate(columns):
        try:
            vif = variance_inflation_factor(scaled, index)
        except Exception:
            vif = np.nan
        vif_rows.append({"Feature": col, "VIF": vif})
    return pd.DataFrame(vif_rows).sort_values("VIF", ascending=False)


def build_transaction_matrix(series: pd.Series) -> pd.DataFrame:
    transactions = series.fillna("").astype(str).apply(
        lambda value: [item.strip() for item in value.replace(";", "|").split("|") if item.strip()]
    )
    unique_items = sorted({item for basket in transactions for item in basket})
    if not unique_items:
        return pd.DataFrame()
    return pd.DataFrame(
        [{item: item in basket for item in unique_items} for basket in transactions],
        dtype=bool,
    )


# =============================================================================
# 3. DATA SOURCE AND SIDEBAR
# =============================================================================
st.sidebar.markdown("## 🛒 CartIQ Engine")
st.sidebar.caption("GCC E-commerce Decision Intelligence")
st.sidebar.divider()

uploaded_file = st.sidebar.file_uploader("Upload a replacement CSV", type=["csv"])
raw_df, source_label = load_source_data(uploaded_file)
cleaned_df, cleaning_log = clean_pipeline(raw_df)

st.sidebar.success(source_label)
st.sidebar.caption(f"{len(cleaned_df):,} cleaned sessions · {cleaned_df.shape[1]} fields")

page = st.sidebar.radio(
    "Navigate workspace",
    [
        "0. Executive Overview",
        "1. Data Cleaning Pipeline",
        "2. Descriptive Checkout Insights",
        "3. Diagnostic Funnel Friction",
        "4. Classification: Cart Conversion",
        "5. Regression: Cart Value Optimization",
        "6. Classification: Abandonment Recovery",
        "7. Clustering: Buyer Archetypes",
        "8. Association Rules: Cart Groupings",
        "9. Executive Strategy Playbook",
    ],
)

st.sidebar.divider()
st.sidebar.download_button(
    "Download cleaned CSV",
    data=cleaned_df.to_csv(index=False).encode("utf-8"),
    file_name="CartIQ_cleaned_export.csv",
    mime="text/csv",
    width="stretch",
)

st.markdown(
    """
    <div class="hero">
        <h1>CartIQ E-commerce Analytics Portal</h1>
        <p>From funnel visibility to predictive conversion, recovery targeting, buyer archetypes, and cross-sell intelligence.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# PAGE 0: EXECUTIVE OVERVIEW
# =============================================================================
if page == "0. Executive Overview":
    require_columns(
        cleaned_df,
        ["checkout_status", "total_cart_value_usd", "historical_clv", "items_in_cart"],
        "Executive overview",
    )

    sessions = len(cleaned_df)
    purchases = int(cleaned_df["checkout_status"].sum())
    conversion_rate = cleaned_df["checkout_status"].mean()
    purchased_value = cleaned_df.loc[cleaned_df["checkout_status"] == 1, "total_cart_value_usd"]
    revenue = purchased_value.sum()
    avg_order_value = purchased_value.mean() if len(purchased_value) else 0

    cols = st.columns(5)
    cols[0].metric("Sessions", f"{sessions:,}")
    cols[1].metric("Purchases", f"{purchases:,}")
    cols[2].metric("Conversion rate", f"{conversion_rate:.1%}")
    cols[3].metric("Converted revenue", f"${revenue:,.0f}")
    cols[4].metric("Average order value", f"${avg_order_value:,.0f}")

    left, right = st.columns([1.05, 1])
    with left:
        funnel_columns = [
            ("Product viewed", "product_viewed"),
            ("Cart added", "cart_added"),
            ("Checkout started", "checkout_started"),
            ("Shipping entered", "shipping_details_entered"),
            ("Payment completed", "payment_stage_completed"),
            ("Purchased", "checkout_status"),
        ]
        available_funnel = [(label, col) for label, col in funnel_columns if col in cleaned_df.columns]
        funnel_values = [int(cleaned_df[col].fillna(0).sum()) for _, col in available_funnel]
        fig_funnel = go.Figure(
            go.Funnel(
                y=[label for label, _ in available_funnel],
                x=funnel_values,
                textinfo="value+percent initial+percent previous",
            )
        )
        fig_funnel.update_layout(title="Customer Journey Funnel", margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig_funnel, width="stretch")

    with right:
        require_columns(cleaned_df, ["user_region"], "Regional conversion chart")
        region_summary = (
            cleaned_df.groupby("user_region", dropna=False)
            .agg(
                Sessions=("checkout_status", "size"),
                Conversion_Rate=("checkout_status", "mean"),
                Converted_Revenue=(
                    "total_cart_value_usd",
                    lambda values: values[cleaned_df.loc[values.index, "checkout_status"] == 1].sum(),
                ),
            )
            .reset_index()
        )
        fig_region = px.bar(
            region_summary,
            x="user_region",
            y="Conversion_Rate",
            text=region_summary["Conversion_Rate"].map(lambda x: f"{x:.1%}"),
            hover_data={"Sessions": True, "Converted_Revenue": ":,.0f", "Conversion_Rate": ":.1%"},
            title="Conversion by Region",
            labels={"user_region": "Region", "Conversion_Rate": "Conversion rate"},
        )
        fig_region.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_region, width="stretch")

    st.subheader("Where management attention should go")
    best_region = region_summary.loc[region_summary["Conversion_Rate"].idxmax()]
    worst_region = region_summary.loc[region_summary["Conversion_Rate"].idxmin()]
    abandoned_count = sessions - purchases
    recovery_rate = (
        cleaned_df.loc[cleaned_df["checkout_status"] == 0, "recovery_email_conversion"].mean()
        if "recovery_email_conversion" in cleaned_df.columns
        else np.nan
    )

    a, b, c = st.columns(3)
    with a:
        st.markdown(
            f"<div class='insight-card'><b>Regional gap</b><br>{best_region['user_region']} leads at "
            f"{best_region['Conversion_Rate']:.1%}, while {worst_region['user_region']} records "
            f"{worst_region['Conversion_Rate']:.1%}. Investigate payment mix and traffic quality before reallocating spend.</div>",
            unsafe_allow_html=True,
        )
    with b:
        st.markdown(
            f"<div class='insight-card'><b>Abandonment pool</b><br>{abandoned_count:,} sessions did not convert. "
            "This is the addressable population for recovery prioritisation rather than blanket discounting.</div>",
            unsafe_allow_html=True,
        )
    with c:
        recovery_text = f"{recovery_rate:.1%}" if pd.notna(recovery_rate) else "Unavailable"
        st.markdown(
            f"<div class='insight-card'><b>Recovery benchmark</b><br>Observed recovery conversion is {recovery_text}. "
            "Use the recovery model to target customers by probability and expected value.</div>",
            unsafe_allow_html=True,
        )


# =============================================================================
# PAGE 1: DATA CLEANING PIPELINE
# =============================================================================
elif page == "1. Data Cleaning Pipeline":
    st.header("⚙️ Automated Cleaning & Engineering Pipeline")
    st.caption("An auditable view of casing fixes, whitespace cleanup, duplicate removal, null handling, and engineered fields.")

    raw_missing = int(raw_df.isna().sum().sum())
    clean_missing = int(cleaned_df.isna().sum().sum())
    cols = st.columns(4)
    cols[0].metric("Raw rows", f"{len(raw_df):,}")
    cols[1].metric("Cleaned rows", f"{len(cleaned_df):,}")
    cols[2].metric("Raw missing cells", f"{raw_missing:,}")
    cols[3].metric("Remaining missing cells", f"{clean_missing:,}")

    st.subheader("Cleaning log")
    st.dataframe(cleaning_log, width="stretch", hide_index=True)

    left, right = st.columns(2)
    with left:
        if "device_category" in raw_df.columns and "device_category" in cleaned_df.columns:
            st.subheader("Device standardisation")
            before_device = raw_df["device_category"].astype(str).value_counts().rename_axis("Raw value").reset_index(name="Rows")
            after_device = cleaned_df["device_category"].value_counts().rename_axis("Clean value").reset_index(name="Rows")
            st.write("**Before**")
            st.dataframe(before_device, width="stretch", hide_index=True)
            st.write("**After**")
            st.dataframe(after_device, width="stretch", hide_index=True)

    with right:
        if "user_region" in raw_df.columns and "user_region" in cleaned_df.columns:
            st.subheader("Region standardisation")
            raw_regions = sorted(raw_df["user_region"].dropna().astype(str).unique().tolist())
            clean_regions = sorted(cleaned_df["user_region"].dropna().astype(str).unique().tolist())
            max_len = max(len(raw_regions), len(clean_regions))
            region_table = pd.DataFrame(
                {
                    "Raw region values": raw_regions + [None] * (max_len - len(raw_regions)),
                    "Cleaned region values": clean_regions + [None] * (max_len - len(clean_regions)),
                }
            )
            st.dataframe(region_table, width="stretch", hide_index=True)

    structural_cols = [c for c in ["days_to_delivery", "delivery_satisfaction_rating"] if c in raw_df.columns]
    if structural_cols and "checkout_status" in raw_df.columns:
        st.subheader("Structural missingness")
        structural = (
            raw_df.groupby("checkout_status")[structural_cols]
            .agg(lambda s: int(s.isna().sum()))
            .rename(index={0: "Abandoned", 1: "Purchased"})
        )
        st.warning(
            "Delivery fields are downstream outcomes. Their nulls for abandoned sessions are meaningful and should not be imputed or used as pre-purchase predictors."
        )
        st.dataframe(structural, width="stretch")

    with st.expander("Preview cleaned dataset"):
        st.dataframe(cleaned_df.head(100), width="stretch", hide_index=True)


# =============================================================================
# PAGE 2: DESCRIPTIVE CHECKOUT INSIGHTS
# =============================================================================
elif page == "2. Descriptive Checkout Insights":
    require_columns(cleaned_df, REQUIRED_BASE_COLUMNS, "Descriptive analysis")
    st.header("📊 Descriptive Checkout Insights")

    st.sidebar.subheader("Page filters")
    region_options = sorted(cleaned_df["user_region"].dropna().unique().tolist())
    tier_options = sorted(cleaned_df["user_tier"].dropna().unique().tolist())
    region_filter = st.sidebar.multiselect("Region", region_options, default=region_options)
    tier_filter = st.sidebar.multiselect("User tier", tier_options, default=tier_options)

    sub_df = cleaned_df[
        cleaned_df["user_region"].isin(region_filter) & cleaned_df["user_tier"].isin(tier_filter)
    ].copy()
    if sub_df.empty:
        st.warning("The selected filters return no sessions.")
        st.stop()

    conversion = sub_df["checkout_status"].mean()
    aov = sub_df.loc[sub_df["checkout_status"] == 1, "total_cart_value_usd"].mean()
    abandonment = 1 - conversion
    cols = st.columns(4)
    cols[0].metric("Filtered sessions", f"{len(sub_df):,}")
    cols[1].metric("Conversion", f"{conversion:.1%}")
    cols[2].metric("Abandonment", f"{abandonment:.1%}")
    cols[3].metric("Converted AOV", f"${aov:,.0f}" if pd.notna(aov) else "$0")

    st.info(
        "Target-leakage guardrail: downstream fields such as shipping completion, delivery time, and delivery rating are excluded from conversion models."
    )

    c1, c2 = st.columns([1.05, 1])
    with c1:
        ct_var = st.selectbox(
            "Compare conversion against",
            ["traffic_source", "device_category", "payment_method_selected", "primary_product_category", "user_region"],
        )
        ct_data = (
            sub_df.groupby(ct_var, dropna=False)["checkout_status"]
            .agg(Sessions="size", Conversion_Rate="mean")
            .reset_index()
            .sort_values("Conversion_Rate", ascending=False)
        )
        fig_ct = px.bar(
            ct_data,
            x=ct_var,
            y="Conversion_Rate",
            text=ct_data["Conversion_Rate"].map(lambda x: f"{x:.1%}"),
            hover_data={"Sessions": True, "Conversion_Rate": ":.1%"},
            labels={"Conversion_Rate": "Conversion rate"},
            title=f"Conversion by {ct_var.replace('_', ' ').title()}",
        )
        fig_ct.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_ct, width="stretch")

    with c2:
        fig_value = px.scatter(
            sub_df,
            x="session_duration_secs",
            y="total_cart_value_usd",
            color=sub_df["checkout_status"].map({0: "Abandoned", 1: "Purchased"}),
            size="items_in_cart",
            hover_data=["user_region", "traffic_source", "payment_method_selected"],
            opacity=0.58,
            title="Session engagement vs. cart value",
            labels={"color": "Outcome", "session_duration_secs": "Session duration (sec)", "total_cart_value_usd": "Cart value (USD)"},
        )
        st.plotly_chart(fig_value, width="stretch")

    st.subheader("Numeric correlation matrix")
    numeric_cols = [
        "past_purchases_count",
        "items_in_cart",
        "average_item_price",
        "session_duration_secs",
        "historical_clv",
        "pages_viewed",
        "items_added_count",
        "total_cart_value_usd",
        "checkout_status",
    ]
    numeric_cols = [c for c in numeric_cols if c in sub_df.columns]
    corr = sub_df[numeric_cols].corr(numeric_only=True)
    fig_hm = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Pearson correlation matrix",
    )
    st.plotly_chart(fig_hm, width="stretch")


# =============================================================================
# PAGE 3: DIAGNOSTIC FUNNEL FRICTION
# =============================================================================
elif page == "3. Diagnostic Funnel Friction":
    require_columns(
        cleaned_df,
        ["user_region", "payment_method_selected", "device_category", "traffic_source", "checkout_status"],
        "Funnel diagnostic",
    )
    st.header("🔍 Diagnostic Funnel Friction")
    st.caption("Find underperforming combinations instead of relying on a single-variable average.")

    friction = (
        cleaned_df.groupby(["user_region", "payment_method_selected"], dropna=False)
        .agg(Sessions=("checkout_status", "size"), Conversion_Rate=("checkout_status", "mean"))
        .reset_index()
    )
    overall_rate = cleaned_df["checkout_status"].mean()
    friction["Gap_vs_Overall"] = friction["Conversion_Rate"] - overall_rate

    fig_friction = px.bar(
        friction,
        x="user_region",
        y="Conversion_Rate",
        color="payment_method_selected",
        barmode="group",
        hover_data={"Sessions": True, "Gap_vs_Overall": ":.1%", "Conversion_Rate": ":.1%"},
        title="Regional conversion by payment method",
        labels={"user_region": "Region", "Conversion_Rate": "Conversion rate", "payment_method_selected": "Payment method"},
    )
    fig_friction.add_hline(y=overall_rate, line_dash="dash", annotation_text=f"Overall {overall_rate:.1%}")
    fig_friction.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_friction, width="stretch")

    left, right = st.columns(2)
    with left:
        device_traffic = (
            cleaned_df.groupby(["device_category", "traffic_source"])
            .agg(Sessions=("checkout_status", "size"), Conversion_Rate=("checkout_status", "mean"))
            .reset_index()
        )
        pivot = device_traffic.pivot(index="device_category", columns="traffic_source", values="Conversion_Rate")
        fig_dt = px.imshow(
            pivot,
            text_auto=".1%",
            aspect="auto",
            color_continuous_scale="Blues",
            title="Device × traffic conversion heatmap",
        )
        st.plotly_chart(fig_dt, width="stretch")

    with right:
        min_sessions = st.slider("Minimum sessions per diagnostic cell", 10, 150, 40, 10)
        low_cells = friction[friction["Sessions"] >= min_sessions].nsmallest(8, "Conversion_Rate").copy()
        low_cells["Conversion rate"] = low_cells["Conversion_Rate"].map(lambda x: f"{x:.1%}")
        low_cells["Gap vs overall"] = low_cells["Gap_vs_Overall"].map(lambda x: f"{x:+.1%}")
        st.subheader("Lowest-performing payment-region cells")
        st.dataframe(
            low_cells[["user_region", "payment_method_selected", "Sessions", "Conversion rate", "Gap vs overall"]],
            width="stretch",
            hide_index=True,
        )

    weakest = friction[friction["Sessions"] >= min_sessions].nsmallest(1, "Conversion_Rate")
    if not weakest.empty:
        row = weakest.iloc[0]
        st.markdown(
            f"<div class='insight-card'><b>Priority diagnostic:</b> {row['payment_method_selected']} in "
            f"{row['user_region']} converts at {row['Conversion_Rate']:.1%}, "
            f"{row['Gap_vs_Overall']:+.1%} versus the overall rate. Validate sample quality, payment UX, fees, and checkout errors before treating this as causal.</div>",
            unsafe_allow_html=True,
        )


# =============================================================================
# PAGE 4: CLASSIFICATION — CART CONVERSION
# =============================================================================
elif page == "4. Classification: Cart Conversion":
    feature_cols = (
        "session_duration_secs",
        "items_in_cart",
        "average_item_price",
        "past_purchases_count",
        "historical_clv",
        "pages_viewed",
        "items_added_count",
    )
    require_columns(cleaned_df, list(feature_cols) + ["checkout_status"], "Cart conversion model")
    st.header("🤖 Classification: Cart Conversion")
    st.caption("Pre-purchase behavioural predictors only; downstream leakage variables are deliberately excluded.")

    try:
        comparison, artifacts = train_classification_suite(cleaned_df, feature_cols, "checkout_status")
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    display_metrics = comparison.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]:
        display_metrics[col] = display_metrics[col].map(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
    st.subheader("Model comparison")
    st.dataframe(display_metrics, width="stretch", hide_index=True)

    selected_model = st.selectbox("Inspect model", comparison["Model"].tolist())
    artifact = artifacts[selected_model]
    selected_row = comparison.loc[comparison["Model"] == selected_model].iloc[0]

    metrics_cols = st.columns(5)
    for box, metric in zip(metrics_cols, ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]):
        box.metric(metric, f"{selected_row[metric]:.1%}" if pd.notna(selected_row[metric]) else "N/A")

    left, right = st.columns(2)
    with left:
        cm = confusion_matrix(artifact["y_test"], artifact["pred"], labels=[0, 1])
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            x=["Abandoned", "Purchased"],
            y=["Abandoned", "Purchased"],
            labels={"x": "Predicted", "y": "Actual", "color": "Sessions"},
            title=f"Confusion matrix — {selected_model}",
            color_continuous_scale="Blues",
        )
        st.plotly_chart(fig_cm, width="stretch")

    with right:
        try:
            fpr, tpr, _ = roc_curve(artifact["y_test"], artifact["prob"])
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=selected_model))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random baseline", line=dict(dash="dash")))
            fig_roc.update_layout(
                title="ROC curve",
                xaxis_title="False-positive rate",
                yaxis_title="True-positive rate",
                xaxis=dict(range=[0, 1]),
                yaxis=dict(range=[0, 1]),
            )
            st.plotly_chart(fig_roc, width="stretch")
        except ValueError:
            st.info("ROC curve requires both target classes in the test sample.")

    importance = extract_feature_importance(artifact["model"], artifact["features"])
    if not importance.empty:
        fig_imp = px.bar(
            importance,
            x="Importance",
            y="Feature",
            orientation="h",
            title=f"Feature influence — {selected_model}",
        )
        st.plotly_chart(fig_imp, width="stretch")

    st.warning(
        "Model scores show predictive discrimination, not causality. Validate on a later time period before using the scores for live interventions."
    )


# =============================================================================
# PAGE 5: REGRESSION — CART VALUE
# =============================================================================
elif page == "5. Regression: Cart Value Optimization":
    reg_features = ("items_in_cart", "average_item_price", "pages_viewed", "items_added_count", "session_duration_secs")
    require_columns(cleaned_df, list(reg_features) + ["total_cart_value_usd"], "Cart-value regression")
    st.header("📈 Regression: Cart Value Optimization")

    alpha = st.sidebar.slider("Regularisation alpha", 0.01, 25.0, 1.0, 0.05)
    comparison, artifacts = train_regression_suite(cleaned_df, reg_features, "total_cart_value_usd", alpha)

    st.subheader("Model comparison")
    display_reg = comparison.copy()
    display_reg["R²"] = display_reg["R²"].map(lambda x: f"{x:.3f}")
    display_reg["MAE"] = display_reg["MAE"].map(lambda x: f"${x:,.2f}")
    display_reg["RMSE"] = display_reg["RMSE"].map(lambda x: f"${x:,.2f}")
    st.dataframe(display_reg, width="stretch", hide_index=True)

    selected_model = st.selectbox("Inspect regression model", comparison["Model"].tolist())
    artifact = artifacts[selected_model]
    selected_row = comparison.loc[comparison["Model"] == selected_model].iloc[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("R²", f"{selected_row['R²']:.3f}")
    c2.metric("MAE", f"${selected_row['MAE']:,.2f}")
    c3.metric("RMSE", f"${selected_row['RMSE']:,.2f}")

    left, right = st.columns(2)
    with left:
        actual_pred = pd.DataFrame({"Actual": artifact["y_test"], "Predicted": artifact["pred"]})
        fig_ap = px.scatter(actual_pred, x="Actual", y="Predicted", opacity=0.55, title="Actual vs. predicted cart value")
        min_val = float(min(actual_pred.min()))
        max_val = float(max(actual_pred.max()))
        fig_ap.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode="lines", name="Perfect prediction", line=dict(dash="dash")))
        st.plotly_chart(fig_ap, width="stretch")

    with right:
        vif_df = calculate_vif(cleaned_df, list(reg_features))
        st.subheader("Variance Inflation Factor")
        st.dataframe(vif_df.round(2), width="stretch", hide_index=True)
        if not vif_df.empty and (vif_df["VIF"] > 5).any():
            st.warning("At least one feature has VIF above 5, indicating meaningful multicollinearity. Ridge or Lasso is preferable for coefficient stability.")
        else:
            st.success("No feature exceeds the common VIF > 5 warning threshold.")

    st.subheader("Regularisation coefficient path")
    alphas = np.logspace(-2, 2, 50)
    X_path = cleaned_df[list(reg_features)].dropna()
    y_path = cleaned_df.loc[X_path.index, "total_cart_value_usd"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_path)
    ridge_coefs = [Ridge(alpha=a).fit(X_scaled, y_path).coef_ for a in alphas]
    fig_path = go.Figure()
    for i, feature in enumerate(reg_features):
        fig_path.add_trace(go.Scatter(x=alphas, y=[coef[i] for coef in ridge_coefs], mode="lines", name=feature))
    fig_path.add_vline(x=alpha, line_dash="dash", annotation_text="Selected alpha")
    fig_path.update_layout(xaxis_type="log", xaxis_title="Alpha", yaxis_title="Standardised coefficient", title="Ridge coefficient shrinkage")
    st.plotly_chart(fig_path, width="stretch")


# =============================================================================
# PAGE 6: CLASSIFICATION — ABANDONMENT RECOVERY
# =============================================================================
elif page == "6. Classification: Abandonment Recovery":
    recovery_features = (
        "historical_clv",
        "coupon_discount_pct_offered",
        "recovery_email_delay_hours",
        "recovery_email_opened",
        "items_in_cart",
        "session_duration_secs",
        "past_purchases_count",
    )
    require_columns(
        cleaned_df,
        list(recovery_features) + ["checkout_status", "recovery_email_conversion"],
        "Recovery model",
    )
    st.header("✉️ Classification: Abandonment Recovery")
    st.caption("Model only the sessions that abandoned the original checkout.")

    recovery_data = cleaned_df[cleaned_df["checkout_status"] == 0].copy()
    if len(recovery_data) < 50:
        st.error("At least 50 abandoned sessions are required for the recovery model.")
        st.stop()

    comparison, artifacts = train_classification_suite(
        recovery_data,
        recovery_features,
        "recovery_email_conversion",
    )
    display_metrics = comparison.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]:
        display_metrics[col] = display_metrics[col].map(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
    st.dataframe(display_metrics, width="stretch", hide_index=True)

    selected_model = st.selectbox("Inspect recovery model", comparison["Model"].tolist())
    artifact = artifacts[selected_model]

    left, right = st.columns(2)
    with left:
        coupon_summary = (
            recovery_data.groupby("coupon_discount_pct_offered")
            .agg(
                Abandoned_Sessions=("recovery_email_conversion", "size"),
                Recovery_Rate=("recovery_email_conversion", "mean"),
                Average_CLV=("historical_clv", "mean"),
            )
            .reset_index()
        )
        fig_coupon = px.bar(
            coupon_summary,
            x="coupon_discount_pct_offered",
            y="Recovery_Rate",
            text=coupon_summary["Recovery_Rate"].map(lambda x: f"{x:.1%}"),
            hover_data={"Abandoned_Sessions": True, "Average_CLV": ":,.0f", "Recovery_Rate": ":.1%"},
            title="Observed recovery rate by discount",
            labels={"coupon_discount_pct_offered": "Coupon discount (%)", "Recovery_Rate": "Recovery rate"},
        )
        fig_coupon.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_coupon, width="stretch")

    with right:
        scored = artifact["X_test"].copy()
        scored["Actual"] = artifact["y_test"].values
        scored["Recovery probability"] = artifact["prob"]
        scored["Expected recovered value"] = scored["Recovery probability"] * scored["historical_clv"]
        top_targets = scored.nlargest(15, "Expected recovered value")
        st.subheader("Highest-value recovery targets in test sample")
        st.dataframe(
            top_targets[["Recovery probability", "historical_clv", "coupon_discount_pct_offered", "recovery_email_delay_hours", "Expected recovered value"]]
            .round(2),
            width="stretch",
            hide_index=True,
        )

    importance = extract_feature_importance(artifact["model"], artifact["features"])
    if not importance.empty:
        fig_imp = px.bar(importance, x="Importance", y="Feature", orientation="h", title=f"Recovery feature influence — {selected_model}")
        st.plotly_chart(fig_imp, width="stretch")

    st.info("Operational use: rank abandoned carts by probability × expected value, then apply the smallest viable incentive rather than sending the same coupon to everyone.")


# =============================================================================
# PAGE 7: CLUSTERING
# =============================================================================
elif page == "7. Clustering: Buyer Archetypes":
    cluster_cols = ["total_cart_value_usd", "session_duration_secs", "past_purchases_count", "historical_clv"]
    require_columns(cleaned_df, cluster_cols, "Buyer clustering")
    st.header("🧬 Clustering: Buyer Archetypes")

    k = st.sidebar.slider("Number of clusters", 2, 7, 4)
    cluster_source = cleaned_df[cluster_cols].dropna().copy()
    scaled = StandardScaler().fit_transform(cluster_source)
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
    labels = kmeans.fit_predict(scaled)
    sil = silhouette_score(scaled, labels) if 1 < k < len(cluster_source) else np.nan

    clustered = cleaned_df.loc[cluster_source.index].copy()
    clustered["Cluster"] = labels.astype(str)
    pca = PCA(n_components=3, random_state=RANDOM_STATE)
    projection = pca.fit_transform(scaled)
    pca_df = pd.DataFrame(projection, columns=["PC1", "PC2", "PC3"])
    pca_df["Cluster"] = labels.astype(str)

    c1, c2, c3 = st.columns(3)
    c1.metric("Clusters", k)
    c2.metric("Silhouette score", f"{sil:.3f}")
    c3.metric("PCA variance captured", f"{pca.explained_variance_ratio_.sum():.1%}")

    fig_3d = px.scatter_3d(
        pca_df,
        x="PC1",
        y="PC2",
        z="PC3",
        color="Cluster",
        opacity=0.65,
        title="3D PCA projection of buyer clusters",
    )
    st.plotly_chart(fig_3d, width="stretch")

    profile = clustered.groupby("Cluster")[cluster_cols].mean().round(1)
    profile["Sessions"] = clustered.groupby("Cluster").size()
    if "checkout_status" in clustered.columns:
        profile["Conversion Rate"] = clustered.groupby("Cluster")["checkout_status"].mean()
    st.subheader("Cluster profile")
    st.dataframe(profile, width="stretch")

    # Human-readable archetype suggestion based on relative profile ranks.
    ranks = profile[cluster_cols].rank(pct=True)
    descriptions = []
    for cluster_id in profile.index:
        row = ranks.loc[cluster_id]
        if row["historical_clv"] >= 0.75 and row["total_cart_value_usd"] >= 0.75:
            label = "High-value loyalists"
        elif row["session_duration_secs"] >= 0.75 and row["total_cart_value_usd"] < 0.50:
            label = "High-engagement browsers"
        elif row["past_purchases_count"] < 0.40 and row["total_cart_value_usd"] >= 0.60:
            label = "New premium shoppers"
        else:
            label = "Value-conscious regulars"
        descriptions.append({"Cluster": cluster_id, "Suggested archetype": label})
    st.dataframe(pd.DataFrame(descriptions), width="stretch", hide_index=True)


# =============================================================================
# PAGE 8: ASSOCIATION RULES
# =============================================================================
elif page == "8. Association Rules: Cart Groupings":
    require_columns(cleaned_df, ["basket_items"], "Association-rule mining")
    st.header("⛓️ Association Rules: Cart Groupings")
    st.caption("Discover products that occur together frequently enough to support bundles and checkout recommendations.")

    min_support = st.sidebar.slider("Minimum support", 0.01, 0.20, 0.04, 0.01)
    min_lift = st.sidebar.slider("Minimum lift", 1.0, 4.0, 1.05, 0.05)

    basket = build_transaction_matrix(cleaned_df["basket_items"])
    if basket.empty or basket.shape[1] < 2:
        st.error("No usable basket-item transactions were found.")
        st.stop()

    frequent_itemsets = apriori(basket, min_support=min_support, use_colnames=True)
    if frequent_itemsets.empty:
        st.warning("No frequent itemsets meet the selected support threshold. Lower minimum support.")
        st.stop()

    try:
        rules = association_rules(
            frequent_itemsets,
            num_itemsets=len(frequent_itemsets),
            metric="lift",
            min_threshold=min_lift,
        )
    except TypeError:
        # Compatibility with mlxtend versions before num_itemsets was introduced.
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_lift)

    if rules.empty:
        st.warning("No association rules meet the selected lift threshold. Lower minimum lift or support.")
        st.stop()

    rules = rules.copy()
    rules["Antecedent"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(x)))
    rules["Consequent"] = rules["consequents"].apply(lambda x: ", ".join(sorted(x)))
    rules = rules.sort_values(["lift", "confidence", "support"], ascending=False)

    c1, c2, c3 = st.columns(3)
    c1.metric("Frequent itemsets", f"{len(frequent_itemsets):,}")
    c2.metric("Rules discovered", f"{len(rules):,}")
    c3.metric("Highest lift", f"{rules['lift'].max():.2f}")

    top_rules = rules[["Antecedent", "Consequent", "support", "confidence", "lift"]].head(25)
    st.dataframe(top_rules.round(3), width="stretch", hide_index=True)

    fig_rules = px.scatter(
        rules.head(150),
        x="support",
        y="confidence",
        size="lift",
        color="lift",
        hover_data=["Antecedent", "Consequent"],
        title="Association-rule quality map",
    )
    st.plotly_chart(fig_rules, width="stretch")

    best_rule = rules.iloc[0]
    st.markdown(
        f"<div class='insight-card'><b>Best bundle candidate:</b> When shoppers buy <b>{best_rule['Antecedent']}</b>, "
        f"they are associated with <b>{best_rule['Consequent']}</b>. Confidence is {best_rule['confidence']:.1%} "
        f"and lift is {best_rule['lift']:.2f}. Test this as a recommendation rather than assuming the rule is causal.</div>",
        unsafe_allow_html=True,
    )


# =============================================================================
# PAGE 9: EXECUTIVE PLAYBOOK
# =============================================================================
elif page == "9. Executive Strategy Playbook":
    require_columns(
        cleaned_df,
        ["checkout_status", "payment_method_selected", "user_region", "historical_clv", "total_cart_value_usd"],
        "Executive playbook",
    )
    st.header("🎯 Executive Strategy Playbook")
    st.caption("Actions are generated from the currently loaded dataset, not from fixed narrative claims.")

    overall_rate = cleaned_df["checkout_status"].mean()
    payment_perf = (
        cleaned_df.groupby("payment_method_selected")
        .agg(Sessions=("checkout_status", "size"), Conversion_Rate=("checkout_status", "mean"))
        .reset_index()
        .sort_values("Conversion_Rate", ascending=False)
    )
    region_perf = (
        cleaned_df.groupby("user_region")
        .agg(Sessions=("checkout_status", "size"), Conversion_Rate=("checkout_status", "mean"))
        .reset_index()
        .sort_values("Conversion_Rate", ascending=False)
    )
    best_payment = payment_perf.iloc[0]
    weakest_region = region_perf.iloc[-1]

    st.subheader("1. Reduce checkout friction")
    st.markdown(
        f"<div class='insight-card'>The strongest observed payment method is <b>{best_payment['payment_method_selected']}</b> "
        f"at {best_payment['Conversion_Rate']:.1%}, compared with the overall rate of {overall_rate:.1%}. "
        "Preserve payment choice and investigate checkout failures by region, device, and traffic source before changing the default option.</div>",
        unsafe_allow_html=True,
    )

    st.subheader("2. Prioritise the weakest geography with cross-feature diagnosis")
    st.markdown(
        f"<div class='insight-card'><b>{weakest_region['user_region']}</b> has the lowest regional conversion at "
        f"{weakest_region['Conversion_Rate']:.1%}. Do not treat geography as the cause; break the gap down by payment method, "
        "device, traffic source, cart value, and membership status.</div>",
        unsafe_allow_html=True,
    )

    st.subheader("3. Protect margin in recovery campaigns")
    if {"checkout_status", "recovery_email_conversion", "coupon_discount_pct_offered"}.issubset(cleaned_df.columns):
        recovered = cleaned_df[cleaned_df["checkout_status"] == 0]
        discount_perf = (
            recovered.groupby("coupon_discount_pct_offered")
            .agg(Sessions=("recovery_email_conversion", "size"), Recovery_Rate=("recovery_email_conversion", "mean"))
            .reset_index()
        )
        best_discount = discount_perf.loc[discount_perf["Recovery_Rate"].idxmax()]
        st.markdown(
            f"<div class='insight-card'>The highest observed recovery rate occurs at a {best_discount['coupon_discount_pct_offered']:.0f}% "
            f"discount ({best_discount['Recovery_Rate']:.1%}). This does not prove the discount caused recovery. Run a randomised test and optimise expected margin, not conversion alone.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.info("Recovery fields are unavailable in the uploaded dataset.")

    st.subheader("4. Build an experimentation roadmap")
    roadmap = pd.DataFrame(
        [
            ["Payment UX", "Payment-method order or copy", "Checkout conversion", "Guardrail: payment errors and margin"],
            ["Recovery timing", "Email after 1h vs 4h vs 12h", "Incremental recovered revenue", "Guardrail: unsubscribe rate"],
            ["Incentive", "No coupon vs minimum viable coupon", "Incremental contribution margin", "Guardrail: discount cost"],
            ["Cross-sell", "Top association-rule bundle vs generic recommendation", "Attach rate and AOV", "Guardrail: checkout completion"],
        ],
        columns=["Experiment", "Treatment", "Primary metric", "Guardrail"],
    )
    st.dataframe(roadmap, width="stretch", hide_index=True)

    st.warning("All dashboard findings are observational unless validated through controlled experiments or robust causal designs.")
