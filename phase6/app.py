import io
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import joblib

import app_common as common

# Pipeline functions need to be importable here. Easiest path: copy the same
# adapter + run_phase4_pipeline cells from the notebook into a module called
# pipeline.py next to this file, then import it.
from pipeline import adapt_sparkov_to_ieee_schema, run_phase4_pipeline

ARTIFACT_DIR = Path(os.environ.get(
    "PHASE5_ARTIFACTS",
    str(Path(__file__).parent / "app_artifacts")
))

st.set_page_config(
    page_title="Fraud Detection — Phase 5",
    page_icon="🛡️",
    layout="wide",
)


# =============================================================================
# Visualization helpers
# =============================================================================
def render_probability_gauge(proba: float, threshold: float):
    """Horizontal gauge showing fraud probability vs threshold."""
    fig, ax = plt.subplots(figsize=(9, 1.1))

    # Background bar (full 0-1 range)
    ax.barh([0], [1.0], color="#2a2f3a", height=0.5, zorder=1)

    # Filled portion (the actual probability), color reflects verdict
    is_fraud = proba >= threshold
    fill_color = "#e74c3c" if is_fraud else "#2ecc71"
    ax.barh([0], [proba], color=fill_color, height=0.5, zorder=2)

    # Threshold marker
    ax.axvline(threshold, color="#f1c40f", linestyle="--",
               linewidth=2, zorder=3, label=f"Threshold {threshold:.2f}")

    # Probability marker (small triangle on top)
    ax.scatter([proba], [0.45], marker="v", s=120, color="white",
               edgecolors="black", linewidths=1.2, zorder=4)
    ax.annotate(f"{proba*100:.1f}%",
                xy=(proba, 0.45), xytext=(0, 14), textcoords="offset points",
                ha="center", fontsize=11, fontweight="bold", color="white")

    # Cosmetics
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 0.9)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], color="#aaa", fontsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="x", colors="#aaa", length=0)
    ax.legend(loc="lower right", frameon=False, fontsize=9,
              labelcolor="#aaa", bbox_to_anchor=(1.0, -0.3))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    fig.tight_layout()
    return fig


# =============================================================================
# Load artifacts (cached so we only do it once per session)
# =============================================================================
@st.cache_resource
def load_model_and_artifacts():
    model = joblib.load(str(ARTIFACT_DIR / "lgbm_sparkov.joblib"))

    with open(ARTIFACT_DIR / "sparkov_artifacts.pkl", "rb") as f:
        artifacts = pickle.load(f)

    with open(ARTIFACT_DIR / "feature_names_sparkov.pkl", "rb") as f:
        feature_names = pickle.load(f)

    with open(ARTIFACT_DIR / "app_meta.json") as f:
        meta = json.load(f)

    return model, artifacts, feature_names, meta


@st.cache_data
def load_sample_csv(filename: str = "sample_transactions.csv"):
    p = ARTIFACT_DIR / filename
    return pd.read_csv(p) if p.exists() else None


# =============================================================================
# Real predictor: schema-map → pipeline → model
# =============================================================================
def predict_one(transaction: dict, model, artifacts, feature_names) -> float:
    """Run a single Sparkov-shaped transaction through the pipeline and model."""
    df = pd.DataFrame([transaction])
    mapped = adapt_sparkov_to_ieee_schema(df)
    processed, _ = run_phase4_pipeline(mapped, fit_artifacts=artifacts)

    X = processed.drop(columns=["isFraud"], errors="ignore")

    # Align to training feature vector
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_names]

    return float(model.predict_proba(X)[:, 1][0])


def predict_batch(df: pd.DataFrame, model, artifacts, feature_names) -> np.ndarray:
    """Score a whole DataFrame of Sparkov-shaped transactions."""
    mapped = adapt_sparkov_to_ieee_schema(df)
    processed, _ = run_phase4_pipeline(mapped, fit_artifacts=artifacts)

    X = processed.drop(columns=["isFraud"], errors="ignore")
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_names]
    return model.predict_proba(X)[:, 1]


# =============================================================================
# Pre-filled scenarios — loaded from app_config.json at runtime (set in main())
# Falls back to SCENARIOS_FALLBACK if config is missing.
# =============================================================================
SCENARIOS_FALLBACK = {
    "🛒 Normal grocery": {
        "amt": 47.32, "trans_date_trans_time": "2020-09-15 10:42:00",
        "category": "grocery_pos", "merchant": "fraud_Lebsack-Stoltenberg",
        "cc_num": 4_532_111_222_333_4441, "zip": "83252",
        "lat": 42.1939, "long": -112.4628,
        "merch_lat": 43.150704, "merch_long": -112.154481,
    },
    "💻 Suspicious online purchase": {
        "amt": 1500.00, "trans_date_trans_time": "2020-09-15 03:17:00",
        "category": "shopping_net", "merchant": "fraud_Smith-Davies",
        "cc_num": 4_532_111_222_333_4441, "zip": "83252",
        "lat": 42.1939, "long": -112.4628,
        "merch_lat": 51.5074, "merch_long": -0.1278,
    },
    "⛽ Local gas station": {
        "amt": 28.50, "trans_date_trans_time": "2020-09-15 18:25:00",
        "category": "gas_transport", "merchant": "fraud_Welch-Wisozk",
        "cc_num": 4_532_111_222_333_4441, "zip": "83252",
        "lat": 42.1939, "long": -112.4628,
        "merch_lat": 42.1820, "merch_long": -112.4530,
    },
    "🌃 Late-night round-amount": {
        "amt": 200.00, "trans_date_trans_time": "2020-09-16 04:33:00",
        "category": "misc_net", "merchant": "fraud_Heller-Ondricka",
        "cc_num": 4_532_111_222_333_4441, "zip": "83252",
        "lat": 42.1939, "long": -112.4628,
        "merch_lat": 36.1699, "merch_long": -115.1398,
    },
}


def init_form_state(scenarios: dict):
    """All form fields start empty. Scenario buttons populate them on click."""
    fields = ["amt", "trans_date_trans_time", "category", "cc_num",
              "merchant", "zip", "lat", "long", "merch_lat", "merch_long"]
    for f in fields:
        if f not in st.session_state:
            st.session_state[f] = ""

# =============================================================================
# UI: same structure as dev_app, just calls real predictors
# =============================================================================
def render_field_label(field: str, schema: dict) -> str:
    icon = "🔴" if field in schema["required_inputs"] else "🟢"
    raw_importance = schema["raw_input_importance"]
    field_value = raw_importance.get(field, 0)

    # Normalize to percentage of total — works for both XGBoost (already in [0,1])
    # and LightGBM (raw split counts, e.g. 26987.0)
    total = sum(raw_importance.values())
    pct = (field_value / total * 100) if total > 0 else 0

    if 0 < pct < 1:
        label = "<1% imp."
    else:
        label = f"{pct:.0f}% imp."
    return f"{icon} {field}  ({label})"


def render_sidebar(meta: dict):
    st.sidebar.title("🛡️ Fraud Detection")
    st.sidebar.markdown("**Phase 5 Production**")
    st.sidebar.divider()
    st.sidebar.metric("Test AUC", f"{meta['auc_test']:.4f}")
    st.sidebar.metric("Test AP", f"{meta['ap_test']:.4f}")
    st.sidebar.metric("Decision threshold", f"{meta['threshold']:.2f}")
    st.sidebar.metric("Train fraud rate", f"{meta['fraud_rate_train']*100:.3f}%")
    st.sidebar.divider()
    st.sidebar.caption(
        f"Split: {meta.get('split_used', 'unknown')}\n\n"
        f"Fields marked 🔴 are required (top 80% of model importance). "
        f"Optional fields (🟢) get safe defaults if blank."
    )


# Placeholders shown when fields are empty: "format — example"
PLACEHOLDERS = {
    "amt": "Decimal dollars, e.g. 47.32",
    "trans_date_trans_time": "YYYY-MM-DD HH:MM:SS, e.g. 2020-09-15 10:42:00",
    "category": "(select from dropdown)",
    "cc_num": "Up to 16 digits, e.g. 4532111222333441",
    "merchant": "Merchant name, e.g. fraud_Lebsack-Stoltenberg",
    "zip": "5 digits, e.g. 83252",
    "lat": "Decimal degrees -90 to 90, e.g. 42.1939",
    "long": "Decimal degrees -180 to 180, e.g. -112.4628",
    "merch_lat": "Decimal degrees -90 to 90, e.g. 43.1507",
    "merch_long": "Decimal degrees -180 to 180, e.g. -112.1545",
}


def _parse_float(raw, field_name: str, errors: list):
    """Parse a string to float; append to errors and return None if bad."""
    if raw is None or str(raw).strip() == "":
        return None
    try:
        return float(str(raw).strip())
    except ValueError:
        errors.append(f"'{field_name}' must be a number (got: {raw!r})")
        return None


def _parse_int(raw, field_name: str, errors: list):
    """Parse a string to int (digits only); append to errors and return None if bad."""
    if raw is None or str(raw).strip() == "":
        return None
    cleaned = str(raw).strip().replace(" ", "").replace("-", "")
    try:
        return int(cleaned)
    except ValueError:
        errors.append(f"'{field_name}' must be a whole number (got: {raw!r})")
        return None


def render_single_tab(schema, threshold, model, artifacts, feature_names,
                       scenarios: dict, categories: list):
    st.subheader("Score a single transaction")
    st.caption(
        "All fields start empty. Required fields must be filled; optional fields "
        "improve accuracy but get safe defaults if blank. Use the scenario buttons "
        "to load realistic examples."
    )

    if scenarios:
        st.markdown("**Quick-load a scenario:**")
        sc_cols = st.columns(len(scenarios))
        for col, (name, values) in zip(sc_cols, scenarios.items()):
            if col.button(name, use_container_width=True, key=f"sc_{name}"):
                # Convert all scenario values to strings for the text inputs
                for k, v in values.items():
                    st.session_state[k] = str(v)
                st.rerun()
        if st.button("🧹 Clear all fields", use_container_width=False):
            for k in PLACEHOLDERS.keys():
                st.session_state[k] = ""
            st.rerun()
        st.divider()

    def _render_field(field: str):
        """Render a single field with the correct widget type based on its name."""
        if field == "category":
            cat_options = [""] + list(categories)
            cur_cat = st.session_state.get("category", "")
            st.selectbox(
                render_field_label(field, schema),
                cat_options,
                index=cat_options.index(cur_cat) if cur_cat in cat_options else 0,
                key=field,
                format_func=lambda x: "— select category —" if x == "" else x,
            )
        elif field == "cc_num":
            st.text_input(
                render_field_label(field, schema),
                key=field,
                placeholder=PLACEHOLDERS.get(field, ""),
                help="Numeric cardholder identifier. Up to 16 digits.",
            )
        else:
            st.text_input(
                render_field_label(field, schema),
                key=field,
                placeholder=PLACEHOLDERS.get(field, ""),
            )

    def _render_field_grid(fields: list[str], n_cols: int = 2):
        """Render a list of fields in a grid of n_cols columns, row-major."""
        if not fields:
            return
        cols = st.columns(n_cols)
        for i, field in enumerate(fields):
            with cols[i % n_cols]:
                _render_field(field)

    required_fields = list(schema["required_inputs"])
    optional_fields = list(schema["optional_inputs"])

    # Required section
    st.markdown("##### 🔴 Required fields")
    _render_field_grid(required_fields, n_cols=2)

    # Optional section in expander
    if optional_fields:
        with st.expander(
            f"🟢 Optional fields ({len(optional_fields)} more — improves accuracy)",
            expanded=False,
        ):
            # Use 3 columns if many optional fields (>4), else 2 for cleaner look
            n_cols_opt = 3 if len(optional_fields) > 4 else 2
            _render_field_grid(optional_fields, n_cols=n_cols_opt)

    st.divider()
    if st.button("🔮 Predict", type="primary", use_container_width=True):
        # Parse text inputs into proper types, collecting parse errors
        parse_errors: list[str] = []
        form_inputs = {
            "amt": _parse_float(st.session_state.get("amt"), "amt", parse_errors),
            "trans_date_trans_time": (st.session_state.get("trans_date_trans_time") or "").strip() or None,
            "category": (st.session_state.get("category") or "").strip() or None,
            "cc_num": _parse_int(st.session_state.get("cc_num"), "cc_num", parse_errors),
            "merchant": (st.session_state.get("merchant") or "").strip() or None,
            "zip": (st.session_state.get("zip") or "").strip() or None,
            "lat": _parse_float(st.session_state.get("lat"), "lat", parse_errors),
            "long": _parse_float(st.session_state.get("long"), "long", parse_errors),
            "merch_lat": _parse_float(st.session_state.get("merch_lat"), "merch_lat", parse_errors),
            "merch_long": _parse_float(st.session_state.get("merch_long"), "merch_long", parse_errors),
        }

        if parse_errors:
            for err in parse_errors:
                st.error(f"❌ {err}")
            return

        errors, warnings = common.validate_inputs(form_inputs, schema)
        if errors:
            for err in errors:
                st.error(f"❌ {err}")
            return

        if warnings:
            with st.expander(f"⚠️ {len(warnings)} input notice(s)", expanded=True):
                for w in warnings:
                    st.warning(w)

        transaction, imputed = common.build_transaction_dict(form_inputs, schema)

        with st.spinner("Scoring transaction..."):
            proba = predict_one(transaction, model, artifacts, feature_names)
            is_fraud = proba >= threshold

        # === Result card ===
        verdict_color = "#e74c3c" if is_fraud else "#2ecc71"
        verdict_bg = "rgba(231, 76, 60, 0.12)" if is_fraud else "rgba(46, 204, 113, 0.12)"
        verdict_icon = "🚨" if is_fraud else "✅"
        verdict_text = "FRAUD" if is_fraud else "LEGITIMATE"

        st.markdown(
            f"""
            <div style="
                background: {verdict_bg};
                border-left: 4px solid {verdict_color};
                padding: 18px 22px;
                border-radius: 6px;
                margin: 8px 0 18px 0;
                display: flex;
                align-items: center;
                gap: 18px;
            ">
                <div style="font-size: 38px; line-height: 1;">{verdict_icon}</div>
                <div style="flex: 1;">
                    <div style="font-size: 22px; font-weight: 700; color: {verdict_color};
                                letter-spacing: 0.5px;">{verdict_text}</div>
                    <div style="font-size: 13px; color: #aaa; margin-top: 2px;">
                        Fraud probability {proba*100:.1f}% &nbsp;•&nbsp;
                        Threshold {threshold:.2f} &nbsp;•&nbsp;
                        Model confidence {max(proba, 1 - proba) * 100:.1f}%
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # === Probability gauge ===
        gauge_fig = render_probability_gauge(proba, threshold)
        st.pyplot(gauge_fig, use_container_width=True)
        plt.close(gauge_fig)

        # === Compact metrics row ===
        m1, m2, m3 = st.columns(3)
        m1.metric("Fraud probability", f"{proba*100:.2f}%")
        m2.metric("Decision threshold", f"{threshold:.2f}")
        margin = (proba - threshold) * 100
        m3.metric("Margin", f"{margin:+.1f} pts",
                  help="How far above (+) or below (−) the threshold the prediction is.")

        with st.expander("Prediction details"):
            try:
                dist = common.haversine_km(
                    transaction["lat"], transaction["long"],
                    transaction["merch_lat"], transaction["merch_long"])
            except Exception:
                dist = float("nan")
            st.markdown(f"""
            - Fraud probability: **{proba*100:.2f}%**
            - Card↔merchant distance: **{dist:.1f} km**
            - Amount: **${form_inputs['amt']:.2f}**
            - Hour of day: **{pd.to_datetime(form_inputs['trans_date_trans_time']).hour}h**
            - Category: **{form_inputs['category']}**
            - Imputed fields: **{imputed if imputed else 'none'}**
            """)


def render_batch_tab(schema, threshold, model, artifacts, feature_names):
    st.subheader("Batch scoring from CSV")
    st.caption(
        "Upload a CSV with the Sparkov column schema. Required columns: "
        f"`{', '.join(common.SPARKOV_REQUIRED_BATCH_COLS)}`. "
        f"Optional: `{', '.join(common.SPARKOV_OPTIONAL_BATCH_COLS)}`."
    )

    sample_50 = load_sample_csv("sample_transactions.csv")
    sample_200 = load_sample_csv("sample_transactions_200.csv")

    cu1, cu2, cu3 = st.columns([1, 1, 3])
    with cu1:
        use_sample_50 = st.button("📋 Sample (50 rows)", use_container_width=True,
                                    disabled=sample_50 is None,
                                    help="Quick demo: 15 fraud + 35 legit")
    with cu2:
        use_sample_200 = st.button("📊 Sample (200 rows)", use_container_width=True,
                                    disabled=sample_200 is None,
                                    help="Larger demo: 60 fraud + 140 legit")
    with cu3:
        upload = st.file_uploader("Or upload your own CSV", type="csv")

    df = None
    if use_sample_50 and sample_50 is not None:
        df = sample_50
    elif use_sample_200 and sample_200 is not None:
        df = sample_200
    elif upload is not None:
        try:
            df = pd.read_csv(upload)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            return

    if df is None:
        st.info("👆 Upload a CSV or click one of the sample buttons to begin.")
        return

    st.markdown(f"**Loaded {len(df):,} transactions.**")
    with st.expander("Preview first 10 rows", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)

    errors, warnings, _ = common.validate_batch_csv(df)
    if errors:
        for err in errors:
            st.error(f"❌ {err}")
        return
    if warnings:
        with st.expander(f"⚠️ Data quality notes ({len(warnings)} found)", expanded=True):
            for w in warnings:
                st.warning(w)
            st.info("Rows with missing fields will be imputed and flagged in results "
                    "as `data_quality = imputed`.")

    with st.spinner(f"Scoring {len(df):,} transactions..."):
        df_imp = common.impute_batch(df)
        probas = predict_batch(df_imp, model, artifacts, feature_names)

    results = df_imp.copy()
    results["fraud_probability"] = probas
    results["predicted_fraud"] = (probas >= threshold).astype(int)
    results["data_quality"] = results["had_missing_fields"].map(
        {0: "✓ complete", 1: "⚠️ imputed"}
    )

    stats = common.batch_descriptive_stats(results, threshold)

    st.divider()
    st.markdown("### Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total transactions", f"{stats['n_total']:,}")
    c2.metric("Predicted fraud", f"{stats['n_fraud']:,}",
               delta=f"{stats['fraud_rate']*100:.2f}% of total")
    c3.metric("Predicted legitimate", f"{stats['n_legit']:,}")
    c4.metric("Total amount flagged",
               f"${stats['flagged_amount']:,.0f}",
               delta=f"{stats['flagged_amount_pct']*100:.1f}% of $ total")

    if stats["has_labels"]:
        st.markdown("##### Performance against ground truth")
        l1, l2, l3, l4 = st.columns(4)
        l1.metric("Recall", f"{stats['recall']*100:.1f}%")
        l2.metric("Precision", f"{stats['precision']*100:.1f}%")
        l3.metric("True positives", f"{stats['tp']:,}")
        l4.metric("False negatives", f"{stats['fn']:,}")

    st.divider()
    st.markdown("### Distributions")
    chart_cols = st.columns(2)

    with chart_cols[0]:
        st.markdown("**Probability distribution**")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(results["fraud_probability"], bins=40, color="#3498db",
                 edgecolor="white", alpha=0.85)
        ax.axvline(threshold, color="#e74c3c", linestyle="--", linewidth=2,
                    label=f"Threshold = {threshold:.2f}")
        ax.set_xlabel("Predicted fraud probability")
        ax.set_ylabel("Number of transactions")
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)
        plt.close(fig)

    with chart_cols[1]:
        st.markdown("**Fraud rate by category**")
        cat_stats = (results.groupby("category")
                      .agg(n_total=("predicted_fraud", "count"),
                            n_fraud=("predicted_fraud", "sum"))
                      .assign(rate=lambda d: d["n_fraud"] / d["n_total"])
                      .sort_values("rate", ascending=True))
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(cat_stats.index, cat_stats["rate"] * 100, color="#e74c3c", alpha=0.85)
        ax.set_xlabel("Predicted fraud rate (%)")
        ax.spines[["top", "right"]].set_visible(False)
        for i, v in enumerate(cat_stats["rate"] * 100):
            ax.text(v + 0.1, i, f"{v:.1f}%", va="center", fontsize=8)
        st.pyplot(fig)
        plt.close(fig)

    st.divider()
    st.markdown("### Most suspicious transactions")
    top_n = st.slider("Show top N", min_value=5, max_value=100, value=20, step=5)
    display_cols = [c for c in
                     ["trans_date_trans_time", "amt", "category", "merchant",
                      "fraud_probability", "predicted_fraud", "is_fraud", "data_quality"]
                     if c in results.columns]
    top = results.sort_values("fraud_probability", ascending=False).head(top_n)
    st.dataframe(
        top[display_cols].style.background_gradient(
            subset=["fraud_probability"], cmap="Reds"
        ).format({"fraud_probability": "{:.4f}", "amt": "${:.2f}"}),
        use_container_width=True,
        height=min(top_n * 35 + 50, 600),
    )

    csv_buf = io.StringIO()
    results.sort_values("fraud_probability", ascending=False).to_csv(csv_buf, index=False)
    st.download_button(
        "⬇️ Download full results (CSV)",
        csv_buf.getvalue().encode("utf-8"),
        "fraud_predictions.csv",
        "text/csv",
        type="primary",
    )


def render_about_tab(meta, schema):
    st.subheader("About this system")
    st.markdown(f"""
        ### Model
        - **Type**: LightGBM classifier (Optuna-tuned, trained on Sparkov)
        - **Training split**: {meta.get('split_used', 'unknown')}
        - **Test ROC-AUC**: {meta['auc_test']:.4f}
        - **Test Average Precision**: {meta['ap_test']:.4f}
        - **Decision threshold**: {meta['threshold']:.2f} (F1-optimal)

        ### Pipeline
        1. **Schema mapping**: Sparkov fields → IEEE-CIS roles
        2. **Cleaning**: outlier capping, deduplication, time derivations
        3. **Feature selection**: drop high-null and highly-correlated columns
        4. **Imputation**: median for numerics, "unknown" for categoricals
        5. **Feature extraction**: frequency encoding, PCA on V-features
        6. **Feature engineering**: behavioural ratios, log/cyclic transforms,
           card×location interactions
        7. **Label encoding** for the remaining string columns

        ### Required fields (~80% of model importance)
        {', '.join(f'`{f}`' for f in schema['required_inputs'])}

        ### Optional fields (filled with safe defaults if missing)
        {', '.join(f'`{f}`' for f in schema['optional_inputs'])}
    """)


def main():
    try:
        model, artifacts, feature_names, meta = load_model_and_artifacts()
    except FileNotFoundError as e:
        st.error(
            "❌ Could not load model artifacts.\n\n"
            f"Expected location: `{ARTIFACT_DIR.resolve()}`\n\n"
            "**Local:** make sure the `app_artifacts/` folder (saved by the "
            "Phase 5 training notebook) sits next to `app.py`, or set the "
            "`PHASE5_ARTIFACTS` environment variable to its absolute path.\n\n"
            "**Streamlit Cloud:** the `app_artifacts/` folder must be committed "
            "to the GitHub repository so the container can read it at startup.\n\n"
            f"Underlying error: {e}"
        )
        st.stop()

    schema = common.load_schema(artifact_dir=ARTIFACT_DIR)
    threshold = meta["threshold"]

    # Load real categories + scenarios from Sparkov data (saved during model training)
    app_config = common.load_app_config(artifact_dir=ARTIFACT_DIR)
    scenarios = app_config.get("scenarios") or SCENARIOS_FALLBACK
    categories = app_config.get("real_categories") or [
        "grocery_pos", "grocery_net", "gas_transport", "misc_pos", "misc_net",
        "shopping_pos", "shopping_net", "entertainment", "food_dining",
        "health_fitness", "home", "kids_pets", "personal_care", "travel",
    ]

    init_form_state(scenarios)
    render_sidebar(meta)

    st.title("🛡️ Fraud Detection System")
    st.caption(
        "Trained on Sparkov simulated credit-card transactions "
        "(1.85M rows, ~0.5% fraud, Jan 2019 – Dec 2020)"
    )

    tab_single, tab_batch, tab_about = st.tabs([
        "🔍 Single transaction",
        "📊 Batch scoring",
        "ℹ️ About",
    ])

    with tab_single:
        render_single_tab(schema, threshold, model, artifacts, feature_names,
                          scenarios, categories)
    with tab_batch:
        render_batch_tab(schema, threshold, model, artifacts, feature_names)
    with tab_about:
        render_about_tab(meta, schema)


if __name__ == "__main__":
    main()