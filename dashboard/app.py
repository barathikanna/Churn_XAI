# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import cloudpickle
from lime.lime_tabular import LimeTabularExplainer
import streamlit.components.v1 as components

st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")
st.title("Customer Churn Prediction & Counterfactual Analysis")

# --------------------------
# Load Data
# --------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(r"D:\Churn_project\Latest\causal_discount_churn_DAG_clean.csv")
    cf_df = pd.read_csv(r"D:\Churn_project\Latest\DiCE_counterfactuals_all.csv")
    X_train_raw = pd.read_csv(r"D:\Churn_project\Latest\X_train_raw.csv")

    for d in [df, X_train_raw]:
        if "review_text" in d.columns:
            d["review_text"] = d["review_text"].fillna("").astype(str)

    df["customer_id"] = df["customer_id"].astype(str)
    cf_df["customer_id"] = cf_df["customer_id"].astype(str)
    if cf_df["customer_id"].str.isnumeric().all():
        cf_df["customer_id"] = cf_df["customer_id"].apply(lambda x: f"CUST_{int(x):06d}")

    return df, cf_df, X_train_raw

df, cf_df, X_train_raw = load_data()

flipper_ids = cf_df.loc[cf_df["is_best_feasible"] == 1, "customer_id"].unique().tolist()

# --------------------------
# Load Model
# --------------------------
@st.cache_resource
def load_model_and_processed():
    with open(r"D:\Churn_project\Latest\xgb_pipeline.pkl", "rb") as f:
        pipeline = cloudpickle.load(f)
    fitted_model = pipeline.named_steps['classifier']
    X_train_processed = pipeline.named_steps['preprocessor'].transform(X_train_raw)
    if hasattr(X_train_processed, "todense"):
        X_train_processed = X_train_processed.todense()
    X_train_processed = np.array(X_train_processed)
    return pipeline, fitted_model, X_train_processed

xgb_pipeline, fitted_model, X_train_processed = load_model_and_processed()

model_features = [
    'age', 'tenure_months', 'hour_spend_on_app', 'visits_last_month',
    'avg_purchase_value', 'number_devices', 'delivery_distance_km',
    'satisfaction_score', 'loyalty_score', 'discount_offer',
    'gender', 'preferred_payment', 'preferred_category', 'review_text'
]

# --------------------------
# Sidebar Navigation
# --------------------------
page = st.sidebar.radio("Select Page", ["Summary", "Individual Customers"])

# ==============================================================
# PAGE 1: SUMMARY
# ==============================================================
if page == "Summary":
    st.header("Overall Summary")
    
    total_customers = len(df)
    total_revenue = df["avg_purchase_value"].sum()
    total_visits = df["visits_last_month"].sum()
    churn_rate = df["churned"].mean() * 100
    churn_count = (df["churned"] == 1).sum()
    potential_loss = df.loc[df["churned"] == 1, "avg_purchase_value"].sum()
    feasible_customers = len(flipper_ids)
    potential_saved = df[df["customer_id"].isin(flipper_ids)]["avg_purchase_value"].sum()

    kpi1, kpi2, kpi3, kpi7 = st.columns(4)
    kpi1.metric("Total Customers", total_customers)
    kpi2.metric("Total Revenue", f"£{total_revenue:,.2f}")
    kpi3.metric("Total Visits", int(total_visits))
    kpi7.metric("Total churners", churn_count)

    kpi4, kpi5, kpi6, kpi8 = st.columns(4)
    kpi4.metric("% Customers Churned", f"{churn_rate:.2f}%")
    kpi5.metric("Potential Revenue Loss", f"£{potential_loss:,.2f}")
    kpi6.metric("Flippers (Best Feasible)", feasible_customers)
    kpi8.metric("Potential Revenue Saved", f"£{potential_saved:,.2f}")

    st.subheader("SHAP Global Feature Importance")
    explainer = shap.Explainer(fitted_model)
    shap_values = explainer(X_train_processed)
    feature_names = xgb_pipeline.named_steps['preprocessor'].get_feature_names_out()
    shap.summary_plot(shap_values, X_train_processed, feature_names=feature_names, show=False)
    st.pyplot(plt.gcf())
    plt.clf()
    
    # Possible Fixes Table
    st.subheader("Possible Fixes (Feasible Counterfactuals)")
    possible_fixes = cf_df[cf_df["is_feasible"] == 1]
    st.dataframe(possible_fixes, use_container_width=True)

# ==============================================================
# PAGE 2: INDIVIDUAL CUSTOMERS
# ==============================================================
elif page == "Individual Customers":
    st.header("Individual Customer Analysis")

    only_churned = st.sidebar.checkbox("Show only churned customers", value=True)
    only_flippers = st.sidebar.checkbox("Show only flippers (best feasible)", value=False)

    filtered_df = df.copy()
    if only_churned:
        filtered_df = filtered_df[filtered_df["churned"] == 1]
    if only_flippers:
        filtered_df = filtered_df[filtered_df["customer_id"].isin(flipper_ids)]

    customer_id = st.sidebar.selectbox("Select Customer", filtered_df["customer_id"].unique())
    customer = df[df["customer_id"] == customer_id].iloc[0]
    
    # Get discount suggestion from best feasible counterfactual
    cust_cf_best = cf_df[(cf_df["customer_id"] == customer_id) & (cf_df["is_best_feasible"] == 1)]
    if not cust_cf_best.empty and "discount_suggestion" in cust_cf_best.columns:
        discount_suggestion = cust_cf_best.iloc[0]["discount_suggestion"]
    else:
        discount_suggestion = "N/A"


    st.subheader(f"Customer: {customer_id}")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Loyalty Score", round(customer["loyalty_score"], 3))
        st.metric("Discount Offered", customer.get("discount_offer", "N/A"))
        st.metric("Satisfaction Score", customer["satisfaction_score"])
    with col2:
        st.metric("Avg Purchase Value", f"${customer.get('avg_purchase_value', 0)}")
        st.metric("Visits Last Month", int(customer.get("visits_last_month", 0)))
        st.metric("Delivery Distance (km)", customer.get("delivery_distance_km", 0))
    with col3:
        st.metric("Age", customer["age"])
        st.metric("Gender", customer["gender"])
        st.metric("Preferred Payment", customer["preferred_payment"])
        

    if pd.notna(customer.get("review_text", "")) and customer["review_text"].strip() != "":
        st.markdown(f"**Review:** {customer['review_text']}")
        
    st.metric("Discount Suggestion", discount_suggestion)

    st.markdown("---")
    st.subheader("Counterfactual Explanations")
    cust_cf = cf_df[cf_df["customer_id"] == customer_id].copy()
    if not cust_cf.empty:
        drop_cols = [
            "customer_id", "preferred_category", "discount_suggestion",
            "is_feasible", "is_best_feasible",
            "tenure_months", "hour_spend_on_app", "visits_last_month",
            "avg_purchase_value", "number_devices", "preferred_payment", "delivery_distance_km", "age","gender"
        ]
        display_cf = cust_cf.drop(columns=[c for c in drop_cols if c in cust_cf.columns], errors="ignore")
        if "change_importance" in display_cf.columns:
            display_cf = display_cf.sort_values("change_importance", ascending=False)
        st.dataframe(display_cf.style.set_properties(**{
            "background-color": "#f0f0f0",
            "color": "#000000"
        }), use_container_width=True)
    else:
        st.warning("No counterfactuals available for this customer.")

    st.markdown("---")
    st.subheader("SHAP Force Plot")
    X_cust_raw = customer[model_features].to_frame().T.copy()
    X_cust_raw["review_text"] = X_cust_raw["review_text"].fillna("").astype(str)
    X_cust_processed = xgb_pipeline.named_steps['preprocessor'].transform(X_cust_raw)
    if hasattr(X_cust_processed, "todense"):
        X_cust_processed = X_cust_processed.todense()
    X_cust_processed = np.array(X_cust_processed)

    explainer_local = shap.Explainer(fitted_model)
    shap_values_local = explainer_local(X_cust_processed)
    fig = shap.force_plot(
        base_value=shap_values_local.base_values[0],
        shap_values=shap_values_local.values[0],
        features=X_cust_processed[0],
        feature_names=xgb_pipeline.named_steps['preprocessor'].get_feature_names_out(),
        matplotlib=True,
        show=False
    )
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")
    st.subheader("LIME Explanation")
    feature_names = xgb_pipeline.named_steps['preprocessor'].get_feature_names_out()
    lime_explainer = LimeTabularExplainer(
        training_data=X_train_processed,
        feature_names=feature_names,
        class_names=['Not Churned', 'Churned'],
        mode='classification'
    )

    def predict_fn(numeric_data):
        return fitted_model.predict_proba(numeric_data)

    lime_exp = lime_explainer.explain_instance(
        X_cust_processed[0],
        predict_fn,
        num_features=10
    )

    # Save LIME HTML and embed
    lime_exp.save_to_file("lime_explanation.html")
    with open("lime_explanation.html", "r", encoding="utf-8") as f:
        lime_html = f.read()
    components.html(lime_html, height=800, scrolling=True)
