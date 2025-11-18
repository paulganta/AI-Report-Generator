import io
import json
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import google.generativeai as genai

# =============================================================
# Streamlit — Olist One-Click App (with optional OpenAI integration)
# Upload CSVs → Clean/Join → KPIs & Charts → Factsheet JSON → AI Report
# =============================================================
# How to run locally:
# 1) pip install streamlit pandas matplotlib numpy openai
# 2) streamlit run streamlit_app.py
#    (Set OPENAI_API_KEY in your environment or paste it in the sidebar)
# =============================================================

st.set_page_config(page_title="Olist AI Reports", layout="wide")
st.title("📦 Olist AI Reports — One-Click ETL, KPIs, Charts & Factsheet")
st.caption(
    "Upload the nine Olist CSVs (from Kaggle) and get ready-to-use marts, plots, "
    "a compact factsheet, and an optional LLM report."
)

# -------------------------------
# OpenAI helper
# -------------------------------

def call_gemini(prompt: str, model: str = "gemini-1.5-flash", temperature: float = 0.2, api_key: str = None) -> str:
    """
    Call Google Gemini using google-generativeai.
    Returns the text content or raises on error.
    """
    # Priority: explicit key → GEMINI_API_KEY → GOOGLE_API_KEY
    api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No Gemini API key provided. Set GEMINI_API_KEY/GOOGLE_API_KEY or enter it in the sidebar."
        )

    genai.configure(api_key=api_key)

    model_obj = genai.GenerativeModel(model)
    resp = model_obj.generate_content(
        prompt,
        generation_config={"temperature": temperature},
    )

    # resp.text is the combined text output
    return resp.text


# -------------------------------
# Upload & Parsing Helpers
# -------------------------------

REQUIRED_FILES = {
    "olist_orders_dataset.csv": "orders",
    "olist_order_items_dataset.csv": "items",
    "olist_order_payments_dataset.csv": "payments",
    "olist_order_reviews_dataset.csv": "reviews",
    "olist_products_dataset.csv": "products",
    "olist_customers_dataset.csv": "customers",
    "olist_geolocation_dataset.csv": "geolocation",  # optional for maps
    "olist_sellers_dataset.csv": "sellers",  # optional for item-level joins
    "product_category_name_translation.csv": "translation",
}

ORDER_DATE_COLS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]
REVIEW_DATE_COLS = ["review_creation_date", "review_answer_timestamp"]


def _read_csv(file, parse_dates=None):
    if parse_dates is None:
        return pd.read_csv(file)
    return pd.read_csv(file, parse_dates=parse_dates)


def load_inputs() -> Dict[str, pd.DataFrame]:
    st.subheader("1) Upload your CSVs")
    cols = st.columns(3)
    uploads: Dict[str, io.BytesIO] = {}

    for i, (fname, key) in enumerate(REQUIRED_FILES.items()):
        with cols[i % 3]:
            uploads[key] = st.file_uploader(label=fname, type="csv", key=f"upl_{key}")

    ready = all(uploads[k] is not None for k in REQUIRED_FILES.values())
    if not ready:
        st.info("⬆️ Please upload all required CSVs to continue.")
        return {}

    df_orders = _read_csv(uploads["orders"], parse_dates=ORDER_DATE_COLS)
    df_reviews = _read_csv(uploads["reviews"], parse_dates=REVIEW_DATE_COLS)
    df_items = _read_csv(uploads["items"])
    df_pays = _read_csv(uploads["payments"])
    df_products = _read_csv(uploads["products"])
    df_custs = _read_csv(uploads["customers"])
    df_geos = _read_csv(uploads["geolocation"])
    df_sellers = _read_csv(uploads["sellers"])
    df_trans = _read_csv(uploads["translation"])

    return {
        "orders": df_orders,
        "reviews": df_reviews,
        "items": df_items,
        "payments": df_pays,
        "products": df_products,
        "customers": df_custs,
        "geolocation": df_geos,
        "sellers": df_sellers,
        "translation": df_trans,
    }


def dedup(df: pd.DataFrame, name: str) -> pd.DataFrame:
    before = len(df)
    df2 = df.drop_duplicates()
    after = len(df2)
    if before != after:
        st.write(f"✅ De-dup **{name}**: removed {before-after} rows")
    return df2


def integrity_report(child_df, child_key, parent_df, parent_key, label) -> int:
    missing = child_df[~child_df[child_key].astype(str).isin(parent_df[parent_key].astype(str))]
    cnt = len(missing)
    if cnt > 0:
        st.warning(f"Integrity: **{label}** — {cnt} rows reference missing `{parent_key}`")
    else:
        st.write(f"✔️ Integrity: **{label}** — no missing refs")
    return cnt


def build_marts(dfs: Dict[str, pd.DataFrame]):
    """
    Return (monthly, category_perf, fact_orders, dim_customer, dim_product).
    """
    orders = dedup(dfs["orders"], "orders")
    items = dedup(dfs["items"], "order_items")
    pays = dedup(dfs["payments"], "payments")
    reviews = dedup(dfs["reviews"], "reviews")
    products = dedup(dfs["products"], "products")
    custs = dedup(dfs["customers"], "customers")
    trans = dedup(dfs["translation"], "translation")

    # IDs → string
    id_cols = ["order_id", "customer_id", "customer_unique_id", "product_id", "seller_id"]
    for c in id_cols:
        for df in [orders, items, products, custs, reviews, pays]:
            if c in df.columns:
                df[c] = df[c].astype(str)

    # Numerics
    for c in ["price", "freight_value"]:
        if c in items.columns:
            items[c] = pd.to_numeric(items[c], errors="coerce")
    if "payment_value" in pays.columns:
        pays["payment_value"] = pd.to_numeric(pays["payment_value"], errors="coerce")

    # Integrity
    integrity_report(items, "order_id", orders, "order_id", "items → orders")
    integrity_report(items, "product_id", products, "product_id", "items → products")
    integrity_report(reviews, "order_id", orders, "order_id", "reviews → orders")
    integrity_report(pays, "order_id", orders, "order_id", "payments → orders")
    integrity_report(orders, "customer_id", custs, "customer_id", "orders → customers")

    # Category translations
    products = products.merge(trans, how="left", on="product_category_name")
    products = products.rename(columns={"product_category_name_english": "category_en"})
    dim_product = products[["product_id", "product_category_name", "category_en"]].copy()

    # Order-level features
    items["line_revenue"] = items["price"]
    order_rev = items.groupby("order_id", as_index=False).agg(
        revenue=("line_revenue", "sum"),
        items_cnt=("order_item_id", "count"),
    )
    freight = items.groupby("order_id", as_index=False).agg(
        freight_value=("freight_value", "sum")
    )
    pay_tot = (
        pays.groupby("order_id", as_index=False)["payment_value"]
        .sum()
        .rename(columns={"payment_value": "payment_value_total"})
    )

    ordx = (
        orders.merge(order_rev, on="order_id", how="left")
        .merge(freight, on="order_id", how="left")
        .merge(pay_tot, on="order_id", how="left")
        .merge(
            custs[["customer_id", "customer_unique_id", "customer_city", "customer_state"]],
            on="customer_id",
            how="left",
        )
    )

    ordx["delivery_days"] = (
        ordx["order_delivered_customer_date"] - ordx["order_purchase_timestamp"]
    ).dt.days
    ordx["order_month"] = ordx["order_purchase_timestamp"].dt.to_period("M").astype(str)
    ordx["freight_share"] = ordx["freight_value"] / (ordx["revenue"].fillna(0) + 1e-9)

    q90 = ordx["delivery_days"].dropna().quantile(0.9) if ordx["delivery_days"].notna().any() else np.nan
    has_est = ordx["order_estimated_delivery_date"].notna() & ordx["order_delivered_customer_date"].notna()
    ordx["is_late"] = False
    ordx.loc[has_est, "is_late"] = (
        ordx.loc[has_est, "order_delivered_customer_date"]
        > ordx.loc[has_est, "order_estimated_delivery_date"]
    )
    no_est = ~has_est & ordx["delivery_days"].notna()
    if not np.isnan(q90):
        ordx.loc[no_est, "is_late"] = ordx.loc[no_est, "delivery_days"] > q90

    delivered = ordx[ordx["order_status"] == "delivered"].copy()

    # Customer features & cohorts
    first_order = delivered.groupby("customer_unique_id")["order_purchase_timestamp"].min().rename(
        "first_order_date"
    )
    cust_feat = delivered.groupby("customer_unique_id").agg(
        orders_cnt=("order_id", "nunique"),
        last_order=("order_purchase_timestamp", "max"),
    ).join(first_order)
    cust_feat["repeat_customer"] = cust_feat["orders_cnt"] > 1
    cust_feat["cohort_month"] = cust_feat["first_order_date"].dt.to_period("M").astype(str)

    dim_customer = (
        custs.merge(cust_feat.reset_index(), on="customer_unique_id", how="left")
        .rename(columns={"customer_city": "city", "customer_state": "state"})
    )

    # Monthly KPIs
    monthly = delivered.groupby("order_month").agg(
        revenue=("revenue", "sum"),
        orders=("order_id", "nunique"),
        avg_delivery_days=("delivery_days", "mean"),
        late_ratio=("is_late", "mean"),
    ).reset_index()
    monthly["aov"] = monthly["revenue"] / monthly["orders"]

    # Reviews by month
    revx = reviews.merge(delivered[["order_id", "order_month"]], on="order_id", how="left")
    review_month = revx.groupby("order_month").agg(
        avg_review_score=("review_score", "mean"),
        reviews=("order_id", "count"),
    ).reset_index()
    monthly = monthly.merge(review_month, on="order_month", how="left")

    # Category performance
    items_cat = (
        items.merge(products[["product_id", "category_en"]], on="product_id", how="left")
        .merge(orders[["order_id", "order_purchase_timestamp", "order_status"]], on="order_id", how="left")
    )
    items_cat = items_cat[items_cat["order_status"] == "delivered"].copy()
    items_cat["order_month"] = items_cat["order_purchase_timestamp"].dt.to_period("M").astype(str)

    category_perf = items_cat.groupby(["order_month", "category_en"]).agg(
        revenue=("price", "sum"),
        items=("order_item_id", "count"),
        avg_freight=("freight_value", "mean"),
    ).reset_index()

    return monthly, category_perf, delivered, dim_customer, dim_product


def build_factsheet(monthly: pd.DataFrame) -> Dict:
    m = monthly.sort_values("order_month").tail(3).copy()
    facts = {
        "periods": m["order_month"].tolist(),
        "revenue": [float(x) if pd.notnull(x) else None for x in m["revenue"].round(2)],
        "orders": [int(x) if pd.notnull(x) else None for x in m["orders"]],
        "aov": [float(x) if pd.notnull(x) else None for x in m["aov"].round(2)],
        "avg_delivery_days": [float(x) if pd.notnull(x) else None for x in m["avg_delivery_days"].round(2)],
        "late_ratio": [float(x) if pd.notnull(x) else None for x in m["late_ratio"].round(3)],
        "avg_review_score": [
            float(x) if pd.notnull(x) else None for x in m["avg_review_score"].round(2)
        ],
    }
    return facts


def build_top_categories_growth(category_perf: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """
    Compute last-2-month growth by category and return top-k rows with:
    category_en, prev_month, last_month, growth, period_prev, period_last
    """
    if category_perf.empty:
        return pd.DataFrame(
            columns=["category_en", "prev_month", "last_month", "growth", "period_prev", "period_last"]
        )

    df = category_perf.copy()
    df = df.sort_values("order_month")
    months = df["order_month"].dropna().unique()
    if len(months) < 2:
        return pd.DataFrame(
            columns=["category_en", "prev_month", "last_month", "growth", "period_prev", "period_last"]
        )

    prev, last = months[-2], months[-1]
    piv = (
        df[df["order_month"].isin([prev, last])]
        .pivot_table(index="category_en", columns="order_month", values="revenue", aggfunc="sum")
        .fillna(0)
    )

    piv["growth"] = piv[last] - piv[prev]
    out = piv.sort_values("growth", ascending=False).head(k).reset_index()
    out = out.rename(columns={prev: "prev_month", last: "last_month"})
    out["period_prev"] = prev
    out["period_last"] = last

    return out[["category_en", "prev_month", "last_month", "growth", "period_prev", "period_last"]]


def make_llm_prompt(facts: Dict, top_growth: pd.DataFrame) -> str:
    top_list = []
    if not top_growth.empty:
        for _, r in top_growth.iterrows():
            top_list.append(
                {
                    "category": str(r["category_en"]) if pd.notnull(r["category_en"]) else "",
                    "prev_revenue": float(r["prev_month"]) if pd.notnull(r["prev_month"]) else 0.0,
                    "last_revenue": float(r["last_month"]) if pd.notnull(r["last_month"]) else 0.0,
                    "growth": float(r["growth"]) if pd.notnull(r["growth"]) else 0.0,
                    "period_prev": r["period_prev"],
                    "period_last": r["period_last"],
                }
            )

    instructions = (
        "You are a Business Analyst. Using the JSON facts (KPIs for the last 3 months) and the top 5 "
        "categories by revenue growth (if provided), write a concise monthly report with: "
        "(1) Executive Summary with MoM deltas, "
        "(2) Sales & Category Performance, "
        "(3) Operations (delivery, late%), "
        "(4) Customer Voice (reviews/sentiment), "
        "(5) three actionable recommendations. "
        "Keep it under 250 words, use bullet points, and quantify every claim."
    )

    payload = {
        "facts": facts,
        "top_categories_growth": top_list,
        "instructions": instructions,
    }
    return json.dumps(payload, indent=2)


def auto_report_from_facts(facts: Dict) -> str:
    p = facts.get("periods", [])
    rev = facts.get("revenue", [])
    ords = facts.get("orders", [])
    aov = facts.get("aov", [])
    dd = facts.get("avg_delivery_days", [])
    late = facts.get("late_ratio", [])
    rs = facts.get("avg_review_score", [])

    def delta(lst):
        if len(lst) >= 2 and lst[-1] is not None and lst[-2] is not None:
            return lst[-1] - lst[-2]
        return None

    def fmt_pct(x):
        return f"{x*100:.1f}%" if x is not None else "—"

    def fmt(x):
        return f"{x:,.2f}" if isinstance(x, (int, float)) and x is not None else "—"

    lines = []
    lines.append("**Executive Summary (last 3 months)**")
    lines.append(f"- Periods: {', '.join(p) if p else '—'}")
    if rev:
        lines.append(
            f"- Revenue (last): ${fmt(rev[-1])} (MoM Δ: {fmt(delta(rev))}). "
            f"Orders: {fmt(ords[-1])}; AOV: ${fmt(aov[-1])}."
        )
    if dd:
        lines.append(
            f"- Operations: Avg delivery days {fmt(dd[-1])} (MoM Δ: {fmt(delta(dd))}); "
            f"Late% {fmt_pct(late[-1]) if late else '—'}."
        )
    if rs:
        lines.append(
            f"- Customer Voice: Avg review score {fmt(rs[-1])} (MoM Δ: {fmt(delta(rs))})."
        )

    lines.append("")
    lines.append("**Sales & Category Performance** — see charts for top categories.")
    lines.append("**Operations** — reduce slow lanes, watch freight share.")
    lines.append("**Customer Voice** — sample low-star reviews to spot themes.")
    lines.append("**Recommendations** — double-down on growers; fix late lanes; boost review prompts.")

    return "\n".join(lines)


# -------------------------------
# Main App Flow
# -------------------------------

with st.sidebar:
    st.header("🔐 Gemini (Google AI, optional)")
    user_gemini_key = st.text_input(
        "Gemini API Key",
        type="password",
        placeholder="AIza...",
        help="Get one free at ai.google.dev; or set GEMINI_API_KEY / GOOGLE_API_KEY in environment.",
    )
    gemini_model = st.selectbox(
        "Gemini model",
        ["gemini-pro", "gemini-1.0-pro-001"],
        index=0,
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)

dfs = load_inputs()
if not dfs:
    st.stop()

with st.spinner("Building marts and charts…"):
    monthly, category_perf, fact_orders, dim_customer, dim_product = build_marts(dfs)

st.subheader("2) Overview — Monthly KPIs")
st.dataframe(monthly.sort_values("order_month").reset_index(drop=True))

# Downloads for marts/dims
st.download_button(
    "⬇️ Download monthly_kpis.csv",
    data=monthly.to_csv(index=False),
    file_name="monthly_kpis.csv",
    mime="text/csv",
)
st.download_button(
    "⬇️ Download category_perf.csv",
    data=category_perf.to_csv(index=False),
    file_name="category_perf.csv",
    mime="text/csv",
)
st.download_button(
    "⬇️ Download fact_orders.csv",
    data=fact_orders.to_csv(index=False),
    file_name="fact_orders.csv",
    mime="text/csv",
)
st.download_button(
    "⬇️ Download dim_customer.csv",
    data=dim_customer.to_csv(index=False),
    file_name="dim_customer.csv",
    mime="text/csv",
)
st.download_button(
    "⬇️ Download dim_product.csv",
    data=dim_product.to_csv(index=False),
    file_name="dim_product.csv",
    mime="text/csv",
)

# -------------------------------
# Charts
# -------------------------------

st.subheader("3) Charts")

if not monthly.empty:
    m_sorted = monthly.sort_values("order_month")

    def line_plot(x, y, title, ylabel):
        fig = plt.figure()
        plt.plot(m_sorted[x], m_sorted[y])
        plt.title(title)
        plt.xlabel(x)
        plt.ylabel(ylabel)
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig

    col1, col2, col3 = st.columns(3)
    with col1:
        st.pyplot(line_plot("order_month", "revenue", "Revenue by Month", "Revenue"))
    with col2:
        st.pyplot(line_plot("order_month", "orders", "Orders by Month", "Orders"))
    with col3:
        st.pyplot(line_plot("order_month", "aov", "AOV by Month", "AOV"))

    col4, col5 = st.columns(2)
    with col4:
        st.pyplot(line_plot("order_month", "avg_delivery_days", "Avg Delivery Days", "Days"))
    with col5:
        if "late_ratio" in monthly.columns:
            st.pyplot(line_plot("order_month", "late_ratio", "Late Delivery Ratio", "Ratio"))

if not category_perf.empty:
    st.markdown("**Top 5 Categories by Total Revenue**")
    top5 = (
        category_perf.groupby("category_en")["revenue"]
        .sum()
        .sort_values(ascending=False)
        .head(5)
        .index.tolist()
    )
    cat_pvt = (
        category_perf[category_perf["category_en"].isin(top5)]
        .pivot_table(index="order_month", columns="category_en", values="revenue", aggfunc="sum")
        .fillna(0)
        .sort_index()
    )

    fig = plt.figure()
    for col in cat_pvt.columns:
        plt.plot(cat_pvt.index, cat_pvt[col], label=str(col))
    plt.title("Top 5 Categories — Monthly Revenue")
    plt.xlabel("order_month")
    plt.ylabel("revenue")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    st.pyplot(fig)

# -------------------------------
# Factsheet & Reports
# -------------------------------

st.subheader("4) LLM Factsheet + Reports")

facts = build_factsheet(monthly)
facts_json = json.dumps(facts, indent=2)

st.markdown("**Factsheet JSON (last 3 months):**")
st.code(facts_json, language="json")
st.download_button(
    "⬇️ Download monthly_facts.json",
    data=facts_json.encode("utf-8"),
    file_name="monthly_facts.json",
    mime="application/json",
)

st.markdown("**Auto-generated Monthly Report (rule-based, no API):**")
st.markdown(auto_report_from_facts(facts))

st.markdown("---")
st.markdown("### ✨ Generate AI Report with Gemini (optional)")

top_growth = build_top_categories_growth(category_perf, k=5)
prompt_json = make_llm_prompt(facts, top_growth)

st.markdown("**Prompt payload sent to Gemini:**")
st.code(prompt_json, language="json")

if st.button("🚀 Generate with Gemini"):
    try:
        gemini_prompt = (
            "You are a precise, data-grounded Business Analyst.\n\n"
            "Using the JSON below, write a concise monthly report with:\n"
            "1. Executive Summary with MoM deltas,\n"
            "2. Sales & Category Performance,\n"
            "3. Operations (delivery, late%),\n"
            "4. Customer Voice,\n"
            "5. Three actionable recommendations.\n\n"
            "Keep it under 250 words, use bullet points, and quantify every claim.\n\n"
            "JSON:\n"
            f"{prompt_json}"
        )

        gemini_key = user_gemini_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        ai_text = call_gemini(
            gemini_prompt,
            model=gemini_model,
            temperature=temperature,
            api_key=gemini_key,
        )

        st.success("Gemini report generated!")
        st.markdown(ai_text)
        st.download_button(
            "⬇️ Download AI Report (Markdown)",
            data=ai_text.encode("utf-8"),
            file_name="ai_report_gemini.md",
            mime="text/markdown",
        )
    except Exception as e:
        st.error(f"Gemini call failed: {e}")
