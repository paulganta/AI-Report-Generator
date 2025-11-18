
# Olist AI Reports — CSV Build Outputs

This run created the following assets under `/Users/killua/Downloads/Olist-ai-reports`:

- `data_raw/`  — Original CSVs copied from /mnt/data
- `data_clean/` — Dimensions like `dim_customer.csv`, `dim_product.csv`
- `marts/`     — Analytical marts (CSV):
  - `monthly_kpis.csv` — revenue, orders, aov, avg_delivery_days, late_ratio, avg_review_score
  - `category_perf.csv` — category × month metrics
  - `fact_orders.csv` — one row per delivered order with key features
- `factsheets/` — `monthly_facts.json` for AI prompts
- `plots/` — PNG charts for your slides

Re-run this build after updating any CSVs; files will be replaced as needed.
