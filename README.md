# CartIQ E-commerce Analytics Portal

A deployable Streamlit dashboard for GCC e-commerce customer-journey analytics.

## Included modules

- Executive funnel overview
- Auditable data-cleaning pipeline
- Descriptive and diagnostic checkout analysis
- Classification: KNN, Decision Tree, Random Forest, Gradient Boosting
- Regression: Linear, Ridge, Lasso, Decision Tree, Random Forest
- Abandonment-recovery classification
- K-means buyer clustering with PCA and silhouette score
- Apriori association-rule mining
- Dataset-driven executive strategy playbook

## Run locally

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Upload this folder to a GitHub repository.
2. Create a new Streamlit app and select `app.py` as the entry point.
3. Keep `requirements.txt` in the repository root.
4. The app uses `data/CartIQ_Clean.csv` when bundled, accepts a replacement CSV, and falls back to generated data if the file is absent.

## Required columns for uploaded data

The main modules expect these core fields:

- `device_category`
- `user_region`
- `traffic_source`
- `payment_method_selected`
- `user_tier`
- `primary_product_category`
- `past_purchases_count`
- `items_in_cart`
- `average_item_price`
- `session_duration_secs`
- `checkout_status`
- `historical_clv`

The cleaning pipeline standardises column names automatically and engineers selected missing fields where possible.
