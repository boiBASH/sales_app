import streamlit as st
import pandas as pd
import numpy as np

from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.decomposition import TruncatedSVD

import matplotlib.pyplot as plt
import seaborn as sns

st.title("MBA AND RS")

# File uploader
uploaded_file = st.file_uploader("Upload your sales_data.csv", type=["csv"])
if uploaded_file is None:
    st.info("📂 Please upload a CSV file to get started.")
    st.stop()

# Load data
try:
    sales_data = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Error reading the CSV file: {e}")
    st.stop()

# Preview
st.write("**Data Preview:**")
st.dataframe(sales_data.head())

# Normalize column names
sales_data.columns = sales_data.columns.str.strip().str.lower()

# Parse date column if present
if 'delivered_date' in sales_data.columns:
    sales_data['delivered_date'] = pd.to_datetime(
        sales_data['delivered_date'], errors='coerce'
    )

# Drop any rows with missing values
sales_data.dropna(inplace=True)

# ─── Market Basket Analysis
st.subheader("Market Basket Analysis")
mba_cols = {'order_id', 'sku_code', 'delivered qty'}
if mba_cols.issubset(sales_data.columns):
    basket = (
        sales_data
        .groupby(['order_id', 'sku_code'])['delivered qty']
        .sum()
        .unstack(fill_value=0)
        .applymap(lambda x: 1 if x > 0 else 0)
    )

    frequent_itemsets = apriori(basket, min_support=0.02, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1.0)

    st.write("**Frequent Itemsets:**")
    st.dataframe(frequent_itemsets.sort_values('support', ascending=False))

    st.write("**Association Rules:**")
    st.dataframe(rules.sort_values('lift', ascending=False))

    st.subheader("Support vs. Confidence")
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=rules,
        x='support', y='confidence',
        hue='lift', size='lift', palette='viridis',
        legend='brief'
    )
    plt.xlabel("Support")
    plt.ylabel("Confidence")
    plt.title("MBA: Support vs. Confidence")
    st.pyplot(plt)
else:
    st.error(f"Missing columns for MBA; need {mba_cols}")

# ─── Recommendation System via TruncatedSVD
st.subheader("Recommendation System (TruncatedSVD)")
rs_cols = {'salesman_code', 'sku_code', 'delivered qty'}
if rs_cols.issubset(sales_data.columns):
    # Ensure salesman_code is numeric int
    sales_data['salesman_code'] = pd.to_numeric(
        sales_data['salesman_code'], errors='coerce'
    )
    sales_data.dropna(subset=['salesman_code'], inplace=True)
    sales_data['salesman_code'] = sales_data['salesman_code'].astype(int)

    # Pivot to user-item matrix
    user_item = sales_data.pivot_table(
        index='salesman_code',
        columns='sku_code',
        values='delivered qty',
        fill_value=0
    )

    # Fit TruncatedSVD
    svd = TruncatedSVD(n_components=20, random_state=42)
    user_factors = svd.fit_transform(user_item.values)
    item_factors = svd.components_
    preds = np.dot(user_factors, item_factors)
    pred_df = pd.DataFrame(preds, index=user_item.index, columns=user_item.columns)

    # Salesman selection
    salesmen = user_item.index.tolist()
    salesman = st.selectbox("Select Salesman Code:", options=salesmen)

    if st.button("Get Recommendations"):
        actual = user_item.loc[salesman]
        scores = pred_df.loc[salesman].copy()
        # Mask out already purchased items
        scores[actual > 0] = -np.inf
        top10 = scores.nlargest(10).reset_index()
        top10.columns = ['SKU_Code', 'Predicted Score']
        st.write("**Top 10 Recommended Products:**")
        st.dataframe(top10)
else:
    st.error(f"Missing columns for RS; need {rs_cols}")
