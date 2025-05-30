import streamlit as st

import pandas as pd

import numpy as np
 
# pip install mlxtend scikit-surprise seaborn

from mlxtend.frequent_patterns import apriori, association_rules

from surprise import Dataset, Reader, SVD

from surprise.model_selection import cross_validate
 
import matplotlib.pyplot as plt

import seaborn as sns
 
st.title("MBA AND RS")
 
uploaded_file = st.file_uploader("Upload your sales_df.csv", type=["csv"])
 
if uploaded_file:

    try:

        sales_data = pd.read_csv(uploaded_file)

        st.write("Data Preview:")

        st.dataframe(sales_data.head())

    except Exception as e:

        st.error(f"Error reading the CSV file: {e}")

        st.stop()
 
    # Normalize column names for safety

    sales_data.columns = sales_data.columns.str.strip().str.lower()
 
    # Parse dates if present

    if 'delivered_date' in sales_data.columns:

        sales_data['delivered_date'] = pd.to_datetime(sales_data['delivered_date'], errors='coerce')
 
    sales_data.dropna(inplace=True)
 
    st.subheader("Market Basket Analysis")

    if {'order_id', 'sku_code', 'delivered qty'}.issubset(sales_data.columns):

        # Pivot to basket

        basket = (

            sales_data

            .groupby(['order_id', 'sku_code'])['delivered qty']

            .sum()

            .unstack(fill_value=0)

            .applymap(lambda x: 1 if x > 0 else 0)

        )
 
        frequent_itemsets = apriori(basket, min_support=0.02, use_colnames=True)

        rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1.0)
 
        st.write("Frequent Itemsets:")

        st.dataframe(frequent_itemsets.sort_values(by='support', ascending=False))
 
        st.write("Association Rules:")

        st.dataframe(rules.sort_values(by='lift', ascending=False))
 
        # Visualization

        st.subheader("Support vs. Confidence Scatter Plot")

        plt.figure(figsize=(8, 5))

        sns.scatterplot(

            data=rules,

            x='support', y='confidence',

            hue='lift', size='lift', palette='viridis',

            legend='brief'

        )

        plt.xlabel("Support")

        plt.ylabel("Confidence")

        plt.title("Support vs. Confidence")

        st.pyplot(plt)

    else:

        st.error("Missing columns for MBA: need 'Order_Id', 'SKU_Code', and 'Delivered Qty'.")
 
    st.subheader("Recommendation System using SVD")

    if {'salesman_code', 'sku_code', 'delivered qty'}.issubset(sales_data.columns):

        reader = Reader(rating_scale=(1, sales_data['delivered qty'].max()))

        data = Dataset.load_from_df(

            sales_data[['salesman_code', 'sku_code', 'delivered qty']],

            reader

        )

        trainset = data.build_full_trainset()

        algo = SVD()

        cross_validate(algo, data, cv=5, verbose=True)

        algo.fit(trainset)
 
        # Input widget

        min_code = int(sales_data['salesman_code'].min())

        max_code = int(sales_data['salesman_code'].max())

        salesman_code = st.number_input(

            "Enter Salesman Code for recommendations:",

            min_value=min_code, max_value=max_code

        )
 
        if st.button("Get Recommendations"):

            product_ids = sales_data['sku_code'].unique()

            preds = [

                (pid, algo.predict(salesman_code, pid).est)

                for pid in product_ids

            ]

            recommendations = (

                pd.DataFrame(preds, columns=['SKU_Code', 'Predicted Rating'])

                .sort_values(by='Predicted Rating', ascending=False)

                .head(10)

            )

            st.write("Top 10 Recommended Products:")

            st.dataframe(recommendations)

    else:

        st.error("Missing columns for RS: need 'Salesman_Code', 'SKU_Code', and 'Delivered Qty'.")
