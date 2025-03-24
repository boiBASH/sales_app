import streamlit as st
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate

st.title("MBA AND RS")

uploaded_file = st.file_uploader("sales_data.csv", type=["csv"])

if uploaded_file is not None:
    try:
        sales_data = pd.read_csv(uploaded_file)
        st.write("Data Preview:")
        st.dataframe(sales_data.head())
    except Exception as e:
        st.error(f"Error reading the CSV file: {e}")
    
    
    if 'Delivered_date' in sales_data.columns:
        sales_data['Delivered_date'] = pd.to_datetime(sales_data['Delivered_date'])

    sales_data.dropna(inplace=True)
    
    
    st.subheader("Market Basket Analysis")
    if 'Order_Id' in sales_data.columns and 'SKU_Code' in sales_data.columns:
        basket = sales_data.groupby(['Order_Id', 'SKU_Code'])['Delivered Qty'].sum().unstack().fillna(0)
        basket = basket.applymap(lambda x: 1 if x > 0 else 0)
        
        frequent_itemsets = apriori(basket, min_support=0.02, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1.0)
        
        st.write("Frequent Itemsets:")
        st.dataframe(frequent_itemsets.sort_values(by='support', ascending=False))
        
        st.write("Association Rules:")
        st.dataframe(rules.sort_values(by='lift', ascending=False))
        
        # Visualization
        st.subheader("Support vs. Confidence Scatter Plot")
        plt.figure(figsize=(8, 5))
        sns.scatterplot(x=rules['support'], y=rules['confidence'], hue=rules['lift'], size=rules['lift'], palette='viridis')
        plt.xlabel("Support")
        plt.ylabel("Confidence")
        plt.title("Support vs. Confidence")
        st.pyplot(plt)
    else:
        st.error("Required columns 'Order_Id' and 'SKU_Code' are missing in the dataset.")
    
    
    st.subheader("Recommendation System using SVD")
    if {'Salesman_Code', 'SKU_Code', 'Delivered Qty'}.issubset(sales_data.columns):
        reader = Reader(rating_scale=(1, sales_data['Delivered Qty'].max()))
        data = Dataset.load_from_df(sales_data[['Salesman_Code', 'SKU_Code', 'Delivered Qty']], reader)
        trainset = data.build_full_trainset()
        
        model = SVD()
        cross_validate(model, data, cv=5)
        model.fit(trainset)
        
        salesman_code = st.number_input("Enter Salesman Code for recommendations:", min_value=int(sales_data['Salesman_Code'].min()), max_value=int(sales_data['Salesman_Code'].max()))
        
        if st.button("Get Recommendations"):
            product_ids = sales_data['SKU_Code'].unique()
            predictions = [model.predict(customer_id, pid).est for pid in product_ids]
            recommendations = pd.DataFrame({'SKU_Code': product_ids, 'Predicted Rating': predictions})
            recommendations = recommendations.sort_values(by='Predicted Rating', ascending=False).head(10)
            
            st.write("Top 10 Recommended Products:")
            st.dataframe(recommendations)
    else:
        st.error("Required columns 'Salesman_Code', 'SKU_Code', and 'Delivered Qty' are missing in the dataset.")
