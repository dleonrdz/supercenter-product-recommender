import streamlit as st
import sys
import os
import random
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from supercenter_product_recommender.db_utilities import read_table
from supercenter_product_recommender.embedding_process import data_preparation_orders
from supercenter_product_recommender.recommenders import get_top_n_recommendations_faiss, get_top_n_recommendations_pinecone_batch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
image_path = os.path.join(PROJECT_ROOT,"streamlit_app/images/logo.jpg")
products_df = read_table('processed_products_data')
products_df['product_id'] = products_df['product_id'].astype(str)

def map_product_info_through_name(products, products_df):
    product_info = products_df[products_df['product_name'].isin(products)]
    return product_info[['product_name', 'product_id', 'department']]

def map_product_info_through_id(products, products_df):
    product_info = products_df[products_df['product_id'].isin(products)]
    return product_info[['product_name', 'product_id', 'department']]


# Title of the app
st.title("Welcome to your SuperCenter Online!")

# Sidebar for selecting products
st.sidebar.title("Select Products to Add to Cart")
products = products_df['product_name'].tolist()
selected_products = st.sidebar.multiselect("Add or remove products:", products)

# Display selected products in the main area with enumeration
st.header("Cart")
if selected_products:
    st.write("Products in your cart:")
    for i, product in enumerate(selected_products, 1):
        st.write(f"{i}. {product}")

    # Create dataframe with the information displayed in the cart
    product_info = map_product_info_through_name(selected_products, products_df)
    cart_data = {
        'order_id': [str(random.randint(1000, 9999))] * len(selected_products),
        'product_name': product_info['product_name'].tolist(),
        'product_id': product_info['product_id'].tolist(),
        'department': product_info['department'].tolist(),
        'cart_inclusion_order': list(range(1, len(selected_products) + 1)),
        'reordered': [random.choice([0, 1]) for _ in selected_products]
    }
    cart_df = pd.DataFrame(cart_data)
    prepared_df = data_preparation_orders(cart_df)

    # Display the dataframe below the cart
    #st.header("Cart Data")
    #st.dataframe(prepared_df)
else:
    st.write("No products in the cart yet. Please add products from the sidebar.")


# Placeholder for recommendations
#st.header("Recommended Products")
#recommendations_placeholder = st.empty()

# Button to generate recommendations
if st.button("Go to payment"):
    recommendations = get_top_n_recommendations_pinecone_batch('supercenter-recommender-system',
                                                               prepared_df['text_feature'].tolist())

    recomendations_info = map_product_info_through_id(recommendations[0], products_df)
    st.header("You may be missing...")

    #recommendations_placeholder.write(f'{recommendations[0]}')
    for recommendation in recomendations_info['product_name']:
        st.write(f"- {recommendation}")
#else:
 #   recommendations_placeholder.write("Add products to the cart and click 'Get Recommendations' to see suggestions.")

# Footer or additional information
st.sidebar.markdown("## About")
st.sidebar.info(
    "This is a demonstration of the product recommendation system proposal. Add products to your cart and get personalized recommendations.")