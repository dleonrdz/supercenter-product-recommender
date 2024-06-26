import streamlit as st
import os


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
image_path = os.path.join(PROJECT_ROOT,"streamlit_app/images/logo.jpg")
products =

# Title of the app
st.title("Welcome to your SuperCenter Online!")

# Sidebar for selecting products
st.sidebar.title("Select Products to Add to Cart")
products = ["Product 1", "Product 2", "Product 3", "Product 4", "Product 5"]  # Placeholder for actual product list
selected_products = st.sidebar.multiselect("Add Products:", products)

# Display selected products in the main area with enumeration
st.header("Cart")
if selected_products:
    st.write("Products in your cart:")
    for i, product in enumerate(selected_products, 1):
        st.write(f"{i}. {product}")
else:
    st.write("No products in the cart yet. Please add products from the sidebar.")


# Placeholder for recommendations
st.header("Recommended Products")
recommendations_placeholder = st.empty()

# Button to generate recommendations
if st.button("Get Recommendations"):
    # Placeholder logic for generating recommendations
    # Replace this with your actual recommendation logic
    recommendations = ["Recommended Product A", "Recommended Product B", "Recommended Product C"]

    recommendations_placeholder.write("Recommendations based on your cart:")
    for recommendation in recommendations:
        recommendations_placeholder.write(f"- {recommendation}")
else:
    recommendations_placeholder.write("Add products to the cart and click 'Get Recommendations' to see suggestions.")

# Footer or additional information
st.sidebar.markdown("## About")
st.sidebar.info(
    "This is a demonstration of a product recommendation system using Streamlit. Add products to your cart and get personalized recommendations.")