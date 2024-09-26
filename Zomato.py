import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import streamlit as st

# photo
st.image(r"C:\Users\test\Zomato.jpg", use_column_width=True)

# Load the saved RandomForest model
model = joblib.load('RandomForest.pkl')

# Function to take user inputs from the Streamlit interface
def get_user_input():
    online_order = st.selectbox("Is the order online?", ("yes", "no"))
    location = st.text_input("Location")
    book_table = st.selectbox("Do you want to book a table?", ("yes", "no"))
    rest_type = st.text_input("Restaurant type")
    dish_liked = st.text_input("Dishes liked")
    cuisines = st.text_input("Cuisines")
    restaurant_category = st.text_input("Restaurant category")
    restaurant_region = st.text_input("Restaurant region")
    ave_cost_for_2 = st.number_input("Average cost for 2", min_value=0.0, step=0.01)

    # Create a DataFrame with the inputs
    data = {
        'online_order': [online_order],
        'location': [location],
        'book_table': [book_table],
        'rest_type': [rest_type],
        'dish_liked': [dish_liked],
        'cuisines': [cuisines],
        'restaurant_category': [restaurant_category],
        'restaurant_region': [restaurant_region],
        'ave_cost_for_2': [ave_cost_for_2]
    }
    
    return pd.DataFrame(data)

# Streamlit app layout
st.title("Zomato Restaurant Review Prediction App")

# Get user input
user_input_df = get_user_input()

# When the user clicks the button, make predictions
if st.button("Predict"):
    # Predict using the loaded model
    y_pred_new = model.predict(user_input_df)
    
    # Show the prediction result
    st.write("Prediction:", y_pred_new)


