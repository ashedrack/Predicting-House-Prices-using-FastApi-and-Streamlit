import streamlit as st
import requests
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd


# Streamlit application
st.title('House Price Prediction')

# Load images
def load_image(image_path):
    img = Image.open(image_path)
    return img

# Sidebar
st.sidebar.title("House Price Prediction")
st.sidebar.image(load_image('most-beautiful-houses2.png'), caption='House Image', use_column_width=True)
st.sidebar.write("This application predicts house prices based on the provided features.")


# Sidebar for model selection
model_choice = st.sidebar.selectbox('Choose a model', ('Gaussian Process Regressor', 'Prophet'))

# Function to make predictions with FastAPI (GPR)
def predict_with_gpr(inputs):
    url = "http://127.0.0.1:8000/predict/gpr/"
    response = requests.post(url, json=inputs)
    if response.status_code == 200:
        return response.json()['predicted_price']
    else:
        st.error("Error: " + response.text)
        return None

# Function to make predictions with FastAPI (Prophet)
def predict_with_prophet(periods):
    url = "http://127.0.0.1:8000/predict/prophet/"
    response = requests.post(url, json={"periods": periods})
    if response.status_code == 200:
        return response.json()
    else:
        st.error("Error: " + response.text)
        return None

if model_choice == 'Gaussian Process Regressor':
    st.header('Predict with Gaussian Process Regressor')

    # Collect input features from the user
    bedrooms = st.number_input("Bedrooms", min_value=0, max_value=20, value=3)
    bathrooms = st.number_input("Bathrooms", min_value=0.0, max_value=10.0, step=0.1, value=2.0)
    sqft_living = st.number_input("Square Feet Living", min_value=0, max_value=10000, value=2000)
    sqft_lot = st.number_input("Square Feet Lot", min_value=0, max_value=100000, value=5000)
    floors = st.number_input("Floors", min_value=0.0, max_value=10.0, step=0.1, value=1.0)
    waterfront = st.selectbox("Waterfront", [0, 1])
    view = st.selectbox("View", [0, 1, 2, 3, 4])
    condition = st.selectbox("Condition", [1, 2, 3, 4, 5])
    grade = st.selectbox("Grade", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    sqft_above = st.number_input("Square Feet Above", min_value=0, max_value=10000, value=1500)
    sqft_basement = st.number_input("Square Feet Basement", min_value=0, max_value=10000, value=0)
    yr_built = st.number_input("Year Built", min_value=1900, max_value=2024, value=2000)
    yr_renovated = st.number_input("Year Renovated", min_value=1900, max_value=2024, value=1900)
    zipcode = st.number_input("Zipcode", min_value=10000, max_value=99999, value=98101)
    lat = st.number_input("Latitude", min_value=-90.0, max_value=90.0, step=0.0001, value=47.0)
    long = st.number_input("Longitude", min_value=-180.0, max_value=180.0, step=0.0001, value=-122.0)
    sqft_living15 = st.number_input("Square Feet Living 15", min_value=0, max_value=10000, value=2000)
    sqft_lot15 = st.number_input("Square Feet Lot 15", min_value=0, max_value=100000, value=5000)
    year = st.number_input("Year", min_value=1900, max_value=2024, value=2020)
    month = st.number_input("Month", min_value=1, max_value=12, value=8)
    day = st.number_input("Day", min_value=1, max_value=31, value=15)

    # Bundle the inputs into a dictionary
    inputs = {
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "sqft_living": sqft_living,
        "sqft_lot": sqft_lot,
        "floors": floors,
        "waterfront": waterfront,
        "view": view,
        "condition": condition,
        "grade": grade,
        "sqft_above": sqft_above,
        "sqft_basement": sqft_basement,
        "yr_built": yr_built,
        "yr_renovated": yr_renovated,
        "zipcode": zipcode,
        "lat": lat,
        "long": long,
        "sqft_living15": sqft_living15,
        "sqft_lot15": sqft_lot15,
        "year": year,
        "month": month,
        "day": day
    }

    # Make the prediction
    if st.button('Predict Price'):
        prediction = predict_with_gpr(inputs)
        if prediction is not None:
            st.write(f'Predicted House Price: ${prediction:,.2f}')

            # Create a plot for GPR
            fig, ax = plt.subplots()
            ax.bar(['Predicted Price'], [prediction], color='blue')
            ax.set_title('GPR Prediction')
            ax.set_ylabel('Price ($)')
            st.pyplot(fig)

elif model_choice == 'Prophet':
    st.header('Predict with Prophet')

    # Collect input period from the user
    periods = st.number_input('Number of Days into the Future', min_value=1, max_value=365, value=30)

    # Make the prediction
    if st.button('Predict Future Prices'):
        forecast = predict_with_prophet(periods)
        if forecast:
            df_forecast = pd.DataFrame(forecast)
            st.write('Future House Prices:')
            st.dataframe(df_forecast)

            # Plot the forecast using Prophet's built-in plotting
            fig, ax = plt.subplots()
            ax.plot(df_forecast['ds'], df_forecast['yhat'], label='Predicted Price', color='blue')
            ax.fill_between(df_forecast['ds'], df_forecast['yhat_lower'], df_forecast['yhat_upper'], color='blue', alpha=0.2)
            ax.set_title('Prophet Forecast')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price ($)')
            st.pyplot(fig)
