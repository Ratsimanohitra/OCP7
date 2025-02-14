import streamlit as st
import pandas as pd
import requests

import plotly.graph_objects as go

# Function to load data from Excel file
@st.cache
def load_data(file_path):
    return pd.read_parquet(file_path)

# Function to send customer data to API and get prediction
def get_prediction(customer_data):
    api_url = "http://127.0.0.1:5000/predict"  # Replace with your API URL
    data_to_predict = {'data': customer_data.drop("SK_ID_CURR", axis=1).values.tolist()} #losrsque les modèles sans l'ID seront entrainé cela deviendra customer_data.drop("SK_ID_CURR", axis=1).values.tolist()
    response = requests.post(api_url, json=data_to_predict)
    return response.json()

# Load data
file_path = "./data/data.parquet"  # Replace with your Excel file path
data = load_data(file_path)

# Streamlit UI
st.title("Customer Prediction Dashboard")

# Select customer by ID
customer_id = st.selectbox("Select Customer ID", data["SK_ID_CURR"].unique())
customer_data = data[data["SK_ID_CURR"] == customer_id]

# Display customer information
st.write(f"You have selected the following customer:{customer_id}")
#st.write(customer_data)

# Get prediction
if st.button("Get Prediction"):
    prediction = get_prediction(customer_data)
    probability = prediction["prediction"][0]

    # Display gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability,
        title={'text': "Prediction Probability"},
        gauge={'axis': {'range': [0, 1]}}
    ))
    st.plotly_chart(fig)