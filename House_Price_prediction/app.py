import streamlit as st
import pandas as pd
import pickle  # Uncomment this if you are loading a saved model

# Set the title and description of the app
st.title("House Price Prediction App")
st.write("Input the details below to predict the house price.")

# Sidebar: Collect user inputs for the prediction features
st.sidebar.header("Input Parameters")

def user_input_features():
    area = st.sidebar.slider('Area (sq ft)', 500, 5000, 2000)
    bedrooms = st.sidebar.slider('Bedrooms', 1, 5, 3)
    bathrooms = st.sidebar.slider('Bathrooms', 1, 4, 2)
    parking = st.sidebar.selectbox('Parking', ('yes', 'no'))
    airconditioning = st.sidebar.selectbox('Air Conditioning', ('yes', 'no'))
    
    # Convert categorical inputs to numeric values
    parking_numeric = 1 if parking == 'yes' else 0
    airconditioning_numeric = 1 if airconditioning == 'yes' else 0

    # Create a DataFrame with the user inputs
    data = {
        'area': [area],
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'parking': [parking_numeric],
        'airconditioning': [airconditioning_numeric]
    }
    features = pd.DataFrame(data)
    return features

# Retrieve the input data
input_df = user_input_features()

# Display the user input parameters
st.subheader("User Input Parameters")
st.write(input_df)

# Load your trained model (If you have one saved with pickle)
with open('housing_model.pkl', 'rb') as f:
    model = pickle.load(f)

# For demonstration purposes, let's create a dummy prediction
# Replace this dummy prediction with your actual model prediction code.
# For example: prediction = model.predict(input_df)
prediction = model  # Dummy calculation

# Show the prediction result
st.subheader("Predicted House Price")
st.write(f"${prediction:,.2f}")
