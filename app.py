import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler, quantile_transform
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('ann_model.h5')

# Load the scaler and label encoders
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f) 
with open('label_encoder_Gender.pkl', 'rb') as f:
    label_encoder_Gender = pickle.load(f)
with open('onehot_encoder_geo.pkl', 'rb') as f:
    onehot_encoder_Geography = pickle.load(f)

# Streamlit app
st.title("Customer Churn Prediction")

# Input fields
st.sidebar.header("Input Features")
CreditScore = st.sidebar.number_input("Credit Score", min_value=000, max_value=850, value=000)
Geography = st.sidebar.selectbox("Geography", options=["France", "Spain", "Germany"])       
Gender = st.sidebar.selectbox("Gender", label_encoder_Gender.classes_)
Age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)
Tenure = st.sidebar.number_input("Tenure", min_value=0, max_value=10, value=3)
Balance = st.sidebar.number_input("Balance", min_value=0, value=100)
NumOfProducts = st.sidebar.number_input("Number of Products", min_value=1, max_value=4, value=1)
HasCrCard = st.sidebar.selectbox("Has Credit Card", options=[0, 1], index=1)
IsActiveMember = st.sidebar.selectbox("Is Active Member", options=[0, 1], index=1)
EstimatedSalary = st.sidebar.number_input("Estimated Salary", min_value=00, value=50000)

input_data = pd.DataFrame({
    'CreditScore': [CreditScore],
    'Geography': [Geography],
    'Gender':[label_encoder_Gender.transform([Gender])[0]],
    'Age': [Age],
    'Tenure': [Tenure],
    'Balance': [Balance],       
    'NumOfProducts': [NumOfProducts],
    'HasCrCard': [HasCrCard],
    'IsActiveMember': [IsActiveMember],
    'EstimatedSalary': [EstimatedSalary]
})

#one-hot encode Geography
geo_encoded = onehot_encoder_Geography.transform(input_data[['Geography']])
geo_encoded_df = pd.DataFrame(geo_encoded.toarray(), columns=onehot_encoder_Geography.get_feature_names_out(['Geography']))
#combine all features into a single dataframe
input_df=pd.concat([input_data, geo_encoded_df], axis=1)
input_df.drop(['Geography'], axis=1, inplace=True)

#Scale the input data
input_data_scaled = scaler.transform(input_df)               

# Predict the output
prediction = model.predict(input_data_scaled)
prediction_probability = prediction[0][0]
if prediction_probability > 0.5:
    st.error(f"The customer is likely to leave the bank with a probability of {prediction_probability:.2f}")    
else:
    st.success(f"The customer is likely to stay with the bank with a probability of {1 - prediction_probability:.2f}")