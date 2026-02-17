# app.py

import streamlit as st
import numpy as np
import pickle

# Load saved model and encoder
model = pickle.load(open("fish_model.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

st.title("🐟 Fish Species Classification ")

st.write("Enter Fish Measurements:")

# User inputs
weight = st.number_input("Weight", min_value=0.0)
length1 = st.number_input("Length1", min_value=0.0)
length2 = st.number_input("Length2", min_value=0.0)
length3 = st.number_input("Length3", min_value=0.0)
height = st.number_input("Height", min_value=0.0)
width = st.number_input("Width", min_value=0.0)

# Prediction button
if st.button("Predict Species"):
    input_data = np.array([[weight, length1, length2, length3, height, width]])
    prediction = model.predict(input_data)
    species = le.inverse_transform(prediction)
    
    st.success(f"Predicted Fish Species: {species[0]}")
