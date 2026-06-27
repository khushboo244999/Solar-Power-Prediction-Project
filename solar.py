#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import numpy as np
import pickle

st.title("☀️ Solar Power Generation Prediction")

model = pickle.load(open("rf_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

ambient = st.number_input("distance-to-solar-noon")
module = st.number_input("Temperature")
irradiation = st.number_input("wind-direction")
lag1 = st.number_input("Previous Power (Lag 1)")
lag2 = st.number_input("Previous Power (Lag 2)")
roll3 = st.number_input("Rolling Mean (3)")
roll6 = st.number_input("Rolling Mean (6)")

if st.button("Predict"):
    data = np.array([[ambient, module, irradiation, lag1, lag2, roll3, roll6]])
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)
    st.success(f"Predicted Power: {prediction[0]:.2f}")

