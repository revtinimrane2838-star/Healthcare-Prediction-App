import streamlit as st
import pandas as pd
import joblib

model = joblib.load ('Risk_model1.ipynb')

st.title ("Health Risk Stratification App")

Age =st.number_input ("Age", min_value=0)
NumberOfStays = st.number_input("NumberOfStays", min_value=0)
TreatmentCost = st.number_input ("TreatmentCost",min_value=0.0)
AbnormalLabCount= st.number_input ("AbnormalLabCount", min_value=0)

if st.button ("Predict"):
    input_data = pd.DataFrame ([[Age,NumberOfStays,TreatmentCost,AbnormalLabCount]], columns = ['Age','NumberOfStays','TreatmentCost','AbnormalLabCount'])
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.write (f"Risk Prediction :{'High Risk'if prediction== 1 else 'Low Risk'}")
    st.write (f"Risk Probability :{round(probability,2)}")

