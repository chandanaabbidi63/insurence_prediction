import streamlit as st
from src.prediction import insurance


st.title("Insurance Premium Prediction")
st.write("This is a simple web application to predict the insurance premium based on the input features.")
Age=st.number_input("Enter Age",min_value=0,max_value=100)
Annual_Income_LPA=st.number_input("Enter Annual Income in LPA",min_value=0.0)
Policy_Term_Years=st.number_input("Enter Policy Term in Years",min_value=0)
Sum_Assured_Lakhs=st.number_input("Enter Sum Assured in Lakhs",min_value=0.0)

if st.button("Predict"):
    ins=insurance()
    result=ins.prediction1(Age,Annual_Income_LPA,Policy_Term_Years,Sum_Assured_Lakhs)
    st.success(f"The predicted Annual Premium in Thousands is: {result}")
    st.balloons()