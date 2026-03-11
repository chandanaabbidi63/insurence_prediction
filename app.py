import streamlit as st



st.title("Insurance Premium Prediction")
st.write("This is a simple web application to predict the insurance premium based on the input features.")
Age=st.number_input("Enter Age",min_value=0,max_value=100)
Annual_Income_LPA=st.number_input("Enter Annual Income in LPA",min_value=0.0)
Policy_Term_Years=st.number_input("Enter Policy Term in Years",min_value=0)
Sum_Assured_Lakhs=st.number_input("Enter Sum Assured in Lakhs",min_value=0.0)

if st.button("Predict"):
    try:
        from src.prediction import insurance

        ins = insurance()
        result = ins.prediction1(Age,Annual_Income_LPA,Policy_Term_Years,Sum_Assured_Lakhs)
        st.success(f"The predicted Annual Premium in Thousands is: {result}")
        st.balloons()
    except ModuleNotFoundError as e:
        st.error(
            f"Missing Python dependency: {e}. Make sure deployment installs requirements.txt."
        )
    except Exception as e:
        st.error(f"Prediction failed: {e}")