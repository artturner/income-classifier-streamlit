import streamlit as st
import pickle
import pandas as pd
import numpy as np

@st.cache_resource
def load_model_and_preprocessor():
    with open('models/xgboost_final.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('models/preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
    
    return model, preprocessor

def main():
    st.set_page_config(
        page_title="Income Prediction App",
        page_icon="💰",
        layout="wide"
    )
    
    st.title("💰 Income Prediction App")
    st.write("Enter a person's attributes to predict their income level (<=50K or >50K)")
    
    # Load model and preprocessor
    try:
        model, preprocessor = load_model_and_preprocessor()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return
    
    # Create input form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Personal Information")
            age = st.number_input("Age", min_value=17, max_value=90, value=39)
            sex = st.selectbox("Sex", ["Male", "Female"])
            race = st.selectbox("Race", [
                "White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"
            ])
            
            st.subheader("Education")
            education = st.selectbox("Education Level", [
                "Bachelors", "Some-college", "11th", "HS-grad", "Prof-school",
                "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters",
                "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"
            ])
            education_num = st.number_input("Education Years", min_value=1, max_value=16, value=13)
            
        with col2:
            st.subheader("Work Information")
            workclass = st.selectbox("Work Class", [
                "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
                "Local-gov", "State-gov", "Without-pay", "Never-worked"
            ])
            
            occupation = st.selectbox("Occupation", [
                "Tech-support", "Craft-repair", "Other-service", "Sales",
                "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
                "Machine-op-inspct", "Adm-clerical", "Farming-fishing",
                "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"
            ])
            
            hours_per_week = st.number_input("Hours per Week", min_value=1, max_value=99, value=40)
            
            st.subheader("Family Information")
            marital_status = st.selectbox("Marital Status", [
                "Married-civ-spouse", "Divorced", "Never-married", "Separated",
                "Widowed", "Married-spouse-absent", "Married-AF-spouse"
            ])
            
            relationship = st.selectbox("Relationship", [
                "Wife", "Own-child", "Husband", "Not-in-family",
                "Other-relative", "Unmarried"
            ])
            
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("Financial Information")
            capital_gain = st.number_input("Capital Gain", min_value=0, max_value=99999, value=0)
            capital_loss = st.number_input("Capital Loss", min_value=0, max_value=4356, value=0)
            
        with col4:
            st.subheader("Location")
            native_country = st.selectbox("Native Country", [
                "United-States", "Cuba", "Jamaica", "India", "Mexico", "South",
                "Puerto-Rico", "Honduras", "England", "Canada", "Germany", "Iran",
                "Philippines", "Italy", "Poland", "Columbia", "Cambodia", "Thailand",
                "Ecuador", "Laos", "Taiwan", "Haiti", "Portugal", "Dominican-Republic",
                "El-Salvador", "France", "Guatemala", "China", "Japan", "Yugoslavia",
                "Peru", "Outlying-US(Guam-USVI-etc)", "Scotland", "Trinadad&Tobago",
                "Greece", "Nicaragua", "Vietnam", "Hong", "Ireland", "Hungary",
                "Holand-Netherlands"
            ])
            
            fnlwgt = st.number_input("Final Weight", min_value=12285, max_value=1484705, value=77516)
        
        submitted = st.form_submit_button("Predict Income")
        
        if submitted:
            # Create input dataframe
            input_data = pd.DataFrame({
                'age': [age],
                'workclass': [workclass],
                'fnlwgt': [fnlwgt],
                'education': [education],
                'education.num': [education_num],
                'marital.status': [marital_status],
                'occupation': [occupation],
                'relationship': [relationship],
                'race': [race],
                'sex': [sex],
                'capital.gain': [capital_gain],
                'capital.loss': [capital_loss],
                'hours.per.week': [hours_per_week],
                'native.country': [native_country]
            })
            
            try:
                # Preprocess the input data
                X_processed = preprocessor.transform(input_data)
                
                # Make prediction
                prediction = model.predict(X_processed)[0]
                prediction_proba = model.predict_proba(X_processed)[0]
                
                # Display results
                st.subheader("Prediction Results")
                
                if prediction == 1:
                    st.success("💰 Predicted Income: **> $50,000**")
                else:
                    st.info("💼 Predicted Income: **≤ $50,000**")
                
                # Show prediction confidence
                confidence = max(prediction_proba) * 100
                st.write(f"**Confidence:** {confidence:.1f}%")
                
                # Show probability breakdown
                st.subheader("Probability Breakdown")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("≤ $50,000", f"{prediction_proba[0]:.1%}")
                
                with col2:
                    st.metric("> $50,000", f"{prediction_proba[1]:.1%}")
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")

if __name__ == "__main__":
    main()