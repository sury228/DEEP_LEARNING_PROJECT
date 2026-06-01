import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder,StandardScaler,OneHotEncoder
import pickle

##load the model
# model = tf.keras.models.load_model("churn_model.h5")
model = tf.keras.models.load_model("../churn_model.h5")

## load the label encoder
# with open("label_encoder.pkl", "rb") as file:
with open("../label_encoder.pkl", "rb") as file:    
    label_encoder = pickle.load(file)

## load the scaler
# with open("scaler.pkl", "rb") as file:
with open("../scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

##load the one hot encoder
# with open("onehot_encoder.pkl", "rb") as file:
with open("../onehot_encoder.pkl", "rb") as file:
    onehot_encoder = pickle.load(file)

## streamlit app
st.title("Customer Churn Prediction")
## getting the user input

geography = st.selectbox("Geography", onehot_encoder.categories_[0])
gender = st.selectbox("Gender", label_encoder.classes_)
age = st.slider("Age", 18, 92)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
estimated_salary = st.number_input("Estimated Salary")
tenure = st.slider("Tenure", 0, 10)
num_of_products = st.slider("Number of Products", 1, 4)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])


input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})


# one hot encode geography
onehot_encoded = onehot_encoder.transform([[geography]])
onehot_encoded_df = pd.DataFrame(onehot_encoded.toarray(), columns=onehot_encoder.get_feature_names_out(['Geography']))

# concatenate the one-hot encoded geography with the input data
input_data = pd.concat([input_data, onehot_encoded_df], axis=1)


##scaling the data
scaled_input = scaler.transform(input_data)

##predicting the output
prediction = model.predict(scaled_input)
prediction_probability = prediction[0][0]



if prediction_probability > 0.5:
    st.write("The customer is likely to churn with a probability of {:.2f}".format(prediction_probability))
else:
    st.write("The customer is not likely to churn with a probability of {:.2f}".format(1 - prediction_probability))
