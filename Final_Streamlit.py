import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import joblib

#  Loading our final trained knn model
model = open("Knn_Classifier.pkl", "rb")
knn_clf = joblib.load(model)

st.title("Iris Flower species Classifier App")

# Loading Images
setosa = Image.open("images/Setosa.jpg")
versicolor = Image.open("images/versicolor.jpeg")
virginica = Image.open('images/virginica.jpg')

# Step 2
st.sidebar.title("Features")

# Initializing features of iris flower
parameter_list = [
    'Sepal Length (cm)', 'Sepal Width (cm)', 'Petal Length (cm)', 'Petal Width (cm)']
parameter_input_values = []
parameter_default_values = ['5.2', '3.2', '4.2', '1.2']

values = []

# Display
for parameter, parameter_df in zip(parameter_list, parameter_default_values):

    values = st.sidebar.slider(label=parameter, key=parameter, value=float(
        parameter_df), min_value=0.0, max_value=8.0, step=0.1)
    parameter_input_values.append(values)

input_variables = pd.DataFrame(
    [parameter_input_values], columns=parameter_list, dtype=float)
st.write('\n\n')

if st.button("Click here to Classify"):
    prediction = knn_clf.predict(input_variables)

    if prediction == 0:
        st.image(setosa, caption="Setosa", width=400)
        st.subheader("Setosa")
    elif prediction == 1:
        st.image(versicolor, caption="Versicolor", width=400)
        st.subheader("Versicolor")
    else:
        st.image(virginica, caption="Virginica", width=400)
        st.subheader("Virginica")
