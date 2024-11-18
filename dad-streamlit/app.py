import streamlit as st
import numpy as np
import pickle as pkl




st.title("This is a website about flowers")
st.header("Flower Predictor")
st.image("flower.jpg", caption="FLower")

st.header("Enter the info on your iris below")

# sepal length
sepal_length = st.number_input("Sepal length (cm)")
# sepal width
sepal_width = st.number_input("Sepal width (cm)")
# petal length
petal_length = st.number_input("Petal length (cm)")
# petal width
petal_width = st.number_input("Petal width (cm)")

new_flower = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

with open('model1.pkl', 'rb') as f:
    model = pkl.load(f)

prediction = model.predict(new_flower)
st.write(prediction)