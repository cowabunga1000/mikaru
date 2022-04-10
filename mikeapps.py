import streamlit as st
import numpy as np
from keras.models import load_model


st.title("This is Michael's Final Project With Databangalore Indonesia")
st.subheader("this project is final project match 2")

st.text("project ini memberikan nilai uji untuk data test ASL dari kaggle")

model_inception = load_model("mike_sl.h5")