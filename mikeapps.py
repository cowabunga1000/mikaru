import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model


st.title("This is Michael's Final Project With Databangalore Indonesia")
st.subheader("this project is final project match 2")

st.text("project ini memberikan nilai uji untuk data test ASL dari kaggle")

model_inception = load_model("mike_sl.h5")

def predict_image(image_upload, model = model_inception):
        im = Image.open(image_upload)
        resized_im = im.resize((150, 150))
        im_array = np.asarray(resized_im)
        im_array = im_array*(1/225)
        im_input = tf.reshape(im_array, shape = [1, 150, 150, 3])

        predict_array = model.predict(im_input)[0]
        paper_proba = predict_array[0]
        rock_proba = predict_array[1]
        scissors_proba = predict_array[2]

        s = [paper_proba, rock_proba, scissors_proba]

        import pandas as pd
        df = pd.DataFrame(predict_array)
        df = df.rename({0:'Probability'}, axis = 'columns')
        prod = ['Paper', 'Rock', 'Scissor']
        df['Product'] = prod
        df = df[['Product', 'Probability']]

        predict_label = np.argmax(model.predict(im_input))

        if predict_label == 0:
            predict_product = 'Paper ✋🏼'
        elif predict_label == 1:
            predict_product = 'Rock ✊🏼'
        else:
            predict_product = 'Scissor ✌🏼'

        return predict_product, df, im, s