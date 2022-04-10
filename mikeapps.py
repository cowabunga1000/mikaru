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
        
    st.sidebar.markdown(
        " \
        Example link for <a href='https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcREzt3OL9DdfGODyBesGVVu8i7MNh0nINfjA6r1PDOj4g8xNnpM1rz3iNootFDzIU4ukZA&usqp=CAU' style='text-decoration: none;'>Rock</a>, \
                         <a href='https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ68ARUGkcMCdcIAWylFRNx8-rEO3TT4uULjXZLMEiN1Jh7SrmtWNrR0bLVC5DoqJs_AsM&usqp=CAU' style='text-decoration: none;'>Paper</a>, \
                         <a href='https://thumbs.dreamstime.com/b/young-male-scissors-gesture-left-hand-concept-rock-paper-game-isolated-white-background-118950800.jpg' style='text-decoration: none;'>Scissors</a>", \
        unsafe_allow_html=True
    )
    
    image_url = st.sidebar.text_input("Please use JPG or JPEG image for better prediction")
    

    if image_url == "":
        st.sidebar.image('https://media.giphy.com/media/QWvra259h4LCvdJnxP/giphy.gif', width=300)
        st.markdown("<h1 style='text-align: center;'>Rock ✊🏼 Paper ✋🏼 Scissors ✌🏼</h1>", unsafe_allow_html=True)
        st.markdown("""
                    """)
        st.image('Images/RPS.png', width=700)
        st.markdown("<h3 style='text-align: center;'>Project by <a href='https://www.linkedin.com/in/myarist/' style='text-decoration: none; color:white;'>Muhammad Yusuf Aristyanto</a></h3>", unsafe_allow_html=True)

    else:
        try:
            file = BytesIO(urlopen(image_url).read())
            img = file
            label, df_output, uploaded_image, s = predict_image(img)
            st.sidebar.image(uploaded_image, width = None)

            st.markdown("<h1 style='text-align: center;'>The Image is Detected as {}</h1>".format(label), unsafe_allow_html=True)
            st.markdown("""
                        """)
            
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(10,6))
            ax = sns.barplot(x = 'Product', y = 'Probability', data = df_output)
            plt.xlabel('')

            for i,p in enumerate(ax.patches):
                height = p.get_height()
                ax.text(p.get_x()+p.get_width()/2.,
                    height + 0.01, str(round(s[i]*100,2))+'%',
                    ha="center") 

            st.pyplot(fig)
        except:
            st.sidebar.image('Images/emot.gif')
            st.markdown("<h1 style='text-align: center; color:red;'>Oh, No! 😱</h1>", unsafe_allow_html=True)
            st.markdown("""
                        """)
            st.image('Images/RPS.png', width=700)
            st.markdown("<h2 style='text-align: center;'>Please Use Another Link Image 🙏🏻</h2>", unsafe_allow_html=True)
        
if __name__ == '__main__':
    main()