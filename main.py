import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


model = tf.keras.models.load_model(r"models/1")

class_name = [
 'Tomato_Bacterial_spot',
 'Tomato_Early_blight',
 'Tomato_Late_blight',
 'Tomato_Leaf_Mold',
 'Tomato_Septoria_leaf_spot',
 'Tomato_Spider_mites_Two_spotted_spider_mite',
 'Tomato__Target_Spot',
 'Tomato__Tomato_YellowLeaf__Curl_Virus',
 'Tomato__Tomato_mosaic_virus',
 'Tomato_healthy']

st.title("Tomato Disease Classifier")

file =  st.file_uploader("Upload Image",type=['jpg',"png"])

if file is not None:
    im= Image.open(file)
    st.image(im, caption='File Uploaded Successfully')


if file is not None and st.button("Predict The Disease"):
    im = np.array(im)
    im = np.expand_dims(im,0)
    prediction =  model.predict(im)

    ind = np.argmax(prediction)
    pred_disease = class_name[ind]
    
    st.write(pred_disease)






