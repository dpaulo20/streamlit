import streamlit as st

import numpy as np
import pandas as pd
import os
#import albumentations as albu
import cv2
import matplotlib.pyplot as plt
import segmentation_models as sm
from clouds_utilities_functions import np_resize, build_masks
from keras.models import load_model
from PIL import Image, ImageOps

HEIGHT = 320
WIDTH = 480

CHANNELS = 3 
NB_CLASSES = 4

st.title("Image Classification with Google's Teachable Machine")

st.header("Brain Tumor MRI Classification Example")

st.text("Upload a brain MRI Image for image classification as tumor or no-tumor")

BACKBONE = 'inceptionv3'

model = sm.FPN(BACKBONE, 
                classes=NB_CLASSES,
                input_shape=(HEIGHT, WIDTH, CHANNELS),
                encoder_weights='imagenet',
                activation='sigmoid',
                encoder_freeze=False)

model_path = st.file_uploader("Choose a h5 file", type="h5")



if model_path is not None:
    
    model.load_weights(model_path)
    

image_path = st.file_uploader("Choose a image", type="jpeg")

if image_path is not None:
     img = Image.open(image_path)
     data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
     image = img
     #image sizing
     size = (224, 224)
     image = ImageOps.fit(image, size, Image.ANTIALIAS)
     image_array = np.asarray(image)
     st.text(image_array.shape)  
        
     #batch_pred_masks = model.predict_generator(check_generator, 
         #                                   workers=1,
          #                                  verbose=1)



#url = 'https://github.com/jithincheriyan/Web_App/blob/mast
#/Transformer_BERT_Model.h5'
#filename = url.split('/')[-1]
#trained_model.load_weights(urllib.request.urlretrieve(url, filename))
