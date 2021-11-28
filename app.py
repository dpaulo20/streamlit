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

##fonction

def visualize_image_mask_prediction(image):
    """ Fonction pour visualiser l'image original, le mask original et le mask predit"""
    fontsize = 14
    class_dict = {0: 'Fish', 1: 'Flower', 2: 'Gravel', 3: 'Sugar'}
    f, ax = plt.subplots(2, 5, figsize=(24,8))
    
    ax[1, 0].imshow(image)
    ax[1, 0].set_title('Original image', fontsize=fontsize)

#    for i in range(4):
#        ax[1, i + 1].imshow(mask_prediction[:, :, i],vmin = 0, vmax = 1)
#        ax[1, i + 1].set_title(f'Prediction {class_dict[i]}', fontsize=fontsize)

############################




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
    

image_path = st.file_uploader("Choose a image", type="jpg")

if image_path is not None:
     img = Image.open(image_path)
     data = np.ndarray(shape=(1, WIDTH, HEIGHT, 3), dtype=np.float32)
     image = img
     #image sizing
     size = (HEIGHT,WIDTH)
     image = ImageOps.fit(image, size, Image.ANTIALIAS)
     image_array = np.asarray(image)
     data[0] = image_array
     st.text(data.shape) 
     #batch_pred_masks = model.predict(data, workers=1,verbose=1)
     visualize_image_mask_prediction(image)



#url = 'https://github.com/jithincheriyan/Web_App/blob/mast
#/Transformer_BERT_Model.h5'
#filename = url.split('/')[-1]
#trained_model.load_weights(urllib.request.urlretrieve(url, filename))
