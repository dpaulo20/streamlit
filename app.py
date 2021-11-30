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
import urllib3
import wget
import gdown


HEIGHT = 320
WIDTH = 480

CHANNELS = 3 
NB_CLASSES = 4

######fonction

def visualize_image_mask_prediction(image,mask_prediction):
    """ Fonction pour visualiser l'image original, le mask original et le mask predit"""

    class_dict = {0: 'Fish', 1: 'Flower', 2: 'Gravel', 3: 'Sugar'}
    
    st.image(image, caption='Uploaded cloud image.', use_column_width=True)
    cols = st.beta_columns(4) 
    for i in range(4):
        title='class  '+class_dict[i]
        cols[i].image(mask_prediction[0,:, :,i],caption=title,width=100)


############################




st.title("Cloud classification project")

st.header("cloud Segmentation Example with FPN-InceptionV3 model ")

st.text("Upload a image of cloud")

BACKBONE = 'resnet50'

model = sm.FPN(BACKBONE, 
                classes=NB_CLASSES,
                input_shape=(HEIGHT, WIDTH, CHANNELS),
                encoder_weights='imagenet',
                activation='sigmoid',
                encoder_freeze=False)

#Downloading h5
url = 'https://drive.google.com/u/0/open?id=17Th3xBfd0Qz3fKHl5vOesLANFOYfsU2s'

path2 = 'FPNresnet50.h5'

if not os.path.exists(path2):
        encoder_url = 'wget -O FPNresnet50.h5 https://drive.google.com/u/0/open?id=17Th3xBfd0Qz3fKHl5vOesLANFOYfsU2s'
        with st.spinner('Downloading model weights for resnet50'):
            os.system(encoder_url)
else:

    print("Model 2 is here.")

model.load_weights(path2)
####

uploaded_file = st.file_uploader("Choose a H5 ...", type="h5")
if uploaded_file is not None:
   model.load_weights(uploaded_file)
else:
   st.text("Non")
    

image_path = st.file_uploader("Choose a image", type="jpg")

if image_path is not None:
     img = Image.open(image_path)
     data = np.ndarray(shape=(1, HEIGHT,WIDTH, 3), dtype=np.float32)
     image = img
     #image sizing
     size = (WIDTH,HEIGHT)
     image = ImageOps.fit(image, size)
     image_array = np.asarray(image)/ 255.
     data[0] = image_array
     st.text(data.shape) 
     batch_pred_masks = model.predict(data)
     visualize_image_mask_prediction(image,batch_pred_masks)




