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
import requests


HEIGHT = 320
WIDTH = 480
CHANNELS = 3 
NB_CLASSES = 4

######fonction de visualisation#########

def visualize_image_mask_prediction(image,mask_prediction):
    """ Fonction pour visualiser l'image original, le mask original et le mask predit"""

    class_dict = {0: 'Fish', 1: 'Flower', 2: 'Gravel', 3: 'Sugar'}
    
    st.image(image, caption='Uploaded cloud image.', use_column_width=True)
    cols = st.beta_columns(4) 
    for i in range(4):
        title='class  '+class_dict[i]
        cols[i].image(mask_prediction[0,:, :,i],caption=title,width=100)


###########################################

######### Fonctions Downloading fichier h5 ######

def download_file_from_google_drive(id, destination):
    URL = "https://drive.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

                
###########################################`

### création du modéle FPN-resnet50 +Download des poids

BACKBONE = 'resnet50'

model_FPN = sm.FPN(BACKBONE, 
                classes=NB_CLASSES,
                input_shape=(HEIGHT, WIDTH, CHANNELS),
                encoder_weights='imagenet',
                activation='sigmoid',
                encoder_freeze=False)



file_id = '17Th3xBfd0Qz3fKHl5vOesLANFOYfsU2s' ## Id du fichier sur le drive google
destination = 'FPN-resnet50.h5'
download_file_from_google_drive(file_id, destination)
model_FPN.load_weights('FPN-resnet50.h5')

###########################################

### création du modéle UNET-resnet50 +Download des poids

BACKBONE = 'resnet50'

model_UNET = sm.Unet(BACKBONE, 
                classes=NB_CLASSES,
                input_shape=(HEIGHT, WIDTH, CHANNELS),
                encoder_weights='imagenet',
                activation='sigmoid',
                encoder_freeze=False)



file_id = '10PVYP69m-vgx0gHhZ2UadovP5dTup5TS' ## Id du fichier sur le drive google
destination = 'UNET-resnet50.h5'
download_file_from_google_drive(file_id, destination)
model_UNET.load_weights('UNET-resnet50.h5')

###########################################




#########Streamlit section###############

st.title("Cloud classification project")

st.header("cloud Segmentation Example ")

st.text("Upload a image of cloud")

### Downloading du image, détection et affichage des prédictions

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
     batch_pred_masks_FPN = model_FPN.predict(data)
     st.text("Prediction FPN - resnet50")
     visualize_image_mask_prediction(image,batch_pred_masks_FPN)
     st.text("Prediction UNET - resnet50")
     visualize_image_mask_prediction(image,batch_pred_masks_UNET)




