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
import requests
from keras import backend as K
import keras
import tensorflow as tf
from keras.applications import VGG16
from keras.models import Model, Sequential, load_model
from keras.layers import LSTM, Dense, RepeatVector,TimeDistributed, Input, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Dropout

K.clear_session()
#graph1 = tf.get_default_graph()
#graph2 = tf.get_default_graph()
#graph3 = tf.get_default_graph()


######params généraux #########

HEIGHT = 320
WIDTH = 480
CHANNELS = 3 
NB_CLASSES = 4

######fonction de visualisation des détéctions #########

def visualize_image_mask_prediction(image,mask_prediction):
    """ Fonction pour visualiser l'image original, le mask original et le mask predit"""

    class_dict = {0: 'Fish', 1: 'Flower', 2: 'Gravel', 3: 'Sugar'}
    cols = st.columns(4) 
    for i in range(4):
        title='class  '+class_dict[i]
        cols[i].image(mask_prediction[0,:, :,i],caption=title,width=100)

def probability_VGG16_display(prediction):
    """ Fonction pour visualiser l'image original, le mask original et le mask predit"""

    class_dict = {0: 'Fish', 1: 'Flower', 2: 'Gravel', 3: 'Sugar'}
    cols = st.columns(4) 
    for i in range(4):
        title='class  '+class_dict[i]
        cols[i].text(prediction[i])


###########################################

######### Fonctions Downloading fichier h5 ######
@st.cache
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



def build_FPN_resnet50():
    global model_FPN
    BACKBONE = 'resnet50'

    ## modifie les noms des layers pour quelles soient uniques
    model_FPN = sm.FPN(BACKBONE, 
                    classes=NB_CLASSES,
                    input_shape=(HEIGHT, WIDTH, CHANNELS),
                    encoder_weights='imagenet',
                    activation='sigmoid',
                    encoder_freeze=False)

    i=1
    for layer in model_FPN.layers:
          layer.name = layer.name + str("_FPN")+str(i)
          i=i+1

    ####

    file_id = '17Th3xBfd0Qz3fKHl5vOesLANFOYfsU2s' ## Id du fichier sur le drive 
    destination = 'FPN-resnet50.h5'
    #download_file_from_google_drive(file_id, destination)
    #model.load_weights('FPN-resnet50.h5')


    try:
        download_file_from_google_drive(file_id, destination)
    except ValueError:
        st.error("erreur chargement H5")



    model_FPN.load_weights('FPN-resnet50.h5')

###########################################


def build_UNET_resnet50():
    ## création du modéle UNET-resnet50 +Download des poids
    global model_UNET
    BACKBONE = 'resnet50'


    model_UNET = sm.Unet(BACKBONE, 
                    classes=NB_CLASSES,
                    input_shape=(HEIGHT, WIDTH, CHANNELS),
                    encoder_weights='imagenet',
                    activation='sigmoid',
                    encoder_freeze=False)

    ## modifie les noms des layers pour quelles soient uniques
    i=1
    for layer in model_UNET.layers:

         layer.name = layer.name + str("_UNET")+str(i)
         i=i+1
    ###

    file_id = '10PVYP69m-vgx0gHhZ2UadovP5dTup5TS' ## Id du fichier sur le drive 
    destination = 'UNET-resnet50.h5'
    #download_file_from_google_drive(file_id, destination)
    #model.load_weights('UNET-resnet50.h5')

    try:
        download_file_from_google_drive(file_id, destination)
    except ValueError:
        st.error("erreur chargement H5")


    model_UNET.load_weights('UNET-resnet50.h5')


###########################################

def vgg16_classfication():
    global model_VGG16

    base_model = VGG16(include_top=False,
                                   weights="imagenet",
                                   input_shape=(224, 224, CHANNELS))

    for layer in base_model.layers:
        layer.trainable = False
    
    model_VGG16 = Sequential()
    model_VGG16.add(base_model)
    model_VGG16.add(Flatten())
    model_VGG16.add(Dense(units = NB_CLASSES, activation = "sigmoid"))

    file_id = '1ulaQnJy6iTSF8JZ6PhpQg5MoIH16fVj-' ## Id du fichier sur le drive 
    destination = 'VGG16.h5'
    

    try:
        download_file_from_google_drive(file_id, destination)
    except ValueError:
        st.error("erreur chargement H5")
    
    model_VGG16.load_weights('VGG16.h5')
    st.text("bebebbebebebbebebebbebe")
        
        


###########################################

@st.cache(ttl=3600, max_entries=10)
def load_output_image(img):
     img = Image.open(image_path)
     data_S = np.ndarray(shape=(1, HEIGHT,WIDTH, 3), dtype=np.float32)
     data_C = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
     
     image = img
     #image sizing
     size = (WIDTH,HEIGHT)
     image = ImageOps.fit(image, size)
     image_C = ImageOps.fit(image, (224, 224))
     image_array = np.asarray(image)/ 255.
     image_array_C = np.asarray(image_C)/ 255.
     data_S[0] = image_array
     data_C[0] = image_array_C
     return image,data_S,data_C
    

#########################################


build_FPN_resnet50()
build_UNET_resnet50()
vgg16_classfication()


#########Streamlit Section###############

st.title("Projet des régions nuageuses")

st.header("Exemple de modéles de segmentation ")

st.text("télécharger une image de nuage au format .jpg")

### Downloading du image, détection et affichage des prédictions

image_path = st.file_uploader("Choisir une image", type="jpg")

if image_path is not None:
     image,data_S,data_C=load_output_image(image_path)
        
     st.image(image, caption='Uploaded cloud image.', use_column_width=True)
  


     batch_pred_masks_UNET = model_UNET.predict(data_S)

        
     st.subheader("Prédiction modéle UNET - Resnet50")

     visualize_image_mask_prediction(image,batch_pred_masks_UNET)
        
     st.subheader("Prédiction modéle FPN - Resnet50")
    


     batch_pred_masks_FPN = model_FPN.predict(data_S)

        
     visualize_image_mask_prediction(None,batch_pred_masks_FPN)
    
     st.subheader("Classification avec VGG16 (probalilité)")


     prediction_class_VGG16 = model_VGG16.predict(data_C)

    
     st.text(prediction_class_VGG16)
            
     class_dict = {0: 'Fish', 1: 'Flower', 2: 'Gravel', 3: 'Sugar'}
     cols = st.columns(4) 
     for i in range(4):
         title='class  '+class_dict[i]
         cols[i].header(np.round(prediction_class_VGG16[0][i],2))

   #  st.text(prediction_class_VGG16)
        



