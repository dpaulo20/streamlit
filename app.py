import streamlit as st

import numpy as np
import pandas as pd
import os
import albumentations as albu
import cv2
import matplotlib.pyplot as plt
import segmentation_models as sm
from clouds_utilities_functions import np_resize, build_masks

HEIGHT = 320
WIDTH = 480

CHANNELS = 3 
NB_CLASSES = 4

st.title("Image Classification with Google's Teachable Machine")

st.header("Brain Tumor MRI Classification Example")

st.text("Upload a brain MRI Image for image classification as tumor or no-tumor")

BACKBONE = 'resnet50'

model = sm.Unet(BACKBONE, 
                classes=NB_CLASSES,
                input_shape=(HEIGHT, WIDTH, CHANNELS),
                encoder_weights='imagenet',
                activation='sigmoid',
                encoder_freeze=False)

#model.load_weights('./model.h5')
