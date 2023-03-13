import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import Preprocess

import matplotlib.pyplot as plt

import os
import shutil

import numpy as np

import cv2
from PIL import Image

# Load data

def predict_captcha(filepath):
    Preprocess.extract_digits(filepath,"temp_captcha_digit_folder")
    model = tf.keras.models.load_model("model/captcha-CNN-1")
    input_filepaths=[]
    for i in range(1,7):
        input_filepaths.append("temp_captcha_digit_folder/"+str(i)+".jpg")
    res=get_prediction(model,input_filepaths)
    shutil.rmtree("temp_captcha_digit_folder")
    return ''.join(map(str, res))
def get_prediction(model,filepath_list):

    img_size = (23, 15)  # fixed size for resizing the images
    X = []

    for filepath in filepath_list:
        img = load_img(filepath, target_size=img_size, color_mode="grayscale")
        img_arr = img_to_array(img)
        X.append(img_arr)

    X = np.array(X)

    # Preprocess the data
    X = X / 255.0  # normalize pixel values



    y_pred = model.predict(X)
    y_pred = np.argmax(y_pred, axis=1)

    return y_pred

# print(predict_captcha("captchas/13-307896.jpg"))      #example usage


