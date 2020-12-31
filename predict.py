import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import cv
from tensorflow.keras.preprocessing import image


def get_prediction(filename):
    img_width, img_height = 160, 160
    img = image.load_img(filename, target_size = (img_width, img_height))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis = 0)

    prediction = model.predict(img)
    prediction = tf.nn.sigmoid(prediction)
    prediction = tf.where(prediction < 0.5, 0, 1)

    return prediction

if __name__ == '__main__':    
    loaded_model = tf.keras.models.load_model('model')

    filename = 'train/mirai/unnamed.jpg'
    filename = 'train/tsubasa/016tsu0034_0.png'
    
    prediction = get_prediction(filename)
    print(prediction.numpy()[0,0])
