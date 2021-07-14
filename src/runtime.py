from Model import ModelWorker
from Pipeline import DataPipeline

"""
A file to run from terminal to test and evaluate a model
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow_datasets as tfds 
import cv2


if __name__ == '__main__':
    pipe = DataPipeline()
    model = ModelWorker(pipeline=pipe)
    model.demo()
    
    # images_dir = 'data/HAM10000_images/'
    # df = pd.read_csv('data/HAM10000_metadata')
    # classes = df['dx'].unique().tolist()
    # df = df.drop_duplicates(subset=['lesion_id'])

    # paths = df['image_id'].apply(lambda x: x + '.jpg')
    # paths = paths.tolist()
    # labels = df['dx'].apply(lambda x: classes.index(x))
    # #run model
    # x = cv2.imread(images_dir+paths[0],cv2.IMREAD_COLOR)
    # # cv2.imshow('image', x)
    # # cv2.waitKey(0)
    # x = cv2.resize(x,(32,32))
    # x = x/255.0
    # x = x.astype(np.float32)
    # # cv2.imshow('image', x)
    # # cv2.waitKey(0)
    # print(x.shape)