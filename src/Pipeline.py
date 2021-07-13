import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow_datasets as tfds 

import os
import cv2
from glob import glob

class Pipeline():
    def __init__(self,path,files):
        """
        das
        """
        self.path = path
        self.files = files

    def load_data(self):
        """
        das
        """
        images=[]
        for file in self.files:
            images.append(glob(os.path.join(self.path,'HAM10000_images/' + file)))
        return images

    def read_image(self,path):
        """
        das
        """
        x = cv2.imread(path,cv2.IMREAD_COLOR)
        x = cv2.resize(x,(256,256))
        x = x/255.0
        x = x.astype(np.float32)
        return x

    def preprocess(self,x,y):
        def f(x,y):
            x = x[0].decode()
            x = self.read_image(x)
            return x,y
        img,label = tf.numpy_function(f,[x,y],[tf.float32,tf.int32])
        img.set_shape([256,256,3])
        return img,label

    def tf_dataset(self,x,y,batch=8):
        ds = tf.data.Dataset.from_tensor_slices((x,y))
        ds = ds.shuffle(buffer_size=10)
        ds = ds.map(self.preprocess)
        ds = ds.batch(batch)
        ds = ds.prefetch(2)
        return ds
    def execute(self): #split this up
        df = pd.read_csv('data/HAM10000_metadata')
        df = df.drop_duplicates(subset=['lesion_id'])
        
        classes = df['dx'].unique().tolist()

        labels = df['dx'].apply(lambda x: classes.index(x))
        paths = df['image_id'].apply(lambda x: x + '.jpg')

        labels = labels.tolist()
        paths = paths.tolist()

        imgs = self.load_data('data/',paths)

        ds = self.tf_dataset(imgs,labels)
        return ds
    def train_val_test(self): # set this up
        # get the train data and split it
        # get the test data and format it
        # return 3 X,y pairs
        pass