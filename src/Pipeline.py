import numpy as np
import pandas as pd
import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# import tensorflow_datasets as tfds 

import os
import cv2
from glob import glob

# init pipeline with train and test dirs
#   split train into validate
# load all the images for train,val,test
# return 3 dataset objects named train,test,val
# 
# create a way to easily get sub sets of the data for HP tuning

#create a bashh script to take a selected folder in downloads and extract it to the proper data format in the data folder of the project dir


class Pipeline():
    """
    A class used for creating pipelines loading image data for classification
    """
    def __init__(self,path,files): #add train,test directories
        """
        Initialize the pipeline
        """
        self.path = path
        self.files = files

    def load_data(self): 
        """
        Get the data from the directories
        """
        self.df = pd.read_csv('data/HAM10000_metadata')
        self.df = self.df.drop_duplicates(subset=['lesion_id'])

        images=[]
        for file in self.files:
            images.append(glob(os.path.join(self.path,'HAM10000_images/' + file)))
        return images

    def read_image(self,path):
        """
        Turn a path into an image
        """
        x = cv2.imread(path,cv2.IMREAD_COLOR)
        x = cv2.resize(x,(256,256))
        x = x/255.0
        x = x.astype(np.float32)
        return x

    def preprocess(self,x,y):
        """
        Process x data from file paths to images
        """
        def f(x,y):
            """
            das
            """
            x = x[0].decode()
            x = self.read_image(x)
            return x,y
        img,label = tf.numpy_function(f,[x,y],[tf.float32,tf.int32])
        img.set_shape([256,256,3])
        return img,label

    def tf_dataset(self,x,y,batch=8):
        """
        Make a TF Dataset from the provided X,y data

        Arguments:
        x: numpy array of images
        y: numpy array of class ids
        
        Returns:
        TF Dataset
        """
        ds = tf.data.Dataset.from_tensor_slices((x,y))
        ds = ds.shuffle(buffer_size=10)
        ds = ds.map(self.preprocess)
        ds = ds.batch(batch)
        ds = ds.prefetch(2)
        return ds
    def execute(self):
        """
        Load data and Return a Dataset
        """
        classes = self.df['dx'].unique().tolist()

        labels = self.df['dx'].apply(lambda x: classes.index(x)).tolist()
        paths = self.df['image_id'].apply(lambda x: x + '.jpg').tolist()

        imgs = self.load_data('data/',paths)

        ds = self.tf_dataset(imgs,labels)
        return ds
    def train_val_test(self): # set this up
        """
        Return a a dataset for train,validate and test
        """
        
        # get the train data and split it
        # get the test data and format it
        # return 3 X,y pairs
        pass