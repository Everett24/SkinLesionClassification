import numpy as np
import pandas as pd
import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# import tensorflow_datasets as tfds 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
# import os
# import cv2
# from glob import glob
# from keras.utils.np_utils import to_categorical
# from tensorflow.python.ops.gen_array_ops import split

# init pipeline with train and test dirs
#   split train into validate
# load all the images for train,val,test
# return 3 dataset objects named train,test,val
# 
# create a way to easily get sub sets of the data for HP tuning

#create a bashh script to take a selected folder in downloads and extract it to the proper data format in the data folder of the project dir

#load df and prep
#load labels and prep
#create image generator
#init take paramters for batch and for count and for 
class BinaryDataPipeline():
    """
    A class used for creating pipelines loading image data for classification
    """
    def __init__(self): #add train,test directories
        """
        Initialize the pipeline
        """

    def load_df(self,path): 
        """
        Get the data from the directories; filterable
        """
        df = pd.read_csv(path)
        print(df.shape[0])
        
        df = df.drop_duplicates(subset=['patient_id'])
        df['image_name'] = df['image_name'].apply(lambda x: x + '.jpg')
        
        #find a better place to do this
        self.classes = df['benign_malignant'].unique().tolist()
        self.len = len(df['benign_malignant'].values)
        self.weights = (df.groupby('target').size()/df.shape[0]).to_dict()
        temp = self.weights[0]
        self.weights[0] = self.weights[1]/2
        self.weights[1] = temp
        print(type(self.weights))
        print(self.len)
        return df
        
    def get_all_data_gen(self):
        train_full = self.load_df('./data/train.csv')
        train_generator = self.get_img_gen(train_full,'image_name','benign_malignant','data/jpeg/train/' )
        return train_generator

    def execute(self):
        """
        Load data and Return a Dataset
        log_image=True : print an image before dataset and test loop
        """
        train_full = self.load_df('./data/train.csv')
        train_split,  test =  train_test_split(train_full,shuffle=False,test_size=.2)
        train,val = train_test_split(train_split,shuffle=False,test_size=.2)
        
        train_generator = self.get_img_gen(train,'image_name','benign_malignant','data/jpeg/train/' )
        val_generator = self.get_img_gen(val,'image_name','benign_malignant','data/jpeg/train/')
        test_generator = self.get_img_gen(test,'image_name','benign_malignant','data/jpeg/train/')

        return train_generator,val_generator,test_generator 
    
    def get_img_gen(self,df, x,y,dir,sub=None):
        test_datagen = ImageDataGenerator( 
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest')
        test_generator = test_datagen.flow_from_dataframe(
            dataframe=df,
            directory=dir,
            x_col =x,
            y_col =y,
            color_mode="grayscale",
            target_size=(32, 32),
            batch_size=10,
            class_mode='binary',
            validate_filenames=False,
            shuffle=True,
            subset=sub)
        return test_generator