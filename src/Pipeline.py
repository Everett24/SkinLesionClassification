import numpy as np
import pandas as pd
import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# import tensorflow_datasets as tfds 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import cv2
from glob import glob
from keras.utils.np_utils import to_categorical
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
class DataPipeline():
    """
    A class used for creating pipelines loading image data for classification
    """
    def __init__(self): #add train,test directories
        """
        Initialize the pipeline
        """
        

    def load_data(self,root,count): 
        """
        Get the data from the directories; filterable
        """
        self.df = pd.read_csv('./data/HAM10000_metadata')
        self.df = self.df.drop_duplicates(subset=['lesion_id'])
        classes = self.df['dx'].unique().tolist()

        # this needs to be filtered first to account for imbalanced data before it can be taken forward [:count] will not suffice

        labels = self.df['dx'].apply(lambda x: classes.index(x)).tolist()[:count]
        paths = self.df['image_id'].apply(lambda x: x + '.jpg').tolist()[:count]
        print(labels)
        print('paths',len(paths))
        images=[]
        for file in paths:
            images.append(glob(os.path.join(root,'HAM10000_images/' + file)))
        return images, labels

    def read_image(self,path):
        """
        Turn a path into an image
        """
        x = cv2.imread(path,cv2.IMREAD_COLOR)
        x = cv2.resize(x,(32,32))
        x = x/255.0
        x = x.astype(np.float32)
        return x

    def preprocess(self,x,y):
        """
        Process x data from file paths to images, resize those images
        """
        def f(x,y):
            """
            das
            """
            x = x[0].decode()
            x = self.read_image(x)
            return x,y
        img,label = tf.numpy_function(f,[x,y],[tf.float32,tf.int32])
        img.set_shape([32,32,3])
        # y = tf.one_hot(tf.cast(label, tf.uint8), 7)
        # y = tf.cast(tf.one_hot(tf.cast(y, tf.int32), 7), dtype=y.dtype)
        # y = to_categorical(y, num_classes=7)

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
        self.len = len(x)
        ds = tf.data.Dataset.from_tensor_slices((x,y))#.repeat()
        # print(ds)
        ds = ds.shuffle(buffer_size=1000)
        ds = ds.map(self.preprocess) #.as_numpy_iterator()
        ds = ds.batch(batch,drop_remainder=True)
        ds = ds.map(self._fixup_shape)
        
        # ds = ds.repeat()
        # ds = ds.prefetch(tf.contrib.data.AUTOTUNE)
        # print(ds)
        ds = ds.prefetch(2)
        return ds
    def _fixup_shape(self,images, labels):
        images.set_shape([None, None, None, 3])
        labels.set_shape([None, 7]) # I have 19 classes
        return images, labels
    def execute(self,count=100,log_image=False):
        """
        Load data and Return a Dataset
        log_image=True : print an image before dataset and test loop
        ->input in cli? 
        """
        

        # imgs,labels = self.load_data('data/',count) #these are only training data
        # test_imgs, test_labels = self.load_data('data/',count) #these are only training data
        # #-> tests unavailable at the moment new dataset is required

        ""
        self.df = pd.read_csv('./data/HAM10000_metadata')
        self.df = self.df.drop_duplicates(subset=['lesion_id'])
        self.classes = self.df['dx'].unique().tolist()
        print(self.classes)
        # this needs to be filtered first to account for imbalanced data before it can be taken forward [:count] will not suffice

        # labels
        # self.df['dx'] = self.df['dx'].apply(lambda x: classes.index(x)).tolist() #[:count]
        # paths
        self.df['dx'] = self.df['dx'].map(str)
        self.df['image_id'] = self.df['image_id'].apply(lambda x: x + '.jpg') #[:count]
        self.df.set_index('image_id')
        # print(self.df['dx'].dtype)
        # print(self.df['dx'].values)
        self.len = len(self.df['dx'].values)
        #, ds_val
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_dataframe(
            dataframe=self.df,
            directory='data/HAM10000_images/',
            x_col ='image_id',
            y_col ='dx',
            color_mode="rgb",
            target_size=(32, 32),
            batch_size=128,
            validate_filenames=False,
            shuffle=False)
        # print('hahahahahahaha',test_generator.classes)

        # ds_train = self.tfdata_generator(imgs,labels,is_training=True)#
        # ds_train = self.train_val_split(imgs,labels,split=.2)
        #
        # ds_test = self.tf_dataset(test_imgs,test_labels)
        return test_generator#ds_train#, ds_val, ds_test
    

    def tfdata_generator(self,images, labels, is_training, batch_size=128):
        '''Construct a data generator using `tf.Dataset`. '''
        def map_fn(image, label):
            '''Preprocess raw data to trainable input. '''
            print(image)
            print(label)
            x = tf.reshape(tf.cast(image, tf.float32), (28, 28, 1))
            y = tf.one_hot(tf.cast(label, tf.uint8), 7)
            return x, y
        
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        # print(dataset)
        if is_training:
            dataset = dataset.shuffle(1000)  # depends on sample size
        dataset = dataset.map(map_fn)
        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat()
        dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

        return dataset 
    def train_val_split(self,imgs,labels,split): # set this up
        """
        Return a a dataset for train,validate and test
        """
        s = int(len(imgs)*split)
        ds_train = self.tf_dataset(imgs[:s],labels[:s])
        ds_val = self.tf_dataset(imgs[s:],labels[s:])
        # get the train data and split it
        # get the test data and format it
        # return 3 X,y pairs
        return ds_train#, ds_val 