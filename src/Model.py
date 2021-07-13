import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras_tuner.tuners import RandomSearch


class Model():
    """
    A Convolutional Neural Network class wrapper for tf.keras
    """
    def __init__(self,pipeline):
        """
        Initialize the CNN with data
        """
        #take in train,val,test from pipeline
        pass
    def build_model(self):
        """
        Create the model that will be used
        """
        pass
    def train(self):
        """
        Run fitting of the model
        """
        #take in cross val data and or batch data
        pass
    def report(self):
        """
        Create a text file containing all the metrics used to score
        Return a list of length k for: TP,FP,TN,FN (Confusion Matrix)
        """
        # send a report to a text file with all the info about the model (se other metrics)
        pass
    def evaluate(self):
        """
        Test the model and return a report
        Args:
        eval_type = 'Val' | 'Test'
        """
        # check the model against val or test data
        pass
    def tune(self):
        """
        Use the parameter tuning hooks to pass in paramter sets
        """
        #run hyper parameter tuning of the model
        #run train variants using specific parameters
        pass
    def save(self):
        """
        Save the model to the directory that is in the pipeline
        """
        pass
    def load(self):
        """
        Load the model stored in the directory
        """
        pass


#reference code from other project

# def build_model(hp):
#     model = keras.Sequential()
    
#     model.add(keras.layers.AveragePooling2D(6,3,input_shape=(300,300,1)))
    
#     model.add(keras.layers.Conv2D(64,3,activation='relu'))    
#     model.add(keras.layers.Conv2D(32,3,activation='relu'))    
#     model.add(keras.layers.Conv2D(16,3,activation='relu'))    
    
#     model.add(keras.layers.MaxPool2D(2,2))    
#     model.add(keras.layers.Dropout(0.5))    
    
#     model.add(keras.layers.Flatten())    
#     model.add(keras.layers.Dense(hp.Choice('dense_layer',[64,128,256,512,1024]),activation='relu'))    
#     model.add(keras.layers.Dense(3,activation='softmax'))
    
#     model.compile(optimizer='adam',
#              loss=keras.losses.SparseCategoricalCrossentropy(),
#              metrics=['accuracy'])
#     return model

# tuner = RandomSearch(
#     build_model,
#     objective='val_accuracy',
#     max_trials=5
# )
# tuner.search(ds,validation_data=(ds))
# best_model = tuner.get_best_models()[0]