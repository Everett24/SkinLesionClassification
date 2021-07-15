from Model import ModelWorker
from Pipeline import DataPipeline
from Binary_Model import BinaryModelWorker
from Binary_Pipeline import BinaryDataPipeline

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
    # pipe = DataPipeline()
    # model = ModelWorker(pipeline=pipe)
    # model.tune()
    #model.evaluate()
    #model.save()
    pipe = BinaryDataPipeline()
    model = BinaryModelWorker(pipeline=pipe)
    model.evaluate()
    # model.tune()
    