import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras_tuner.tuners import RandomSearch
from tensorflow.python.data.ops.dataset_ops import make_one_shot_iterator
from datetime import date


class ModelWorker():
    """
    A Convolutional Neural Network class wrapper for tf.keras
    """
    def __init__(self,pipeline):
        """
        Initialize the CNN with data
        """
        self.pipe = pipeline
        self.model = None
        self.hp_dict = {}
        pass
    def register_hp(self,name):
        """
        Passes model_hp to dictionary {'name':[]}
        """
        self.hp_dict[name] = []
        pass
    def build_model(self,):
        """
        Create the model that will be used
        """
        model = keras.Sequential()
    
        # model.add(keras.layers.AveragePooling2D(6,3,input_shape=(300,300,1)))
        # model.add(input_layer = Input(shape=x.shape[1:]))
        model.add(keras.layers.Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(32,32,3)))
        model.add(keras.layers.MaxPool2D(pool_size=(2,2)))    
        model.add(keras.layers.Conv2D(32,kernel_size=(3,3),activation='relu'))
        model.add(keras.layers.MaxPool2D(pool_size=(2,2)))    
        model.add(keras.layers.Conv2D(32,kernel_size=(3,3),activation='relu'))
        model.add(keras.layers.MaxPool2D(pool_size=(2,2)))    

        model.add(keras.layers.Dropout(0.5))            
        model.add(keras.layers.Flatten())
        # if(self.hp is not None):   
        #     model.add(keras.layers.Dense(hp.Choice('dense_layer',[64,128,256,512,1024]),activation='relu'))   
        model.add(keras.layers.Dense(128,activation='relu'))    
        model.add(keras.layers.Dense(7,activation='softmax'))
        
        model.compile(optimizer='adam',
                loss=keras.losses.CategoricalCrossentropy(),
                metrics=[keras.metrics.Accuracy(),keras.metrics.Precision(),keras.metrics.Recall()])#create parameter for
        
        return model
    def train(self):
        """
        Run fitting of the model on a cross validated run
        """
        pass
    def report(self,metrics):
        """
        Create a text file containing all the metrics used to score
        Return a list of length k for: TP,FP,TN,FN (Confusion Matrix)
        """
        report_sum = self.model.Summary()
        report_scores = metrics
        
        rep_full = report_sum + '\n' + report_scores + '\n'
        
        file_name = date.today() + '.txt'
        
        f = open(file_name, "w")
        f.write(rep_full)
        f.close()
    def evaluate(self):
        """
        Test the model and return a report
        Args:
        eval_type = 'Val' | 'Test'
        """
        self.report()

        pass
    def tune(self):
        """
        Use the parameter tuning hooks to pass in paramter sets
        """
        
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
    def demo(self):
        """
        run a basic pass on a small data set
        """
        a = self.pipe.execute(count=1000)
        model = self.build_model()
        # X_train = list(map(lambda x: x[0], b))
        # y_train = list(map(lambda x: x[1], b))
        # model.fit(X_train,y_train)
        print(model.summary())

        BATCH_SIZE = 50
        # train_data = a.batch(BATCH_SIZE)#list(map(lambda x: x[0], a.batch(BATCH_SIZE)))
        validation_data = a#.take(1)#.batch(BATCH_SIZE)#list(map(lambda x: x[0], b.batch(BATCH_SIZE))) 
        # validation_data = validation_data.set_shape([32,32,3])
        # print('validationdata  ' , validation_data)
        # print('validationdata  ' , validation_data.next())
        # .make_one_shot_iterator()

        print('starting fit')
        model.fit(validation_data, steps_per_epoch=self.pipe.len // 128,epochs=1, validation_data=validation_data,validation_steps=1,verbose=True)
        print('ending fit')
        
        arg = np.argmax(model.predict(validation_data), axis=1)[0]
        print( 'pred ______________',arg)
        print(self.pipe.classes[arg])

        


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