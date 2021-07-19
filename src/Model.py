import numpy as np
# import pandas as pd
# from pandas.core.algorithms import mode
# import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras.layers import ReLU
from keras_tuner.tuners import RandomSearch
# from tensorflow.python import saved_model
# from tensorflow.python.data.ops.dataset_ops import make_one_shot_iterator
from datetime import datetime
# import keras_tuner as kt
# from tensorflow.python.keras.metrics import AUC, Precision

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
        pass
   
    def build_model(self):
        """
        Create the model that will be used
        """
        model = keras.models.Sequential()
        model.add(keras.layers.Conv2D(256, (3, 3), activation="relu", input_shape=(32, 32, 3)))
        #model.add(BatchNormalization())
        model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))  
        model.add(keras.layers.Dropout(0.3))

        model.add(keras.layers.Conv2D(128, (3, 3),activation='relu'))
        #model.add(BatchNormalization())
        model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))  
        model.add(keras.layers.Dropout(0.3))

        model.add(keras.layers.Conv2D(64, (3, 3),activation='relu'))
        #model.add(BatchNormalization())
        model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))  
        model.add(keras.layers.Dropout(0.3))
        model.add(keras.layers.Flatten())

        model.add(keras.layers.Dense(32))
        model.add(keras.layers.Dense(7, activation='softmax'))
                
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=.001),
                loss=keras.losses.CategoricalCrossentropy(),
                metrics= ['acc', keras.metrics.Recall()])#'acc', keras.metrics.AUC(),keras.metrics.Precision(), keras.metrics.Recall(thresholds=hp.Choice('thresh',[0.,.25,.5,.75] ))])#create parameter for
        
        return model
   
    def report(self,metrics):
        """
        Create a text file containing all the metrics used to score
        Return a list of length k for: TP,FP,TN,FN (Confusion Matrix)
        """
        report_sum = ''#self.model.summary()
        report_scores = [str(m) for m in metrics] 
        
        rep_full = report_sum + '\n' + report_scores + '\n'
        
        file_name = '/logs/' + str(datetime.ctime()) + '.txt'
        
        f = open(file_name, "w")
        f.write(rep_full)
        f.close()

    def evaluate(self):
        """
        Test the model and return a report
        Args:
        eval_type = 'Val' | 'Test'
        """
        self.model = self.build_model()
        print(self.model.summary())
        train,val,test = self.pipe.execute()
        print('starting fit')
        self.model.fit(train, epochs=5,validation_data=val,verbose=True)
        print('ending fit')
        eval = self.model.evaluate(test)

        pass
    def tune(self):
        """
        Use the parameter tuning hooks to pass in paramter sets
        """

        tuner = RandomSearch(
            self.build_model,
            objective='val_loss',
            max_trials=30
        )
        train,val,test = self.pipe.execute()
        self.pipe.get_all_data_gen()
        tuner.search(train,epochs=3,validation_data=val,batch_size=32)
        bm = tuner.get_best_models()[0]
        bm.evaluate(test)
        tuner.get_best_models()[0].save('./models/' + str(datetime.ctime) + "___Best_Model")
        pass
    def save(self):
        """
        Save the model to the directory that is in the pipeline
        """
        self.model.save('./models/' + str(datetime.ctime))
        pass
    def load(self,model):
        """
        Load the model stored in the directory
        """
        pass
    def demo(self):
        """
        run a basic pass on a small data set
        """
        a = self.pipe.execute()
        model = self.build_model()
       
        print(model.summary())

        validation_data = a 

        print('starting fit')
        model.fit(validation_data, steps_per_epoch=self.pipe.len // 64,epochs=1,verbose=True)
        print('ending fit')
        
        arg = np.argmax(model.predict(validation_data,), axis=1)[0]
        print( 'pred ______________',arg)
        print(self.pipe.classes[arg])
