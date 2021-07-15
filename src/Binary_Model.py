import numpy as np
import pandas as pd
from pandas.core.algorithms import mode
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import ReLU
from keras_tuner.tuners import RandomSearch
from tensorflow.python import saved_model
from tensorflow.python.data.ops.dataset_ops import make_one_shot_iterator
from datetime import datetime
import keras_tuner as kt
from tensorflow.python.keras.metrics import AUC, Precision
from sklearn.metrics import classification_report
#define parameters in init, can steps be paused or resumed?
#
#get data from 33k set.
# switch model to sigmoid single end
# show imbalance
#
#
class BinaryModelWorker():
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
   
    def build_model(self,hp):
        """
        Create the model that will be used
        """
        model = keras.Sequential()
        # model.add(input_layer = Input(shape=x.shape[1:]))
        model.add(keras.layers.Conv2D(32,padding='same',kernel_size=3,activation='relu',input_shape=(64,64,3)))
        model.add(keras.layers.MaxPool2D(pool_size=2))    
        model.add(keras.layers.Dropout(.5))            

        model.add(keras.layers.Conv2D(64,padding='same',kernel_size=3,activation='relu'))
        model.add(keras.layers.MaxPool2D(pool_size=2))    

        model.add(keras.layers.Flatten())
        # if(self.hp is not None):   
        #     model.add(keras.layers.Dense(hp.Choice('dense_layer',[64,128,256,512,1024]),activation='relu'))   
        model.add(keras.layers.Dense(1024,activation='relu'))    
        model.add(keras.layers.Dense(64,activation='relu'))    
        model.add(keras.layers.Dense(1,activation='sigmoid'))
        
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=.3),
                loss=keras.losses.BinaryCrossentropy(),
                metrics=[keras.metrics.Accuracy(), keras.metrics.AUC(),keras.metrics.Precision(), keras.metrics.Recall()])#create parameter for
        
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
        self.model = self.build_model(None)
        print(self.model.summary())
        train,val,test = self.pipe.execute()#val,
        print('starting fit')
        self.model.fit(train,validation_data=val, epochs=52,verbose=True)# validation_data=validation_data,validation_steps=1,
        print('ending fit')
        eval = self.model.evaluate(test)
        pred = self.model.predict(test)
        predicted = np.argmax(pred, axis=1)
        report = classification_report(np.argmax(test['target'], axis=1), predicted)
        print(report)
        
        # self.report(eval)

        pass
    def tune(self):
        """
        Use the parameter tuning hooks to pass in paramter sets
        """
        tuner = RandomSearch(
            self.build_model,
            objective='val_accuracy',
            max_trials=100,
            overwrite = True
        )
        all_data = self.pipe.get_all_data_gen()
        print(all_data.shape)
        tuner.search(all_data,epochs=10,batch_size=32)
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
        model.fit(validation_data, steps_per_epoch=self.pipe.len // 64,epochs=1,verbose=True)# validation_data=validation_data,validation_steps=1,
        print('ending fit')
        
        arg = np.argmax(model.predict(validation_data,), axis=1)[0]
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