import numpy as np
import pandas as pd
from pandas.core.algorithms import mode
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# from tensorflow.keras.layers import ReLU
from keras_tuner.tuners import RandomSearch
# from tensorflow.python import saved_model
# from tensorflow.python.data.ops.dataset_ops import make_one_shot_iterator
from datetime import datetime
import keras_tuner as kt
# from tensorflow.python.keras.metrics import AUC, Precision
# from sklearn.metrics import classification_report
# from keras.preprocessing import image
import matplotlib.pyplot as plt
from sklearn.utils import resample
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
        
        model.add(layers.Conv2D(256,padding='same',kernel_size=3,activation='relu',input_shape=(32,32,1)))
        model.add(layers.MaxPool2D(pool_size=2))    

        model.add(layers.Conv2D(128,padding='same',kernel_size=3,activation='relu'))
        model.add(layers.MaxPool2D(pool_size=2))    
        
        model.add(layers.Conv2D(64,padding='same',kernel_size=3,activation='relu'))
        model.add(layers.MaxPool2D(pool_size=2)) 
        model.add(layers.Dropout(0.5))


        model.add(layers.Flatten())
        # if(self.hp is not None):   
        #     model.add(keras.layers.Dense(hp.Choice('dense_layer',[64,128,256,512,1024]),activation='relu'))   
        model.add(layers.Dense(64,activation='relu'))    
        model.add(layers.Dense(1,activation='sigmoid'))
        
        model.compile(optimizer=keras.optimizers.Adam(),
                loss=keras.losses.BinaryCrossentropy(),
                metrics=[keras.metrics.Recall(),keras.metrics.Accuracy(),keras.metrics.Precision(),keras.metrics.AUC()])#create parameter for
        
        return model
    def print_graph(self):
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']
        epochs = range(1, len(loss) + 1)
        plt.plot(epochs, loss, 'y', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        pass
    def evaluate(self):
        """
        Test the model and return a report
        Args:
        eval_type = 'Val' | 'Test'
        """
        self.model = self.build_model(None)
        print(self.model.summary())
        train,val,test = self.pipe.execute()
        print('starting fit')
        self.history = self.model.fit(train,class_weight=self.pipe.weights, validation_data=val, epochs=10,verbose=True)
        print('ending fit')
        
        self.print_graph()
        # self.print_example(test)
        eval = self.model.evaluate(test)
        
        
        # pred = self.model.predict(test)
        # predicted = np.argmax(pred, axis=1)
        # true = np.argmax(test['benign_malignant'].astype(int))
        # report = list(zip(true,predicted))
        #report = classification_report(np.argmax(test['benign_malignant'].astype(int)), predicted)
        # print(report)

        
        
        pass
    def print_example(self,gen):
        x,y = gen.next()
        for i in range(0,1):
            image = x[i]
            proba = self.model.predict(image)
            best = np.argsort(proba[0])[:-2:-1]    
            plt.imshow(image.transpose(2,1,0))
            for i in range(2):
                print("{}".format(self.pipe.classes[best[i]])+" ({:.3})".format(proba[0][best[i]]))   
            plt.show()

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

