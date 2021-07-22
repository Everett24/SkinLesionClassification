from tensorflow import keras

def MultiClass_BuildModel_Standard():
    pass

def Binary_BuildModel_Standard():
    pass

def Binary_BuildModel_Deep():
    model = keras.Sequential()
        
    model.add(keras.layers.Conv2D(16,padding='same',kernel_size=3,activation='relu',input_shape=(32,32,1)))
    model.add(keras.layers.MaxPool2D(pool_size=2))    

    model.add(keras.layers.Conv2D(128,padding='same',kernel_size=3,activation='relu'))
    model.add(keras.layers.MaxPool2D(pool_size=2))    
    
    model.add(keras.layers.Conv2D(64,padding='same',kernel_size=3,activation='relu'))
    model.add(keras.layers.MaxPool2D(pool_size=2)) 
    model.add(keras.layers.Dropout(0.5))


    model.add(keras.layers.Flatten())
    
    model.add(keras.layers.Dense(64,activation='relu'))    
    model.add(keras.layers.Dense(1,activation='sigmoid'))
    
    model.compile(optimizer=keras.optimizers.Adam(),
            loss=keras.losses.BinaryCrossentropy(),
            metrics=[keras.metrics.Recall(),keras.metrics.Accuracy(),keras.metrics.Precision(),keras.metrics.AUC()])#create parameter for
    
    return model
    
def MultiClass_BuildModel_Deep():
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(1024, (3, 3), activation="relu", input_shape=(32, 32, 3)))
    #model.add(BatchNormalization())
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))  
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Conv2D(512, (3, 3),activation='relu'))
    #model.add(BatchNormalization())
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))  
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Conv2D(256, (3, 3),activation='relu'))
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
            
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=.00001),
            loss=keras.losses.CategoricalCrossentropy(),
            metrics= ['acc', keras.metrics.Recall(), keras.metrics.AUC(),keras.metrics.Precision()])#create parameter for
    return model

def Binary_BuildModel_Shallow():
    pass

def MultiClass_BuildModel_Shallow():
    pass

