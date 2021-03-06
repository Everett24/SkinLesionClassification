from Model import ModelWorker
from Pipeline import DataPipeline
from Binary_Model import BinaryModelWorker
from Binary_Pipeline import BinaryDataPipeline
import My_Models as mm

"""
A file to run from terminal to test and evaluate a model
"""

# My_Models.py -> 'Name'_Build_Model(): pass in hp if a 'tune' function is passed in with 'build_model'
# Pass a function in to model
# 
#
#
#
#
#
#


if __name__ == '__main__':
    
    '''
        Create a build model function in My_Models
        Create a data pipeline by passing in a image path
        Create a model by passing in a pipeline and a build model function 
    '''


    pipe = DataPipeline('data/HAM10000_images')
    model = ModelWorker(pipeline=pipe, bm=mm.MultiClass_BuildModel_Deep)
    # model.tune()
    model.evaluate()
    #model.save()
    # tf.debugging.set_log_device_placement(True)

    # print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    # print('GPUS DATA:    ',tf.config.list_physical_devices('GPU'))
    # # Create some tensors
    # a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    # b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    # c = tf.matmul(a, b)

    # print(c)



    # pipe = BinaryDataPipeline()
    
    # df = pipe.load_df('./data/train.csv')
    # print( df.groupby('target').size()/df.shape[0])
    # model = BinaryModelWorker(pipeline=pipe)
    # model.evaluate()


    #model.tune()
    

    #test with a batch all of one class for each class
    #test with an equal split of classes

    #sudo apt install nvidia-cuda-toolkit