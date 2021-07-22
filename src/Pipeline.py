import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from scipy import stats
from sklearn.preprocessing import LabelEncoder
class DataPipeline():
    """
    A class used for creating pipelines loading image data for classification
    """
    def __init__(self,path):
        """
        Initialize the pipeline
        """
        self.path = path

    def load_df(self,path): 
        """
        Get the data from the directories; filterable
        """
        self.df = pd.read_csv(path)
        self.clean_df()
        self.relabel()
        self.resample_df()

    def clean_df(self):
        df = self.df
        df = df.drop(['lesion_id','dx_type','age','sex','localization','dataset'],axis=1)
        df['image_id'] = df['image_id'].apply(lambda x: x + '.jpg')
        df = df.drop(df[df['dx'] == 'nv'].sample(frac=0.85).index)
        self.classes = df['dx'].unique().tolist()
        self.weights = df.groupby('dx').size()/df.shape[0]
        self.len = len(df['dx'].values)
        self.df = df
    def relabel(self):
        le = LabelEncoder()
        le.fit(self.df['dx'])
        LabelEncoder()
        self.df['label'] = le.transform(self.df["dx"]) 
        
    def resample_df(self):
        df = self.df
        df_0 = df[df['label'] == 0]
        df_1 = df[df['label'] == 1]
        df_2 = df[df['label'] == 2]
        df_3 = df[df['label'] == 3]
        df_4 = df[df['label'] == 4]
        df_5 = df[df['label'] == 5]
        df_6 = df[df['label'] == 6]

        n_samples=500 
        df_0_balanced = resample(df_0, replace=True, n_samples=n_samples, random_state=42) 
        df_1_balanced = resample(df_1, replace=True, n_samples=n_samples, random_state=42) 
        df_2_balanced = resample(df_2, replace=True, n_samples=n_samples, random_state=42)
        df_3_balanced = resample(df_3, replace=True, n_samples=n_samples, random_state=42)
        df_4_balanced = resample(df_4, replace=True, n_samples=n_samples, random_state=42)
        df_5_balanced = resample(df_5, replace=True, n_samples=n_samples, random_state=42)
        df_6_balanced = resample(df_6, replace=True, n_samples=n_samples, random_state=42)

        #Combined back to a single dataframe
        self.df = pd.concat([df_0_balanced, df_1_balanced, 
                                    df_2_balanced, df_3_balanced, 
                                    df_4_balanced, df_5_balanced, df_6_balanced])
        
    def get_all_data_gen(self):
        train_full = self.load_df(self.path)
        train_generator = self.get_img_gen(train_full,'image_id','label', self.path )
        return train_generator

    def execute(self):
        """
        Load data and Return a Dataset
        log_image=True : print an image before dataset and test loop
        """
        #split
        train_full = self.load_df('./data/HAM10000_metadata')
        train_split,  test =  train_test_split(train_full,shuffle=False,test_size=.2)
        train,val = train_test_split(train_split,shuffle=False,test_size=.2)
        #make generators
        train_generator = self.get_img_gen(train,'image_id','dx',self.path)
        val_generator = self.get_img_gen(val,'image_id','dx',self.path)
        test_generator = self.get_img_gen(test,'image_id','dx',self.path)

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
            color_mode="rgb",
            target_size=(32, 32),
            classes=self.classes,
            batch_size=6,
            validate_filenames=False,
            shuffle=True,
            subset=sub)
        return test_generator