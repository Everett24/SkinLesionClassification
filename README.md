# SkinLesionClassification

### This project is under heavy development and is not ready to be used at this time

## The Project
This project looks at dirmetology images of skin lesions and classifies them into one of 7 categories.
This project looks at dirmetology images of skin lesions and classifies them as benign or malignant.
 ### The goal is to get a model with a high recall score
 - As of now the model is not the most succesful

## Steps to follow along
- Fork the repository
- Download the data from https://www.kaggle.com/c/siim-isic-melanoma-classification/data?select=tfrecords <br />
    (This version is a larger dataset than the ones available at: ISIC &  Dataverse)
- Make sure you have a extracted 'train.csv' to the data folder as well as 'jpeg/train' 
- Run runtime.py
- Configure settings as desired in runtime.py

## Tour of the Repository
- notebooks contains exploratory tests of the code that became the src code
- src contains <br />
  Pipeline.py: A class used to load and preprocess data for the model <br />
  Model.py: A class used to easily configure the CNN <br />
  Binary_Pipeline.py: A class used to load and preprocess binary data for the model <br />
  Bnary_Model.py: A class used to easily configure the binary CNN <br />
  runtime.py: A file used to cleanly create and test a model <br />
