# SkinLesionClassification

# This project is under heavy development and is not ready to be used at this time

## The Project
This project looks at dirmetology images of skin lesions and classifies them into one of 7 categories.

## Steps to follow along
- Fork the repository
- Download the data from https://www.kaggle.com/c/siim-isic-melanoma-classification/data?select=tfrecords <br />
    (This version is a larger dataset than the ones available at: ISIC &  Dataverse) //embed these links
- Run configure_data.sh targeting your download from the project root directory
- Run runtime.py
    (Note this will take a while the first time you run it, as there will be no previously saved model
- Configure settings as desired in runtime.py

## Tour of the Repository
- notebooks contains exploratory tests of the code that became the src code
- src contains <br />
  Pipeline.py: A class used to load and preprocess data for the model <br />
  Model.py: A class used to easily configure the CNN <br />
  runtime.py: A file used to cleanly create and test a model <br />
- data will be created from configure_data.sh after you download the data
