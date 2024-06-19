# MachineLearning_1

Welcome to the Sign Language Recognition Project! This repository contains code for recognizing signs and translating them into corresponding letters. The project uses machine learning to train a model that can interpret sign language symbols. Below is a brief description of each file in the repository and instructions on how to set up and run the project.

Project Structure

app.py: This is the main application file. It contains the code for running the web or desktop application where users can interact with the trained model to recognize signs.

trainmodel.py: This file is responsible for training the machine learning model. It includes data preprocessing, model architecture, training loops, and evaluation metrics.

function.py: This file contains helper functions used throughout the project, such as data augmentation, image preprocessing, and utility functions.

collectdata.py: This script is used to collect and preprocess data for training the model. It include code for capturing images from a webcam, labeling data, and organizing the dataset.

data.py: This file handles data loading and manipulation. It includes functions for loading the dataset, splitting it into training and validation sets, and other data-related tasks.

Setup Instructions:
Clone the Repository
git clone https://github.com/yourusername/sign-language-recognition.git
cd sign-language-recognition


Install dependencies:
- scikit-learn 1.5.0
  pip install scikit-learn
- mediapipe 0.10.14
  pip install mediapipe
- opencv-python 4.10.0.84
  pip install opencv-python
- tensorflow
  pip install tensorflow

Collect Data:[Create your Dataset as Dataset is not provided in repository]
Run the collectdata.py script to gather and preprocess data for training the model

Run data.py

Run function.py

Train the model
Once the data is collected, train the model using the trainmodel.py script

Run the Application
After training the model, you can run the application using the app.py script by app.py



Happy Coding!

RutujaWagh19(Rutuja Wagh).
