# CW Binary classification

This project created in puprpose to predict next-day rain in Australia, based on 10 years of daily weather observations from many locations across Australia. 
Main point of this project - create model which will help to predict next-day rain.


## Installation

To install the project, follow these steps:

1. Download .zip file
2. Navigate to the project file
3. Check if you have all libraries on your local device for start the application

## Usage

Data(train and test) should be placed in folder "data". They are creating there automatically after running file separate_data, but for this you need to change pathes all over each file in your drectory.

In the pipeline folder you can find four files - train_model for training, test_model - for building predictions, additional one - preprocessing - for checking data before training and testing, and start. Start one is the one you need to run for training and testing models.
In the models you can see models - they`re created after training model and using it for testing and prediction.


## Models

In this project I used 1 model:
- Random Forest;

The model accuracy is 86%