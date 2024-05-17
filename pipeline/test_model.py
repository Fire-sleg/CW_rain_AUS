import pandas as pd
import pickle
import numpy as np
from preprocessing_data import preprocess_testing_data

def test_model(file_name: str = 'new_input.csv', model_name: str = 'rf'):
    # loading ds
    ds = pd.read_csv('C:/D/Final Project/wetherAUS prediction/data/' + file_name)

    # preprocessing ds
    ds = preprocess_testing_data(ds)

    # split ds
    X = ds.drop(columns=['RainTomorrow'])
    Y = ds['RainTomorrow']

    # testing model
    with open(f'C:/D/Final Project/wetherAUS prediction/models/{model_name}.pkl', 'rb') as f:
        model = pickle.load(f)

    predictions = model.predict(X)
    df = pd.DataFrame({'Y': Y, 'Predictions': predictions})
    # saving predictions
    pd.DataFrame(df).to_csv('C:/D/Final Project/wetherAUS prediction/data/predictions.csv', index=False)

    # printing accuracy of predictions
    accuracy = (predictions == Y).mean()
    print(f'Accuracy: {accuracy}')