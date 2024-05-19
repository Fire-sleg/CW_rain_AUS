import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from preprocessing_data import preprocess_train_data
import model_best_hyperparams

def train_model(file_name: str = 'train.csv', model_name: str = 'rf'):
    # loading ds
    ds = pd.read_csv('C:/D/Final Project/wetherAUS prediction/data/' + file_name)

    # preprocessing ds
    ds = preprocess_train_data(ds)

    # split ds
    X = ds.drop(columns=['RainTomorrow'])
    Y = ds['RainTomorrow']

    # models
    models = {
        'rf': RandomForestClassifier(**model_best_hyperparams.params_rf),
    }

    # training model
    model = models[model_name]
    model.fit(X, Y)

    # saving model
    with open(f'C:/D/Final Project/wetherAUS prediction/models/{model_name}.pkl', 'wb') as f:
        pickle.dump(model, f)