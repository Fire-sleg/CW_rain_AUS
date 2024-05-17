import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample
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

    smote = SMOTE()
    X_resampled, Y_resampled = smote.fit_resample(X, Y)


    # Розділяємо датасет на класи
    class_majority = ds[ds['RainTomorrow'] == 0.0]
    class_minority = ds[ds['RainTomorrow'] == 1.0]

    # Випадково відбираємо з більшим класом кількість екземплярів, рівну кількості екземплярів у меншому класі
    class_majority_downsampled = resample(class_majority,
                                        replace=False,  # без заміни
                                        n_samples=len(class_minority),  # збалансована кількість екземплярів
                                        random_state=68)  # для відтворюваності

    # Об'єднуємо вирівняні класи
    #ds = pd.concat([class_majority_downsampled, class_minority])

    # split ds
    X = ds.drop(columns=['RainTomorrow'])
    Y = ds['RainTomorrow']

    # models
    models = {
        'rf': RandomForestClassifier(**model_best_hyperparams.params_rf),
    }

    # training model
    model = models[model_name]
    model.fit(X_resampled, Y_resampled)
    #model.fit(X, Y)

    # saving model
    with open(f'C:/D/Final Project/wetherAUS prediction/models/{model_name}.pkl', 'wb') as f:
        pickle.dump(model, f)