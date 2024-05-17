import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.simplefilter('ignore')

def preprocess_train_data(ds: pd.DataFrame) -> pd.DataFrame:
    ds = ds.drop(["Date", "Evaporation", "Sunshine", "Cloud9am", "Cloud3pm"], axis=1) 

    ds.dropna(subset=['RainTomorrow'], inplace=True)

    for column in ds.columns:
        if ds[column].dtype == 'object':  
            ds[column].fillna('Missing', inplace=True)
        else: 
            median_value = ds[column].median()
            ds[column].fillna(median_value, inplace=True)
    
    

    le = LabelEncoder()

    le_col = ds.select_dtypes(include=['object']).columns

    for x in le_col:
        ds[x] = le.fit_transform(ds[x])

    with open('C:/D/Final Project/wetherAUS prediction/pipeline/settings/label_encoder.pkl', 'wb') as pick:
        pickle.dump(le, pick)



    Rainfall_upper_limit, Rainfall_lower_limit = find_skewed_boundaries(ds, 'Rainfall', 4)
    outliers_Rainfall = np.where(ds['Rainfall'] > Rainfall_upper_limit, True,
                       np.where(ds['Rainfall'] < Rainfall_lower_limit, True, False))
    ds = ds.loc[~outliers_Rainfall, ]


    cols_to_scale = ds.select_dtypes(include=['float64', 'int64']).columns

    scaler = MinMaxScaler()
    ds[cols_to_scale] = scaler.fit_transform(ds[cols_to_scale])

    with open('C:/D/Final Project/wetherAUS prediction/pipeline/settings/scaler.pkl', 'wb') as pick:
        pickle.dump(scaler, pick)

    ds.to_csv('C:/D/Final Project/wetherAUS prediction/data/data_train.csv', index=False)

    return ds

def preprocess_testing_data(ds: pd.DataFrame) -> pd.DataFrame:
    ds = ds.drop(["Date", "Evaporation", "Sunshine", "Cloud9am", "Cloud3pm"], axis=1) 

    ds.dropna(subset=['RainTomorrow'], inplace=True)

    for column in ds.columns:
        if ds[column].dtype == 'object':  
            ds[column].fillna('Missing', inplace=True)
        else: 
            median_value = ds[column].median()
            ds[column].fillna(median_value, inplace=True)
    
    
    with open('C:/D/Final Project/wetherAUS prediction/pipeline/settings/label_encoder.pkl', 'rb') as pick:
        le: LabelEncoder = pickle.load(pick)

    le_col = ds.select_dtypes(include=['object']).columns

    for x in le_col:
        ds[x] = le.fit_transform(ds[x])

    Rainfall_upper_limit, Rainfall_lower_limit = find_skewed_boundaries(ds, 'Rainfall', 4)
    outliers_Rainfall = np.where(ds['Rainfall'] > Rainfall_upper_limit, True,
                       np.where(ds['Rainfall'] < Rainfall_lower_limit, True, False))
    ds = ds.loc[~outliers_Rainfall, ]


    cols_to_scale = ds.select_dtypes(include=['float64', 'int64']).columns


    with open('C:/D/Final Project/wetherAUS prediction/pipeline/settings/scaler.pkl', 'rb') as pick:
        scaler: MinMaxScaler = pickle.load(pick)
    
    ds[cols_to_scale] = scaler.fit_transform(ds[cols_to_scale])
    ds[cols_to_scale].describe()
    ds.to_csv('C:/D/Final Project/wetherAUS prediction/data/data_test.csv', index=False)
    return ds



def find_skewed_boundaries(ds, variable, distance):

    IQR = ds[variable].quantile(0.75) - ds[variable].quantile(0.25)

    lower_boundary = ds[variable].quantile(0.25) - (IQR * distance)
    upper_boundary = ds[variable].quantile(0.75) + (IQR * distance)

    return upper_boundary, lower_boundary