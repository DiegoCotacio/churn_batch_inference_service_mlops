# Funciones de preprocesamiento de datos para dataset de training

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

# Remoción de duplicados
def remove_duplicates(data):
    data.drop_duplicates(inplace = True)
    return data

# imputación de nulos
def imput_missing_values(data):
    # Para variables categoricas
    categorical_cols = data.select_dtypes(include='object').columns
    categorical_imputer = SimpleImputer(strategy = 'most_frequent')
    data[categorical_cols] = categorical_imputer.fit_transform(data[categorical_cols])

    #Para variables numericas
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).colmuns
    numeric_imputer = SimpleImputer(strategy='mean')
    data[numeric_cols] = numeric_imputer.fit_transform(data[numeric_cols])

    return data

# Normalización de variables numericas
def normalize_numeric_variable(data):
    numeric_cols = ['monthlycharges', 'totalcharges', 'tenure']
    scaler = MinMaxScaler()
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

    return data

#Codificación de variables categoricas manual
def map_categorical_variables(data):
    mapping = {
        'multiplelines': {'No': 0, 'Yes': 0.5, 'No phone service': 1},
        'internetservice': {'Fiber optic': 0, 'DSL': 0.5, 'No': 1},
        'onlinesecurity': {'No': 0, 'Yes': 0.5, 'No internet service': 1},
        'onlinebackup': {'No': 0, 'Yes': 0.5, 'No internet service': 1},
        'deviceprotection': {'No': 0, 'Yes': 0.5, 'No internet service': 1},
        'techsupport': {'No': 0, 'Yes': 0.5, 'No internet service': 1},
        'streamingtv': {'No': 0, 'Yes': 0.5, 'No internet service': 1},
        'streamingmovies': {'No': 0, 'Yes': 0.5, 'No internet service': 1},
        'contract': {'Month-to-month': 0, 'One year': 0.5, 'Two year': 1},
        'gender' : {'Male' : 0, 'Female': 1},
        'paymentmethod': {'Electronic check': 0, 'Mailed check': 0.3, 'Bank transfer (automatic)': 0.6, 'Credit card (automatic)': 1},
        'paperlessbilling' : {'Yes' : 1, 'No': 0},
        'partner' : {'Yes' : 1, 'No': 0},
        'dependents' : {'Yes' : 1, 'No': 0},
        'phoneservice' : {'Yes' : 1, 'No': 0},
        'seniorcitizen' : {'0' : 0, '1': 1},
        'churn' : {'Yes' : 1, 'No': 0}
    }

    data.replace(mapping, inplace = True)
    
    return data

