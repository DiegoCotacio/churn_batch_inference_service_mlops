# Funciones de preprocesamiento de datos para dataset de training

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

def convert_dt(df):   
    df['totalcharges'] = pd.to_numeric(df['totalcharges'], errors='coerce')
    df['clienteidentifier'] = df['clienteidentifier'].astype(str)
    df['multiplelines'] = df['multiplelines'].astype(str)
    df['internetservice'] = df['internetservice'].astype(str)
    df['onlinesecurity'] = df['onlinesecurity'].astype(str)
    df['onlinebackup'] = df['onlinebackup'].astype(str)
    df['deviceprotection'] = df['deviceprotection'].astype(str)
    df['techsupport'] = df['techsupport'].astype(str)
    df['streamingtv'] = df['streamingtv'].astype(str)
    df['streamingmovies'] = df['streamingmovies'].astype(str)
    df['contract'] = df['contract'].astype(str)
    df['paymentmethod'] = df['paymentmethod'].astype(str)
    df['gender'] = df['gender'].astype(str)
    df['paperlessbilling'] = df['paperlessbilling'].astype(str)
    df['partner'] = df['partner'].astype(str)
    df['dependents'] = df['dependents'].astype(str)
    df['phoneservice'] = df['phoneservice'].astype(str)
    df['seniorcitizen'] = df['seniorcitizen'].astype(str)
    df['monthlycharges'] = df['monthlycharges'].astype(float)
    df['totalcharges'] = df['totalcharges'].astype(float)
    df['tenure'] = df['tenure'].astype(float)
    df['churn'] = df['churn'].astype(str)
    return df
    #df['fecha_ingreso'] = pd.to_datetime(df['fecha_ingreso']).dt.date


def reindex_df(df):
    new_order = ['clienteidentifier','multiplelines','internetservice','onlinesecurity','onlinebackup',
             'deviceprotection','techsupport','streamingtv','streamingmovies','contract','paymentmethod',
             'gender','paperlessbilling','partner','dependents','phoneservice','seniorcitizen',
             'monthlycharges','totalcharges','tenure','fecha_ingreso']
    
    data = df.reindex(columns=new_order)
    return data

data = pd.DataFrame(df)
data.columns = data.columns.str.lower()


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
        #'churn' : {'Yes' : 1, 'No': 0}
    }

    data.replace(mapping, inplace = True)
    
    return data

