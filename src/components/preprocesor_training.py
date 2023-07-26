# Funciones de preprocesamiento de datos para dataset de training

import sys
import os
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

        def get_data_transformer_object(self):

            try:
                numerical_columns = []
                categorical_columns = []

                num_pipeline = Pipeline(
                    steps = [
                        ("imputer", SimpleImputer(strategy= "median")),
                        ("scaler", StandardScaler())
                    ])
                
                cat_pipeline = Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy= "most_frequent")),
                        ("one_hot_encoder", OneHotEncoder()),
                        ("scaler", StandardScaler(with_mean=False))
                    ]
                )

                logging.info(f"Categorical columns: {categorical_columns}")
                logging.info(f"Numerical columns: {numerical_columns}")

                preprocessor = ColumnTransformer(
                    [
                        ("num_pipeline", num_pipeline, numerical_columns),
                        ("cat_pipeline", cat_pipeline, categorical_columns)
                    ]
                 )
                 
                return preprocessor
            
            except Exception as e:
                raise CustomException(e,sys)

   
    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Completed loading of train/test data")

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            target_columns_name = "churn"

            # X and y in Train dataset
            input_feature_train_df = train_df.drop(columns=[target_columns_name], axes = 1)
            target_feature_train_df = train_df[target_columns_name]

            #X and y in Test dataset
            input_feature_test_df = test_df.drop(columns=[target_columns_name], axis = 1)
            target_feature_test_df = test_df[target_columns_name]

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            raise CustomException(e, sys)
        






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

