
import optuna
#from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import pandas as pd
import numpy as np
import pickle

#utils
from utils.optuna_hyp_params import best_xgboost_params
from utils.preprocesor_training import remove_duplicates, imput_missing_values, normalize_numeric_variable, map_categorical_variables

# carga de datos de feature view
data = pd.read_csv = 'ssss'

#----------------- FEATURE ENGINEERING----------------------------

df_proc = remove_duplicates(data)
df_proc = imput_missing_values(data)
df_proc = normalize_numeric_variable(data)
df_proc = map_categorical_variables(data)

#------------------- ENTRENAMIENTO:-------------------------------

#Steps:
 # 1. Dividir conjunto de datos
 # 2. Instanciar modelo XGBoost
 # 3. Instanciar metricas de evaluación r1_score, accuracy y confusion matrix
 # 4. Entrenar el modelo con StratifiedKfold cross validation 

#dividir conjunto de datos
X = df_proc.drop(['churn', 'cliente_identifier'], axis = 1)
y = df_proc['churn']

# instancia parametros y modelo
params = best_xgboost_params(df_proc)
xgb = XGBClassifier(**params)

# instancial metricas de evaluación
f1_scores = []
accuracies = []
confusion_matrices = []


