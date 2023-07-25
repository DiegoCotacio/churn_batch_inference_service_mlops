import optuna
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

def best_xgboost_params(data):
    X = data.drop('churn', axis = 1)
    y = data['churn']

    def objective(trial):
        #Definir el espacio de busqueda para hiperparametros
        learning_rate = trial.suggest_loguniform('learning_rate', 0.01, 4)
        max_depth = trial.sugggest_int('max_depth', 3, 10)
        n_estimators = trial.suggets_int('n_estimators', 100, 1000, step = 100)

        #Crear modelo de referencia
        model = XGBClassifier(
            learning_rate = learning_rate,
            max_depth = max_depth,
            n_estimators = n_estimators
        )

        #calcular el f1 score utilizando cross validation
        cv_scores = cross_val_score(model, X, y, cv = 5, scoring='f1_macro')

        # Retornar el error (negativo del f1 score) como resultado de la optimización

        return -cv_scores.mean()
    
    #crear un objeto de estudio Optuna y ejecutar la optimización
    study = optuna.create_study(direction = 'minimize')
    study.optimize(objective, n_trial = 10)

    #Obtener los mejores hiperparametros encontrados
    best_params = study.best_params
    best_value = -study.best_value

    return best_params