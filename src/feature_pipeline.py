import pandas as pd
import numpy as np
import hopsworks
import os


df= pd.read_csv('dataset.csv')

"""
preprocesamiento base: imputacion de nulos, eliminacion de duplicados.

"""

#setting hopsworks access
project = hopsworks.login()
fs = project.get_feature_store()


churn_dataset = fs.get_or_create_feature_group(
    name="churn_dataset_train",
    version=1,
    description="Este es batch inference pipeline.",
    primary_key=['customerID'],
)

