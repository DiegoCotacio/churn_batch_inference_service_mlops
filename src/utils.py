import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()


# Al importar la feature store:
# pip install --upgrade hsfs confluent_kafka