import pandas as pd 
import joblib
import config

def make_predictions(input_data):
    myPipeline = joblib.load(filename=config.PIPELINE_NAME)
    resultado = myPipeline.predict(input_data)
    return resultado