import numpy as np 
import pandas as pd

from sklearn.model_selection import train_test_split
import joblib

import pipeline # Other file
import config # Other file

def run_training():
    print("Training model...")

    data = pd.read_csv(config.TRAINING_DATA_FILE)
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.FEATURES], 
        data[config.TARGET],
        test_size=0.1,
        random_state=0
    )

    pipeline.breast_cancer_classification.fit(X_train[config.FEATURES], y_train)
    joblib.dump(pipeline.breast_cancer_classification, config.PIPELINE_NAME)

    print("Training has finished.")

if __name__ == '__main__':
    run_training()