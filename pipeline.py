from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

import preprocessors as pp # Other file
import config # Other file

price_pipeline = Pipeline([
    ('categorical_imputer',
        pp.CategoricalImputer(variables=config.CATEGORICAL_VARS_WITH_NA)),
    ('numerical_imputer', 
        pp.NumericalImputer(variables=config.NUMERICAL_VARS_WITH_NA)),
    ('temporal_variables',
        pp.TemporalVariableEstimator(
            variables=config.TEMPORAL_VARS, 
            reference_variable=config.TEMPORAL_REFERENCE)),
    ('rare_label_encoder',
        pp.RareLabelCategoricalEncoder(
            total=0.05,
            variables=config.CATEGORICAL_VARS)),
    ('categorical_encoder',
        pp.CategoricalEncoder(variables=config.CATEGORICAL_VARS)),
    ('log_transformer',
        pp.LogTransformer(variables=config.NUMERICAL_LOG_VARS)),
    ('drop_features',
        pp.DropUnnecessaryFeatures(variables_to_drop=config.DROP_FEATURES)),
    ('scaler', MinMaxScaler()),
    ('rf_model', RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0))
])