import numpy as np 
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

# Class for data imputing
class CategoricalImputer(BaseEstimator, TransformerMixin):
    
    def __init__(self, variables=None):
        if(not isinstance(variables, list)):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for features in self.variables:
            X[features] = X[features].fillna('Missing')
        return X

# Numerical imputer
class NumericalImputer(BaseEstimator, TransformerMixin):
    
    def __init__(self, variables=None):
        if(not isinstance(variables, list)):
            self.variables = [variables]
        else:
            self.variables = variables
    
    def fit(self, X, y=None):
        self.imputer_dict_ = {}
        for features in self.variables:
            self.imputer_dict_[features] =  X[features].mode()[0]
        return self

    def transform(self, X):
        X = X.copy()
        for features in self.variables:
            X[features].fillna(self.imputer_dict_[features], inplace=True)
        return X

# Transformation of time related variables
class TemporalVariableEstimator(BaseEstimator, TransformerMixin):
    
    def __init__(self, variables=None, reference_variable=None):
        if(not isinstance(variables, list)):
            self.variables = [variables]
        else:
            self.variables = variables
        self.reference_variable = reference_variable

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for features in self.variables:
            X[features] = X[self.reference_variable] - X[features]
        return X

# Categorical labels with low frequency
class RareLabelCategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None, total=0.05):
        if(not isinstance(variables, list)):
            self.variables = [variables]
        else:
            self.variables = variables
        self.total = total   

    def fit(self, X, y=None):
        self.encoder_dict_ = {}
        for var in self.variables:
            temp = pd.Series(X[var].value_counts() / np.float(len(X)))
            self.encoder_dict_[var] = list(temp[temp >= self.total].index)
        return self

    def transform(self, X):
        X = X.copy()
        for features in self.variables:
            X[features] = np.where(X[features].isin(self.encoder_dict_[features]), 
                X[features], 'Rare')
        return X

# Encoder for categorical variables
class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):
        if(not isinstance(variables, list)):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y):
        temp = pd.concat([X, y], axis=1)
        temp.columns = list(X.columns) + ['target'] 

        self.encoder_dict_ = {}

        for var in self.variables:
            t = temp.groupby([var])['target'].mean().sort_values(ascending=True).index
            self.encoder_dict_[var] = {k: i for i, k in enumerate(t, 0)}
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].map(self.encoder_dict_[feature])
        return X

# Log-transform variables
class LogTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, variables=None):
        if(not isinstance(variables, list)):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        for feature in self.variables:
            if any(X[feature]<=0):
                pass
            else:
                X[feature] = np.log(X[feature])
        return X

class DropUnnecessaryFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, variables_to_drop=None):
        self.variables = variables_to_drop
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X = X.drop(self.variables, axis=1)
        return X



