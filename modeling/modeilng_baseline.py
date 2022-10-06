""" 
Script to generate our baseline model where we'll use total square footage
to predict SalePrice with linear regression.

Example of a tranformation pipeline here: https://www.kaggle.com/code/fk0728/feature-engineering-with-sklearn-pipelines/notebook
https://adamnovotny.com/blog/custom-scikit-learn-pipeline.html
"""

<<<<<<< HEAD
=======

import datetime
import multiprocessing
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.pipeline import FeatureUnion, Pipeline 
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

sys.path.append('..')
from support import data_functions as df


# Load train data.
data = df.load_train()

# Split the data.
X_train, X_valid, y_train, y_valid = train_test_split(
    data.drop(columns='SalePrice'), data[['SalePrice']],
    test_size=0.2, random_state=23
)

# Test out creating a class for an sklearn pipeline.
# Get selected features in pipeline.
class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return X.loc[:, self.feature_names].copy(deep=True)

# Calculate total square footage.
class FeatTotalSf(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        sf_cols = ['1stFlrSF', '2ndFlrSF', 'TotalBsmtSF']
        X.loc[:, "sf_total"] = X[sf_cols].sum(axis=1)
        return X

# PICK UP HERE
# - Select SF features
# - Use simple imputer strategy='constant' (0 fill) for any missing
# values
# Then train a linear regression model on the split

test_pipeline = Pipeline(steps=[
    ("fe_squarefootage", FeatTotalSf()),
    ("feature_selection", FeatureSelector(['sf_total']))
])

test = test_pipeline.fit_transform(data)
>>>>>>> modeling
