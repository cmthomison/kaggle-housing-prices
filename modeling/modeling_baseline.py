""" 
Script to generate our baseline model where we'll use total square footage
to predict SalePrice with linear regression.

Example of a tranformation pipeline here: https://www.kaggle.com/code/fk0728/feature-engineering-with-sklearn-pipelines/notebook
https://adamnovotny.com/blog/custom-scikit-learn-pipeline.html
"""


import datetime
import multiprocessing
import pandas as pd
import numpy as np
import datetime as dt
from sklearn import linear_model
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
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

        # Fill nulls (within the function to retain column names).
        X[sf_cols] = X[sf_cols].fillna(0)

        # Sum to get total square footage.
        X.loc[:, "sf_total"] = X[sf_cols].sum(axis=1)
        return X

# Create a pipeline for the baseline model.
numeric_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='constant'))
])

col_processor = ColumnTransformer(transformers=[
    ('number', numeric_pipeline, ['1stFlrSF', '2ndFlrSF','TotalBsmtSF'])
])

test_pipeline = Pipeline(steps=[
    ("fe_squarefootage", FeatTotalSf()),
    ("feature_selection", FeatureSelector(['sf_total']))
])

preprocessing = Pipeline(steps=[
    ("fe_squarefootage", FeatTotalSf()),
    ("feature_selection", FeatureSelector(['sf_total']))
])

linear_regression = LinearRegression()

lr_pipeline = Pipeline(steps=[
    ("preprocessing", preprocessing),
    ("modeling", linear_regression)
])

lr_model = lr_pipeline.fit(X_train, y_train)
y_predict = lr_model.predict(X_valid)

# Score the model.
r2 = metrics.r2_score(y_valid, y_predict)
mse = metrics.mean_squared_error(y_valid, y_predict)
rmse = metrics.mean_squared_error(y_valid, y_predict, squared=False)
rmsle = metrics.mean_squared_log_error(y_valid, y_predict, squared=False)

print(f'R2: {r2}')
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'RMSLE: {rmsle}')

# Export scores.
export = {
    'model': 'Baseline',
    'algorithm': 'Linear Regression',
    'date_completed': dt.datetime.now(),
    'r2': r2,
    'mse': mse,
    'rmse': rmse,
    'rmsle': rmsle
}

pd.DataFrame(export, index=[0]).to_csv('~/Documents/projects/kaggle-housing-prices/modeling/results/modeling_results.csv', index=False)

# Retrain the model on the full data before submitting a prediction.
X = data.drop(columns='SalePrice')
y = data[['SalePrice']]

full_lr_model = lr_pipeline.fit(X, y)

# Generate predictions for submission.
X_test = pd.read_csv(r'~/Documents/projects/kaggle-housing-prices/data/test.csv')
y_test = full_lr_model.predict(X_test)

submission = X_test[['Id']].reset_index(drop=True)
submission['SalePrice'] = y_test
submission.to_csv(r'~/Documents/projects/kaggle-housing-prices/data/submission_base_lr_1.csv', index=False)