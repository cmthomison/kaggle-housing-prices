from random import random
import pandas as pd
import numpy as np
import os
import sys

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error

# Load data.
train = pd.read_csv(r'~/Documents/projects/kaggle-housing-prices/data/train_fe.csv')
stest = pd.read_csv(r'~/Documents/projects/kaggle-housing-prices/data/test_fe.csv')
otest = pd.read_csv(r'~/Documents/projects/kaggle-housing-prices/data/test.csv')

# Let's start with linear regression.
# At this time, I'm not going to transform any variables- but I may at a later
# time.
# I get overly hung up on this: https://data.library.virginia.edu/normality-assumption/

# Split the data.
x = train.drop(columns='SalePrice')
y = train[['SalePrice']]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=23)

# Linear regression
mod_lr = LinearRegression()
mod_lr.fit(x_train,y_train)

y_prediction =  mod_lr.predict(x_test)
y_prediction

# Scoring the model
score = r2_score(y_test,y_prediction)
print(f"R2: {score}")
print(f"MSE: {mean_squared_error(y_test,y_prediction)}")
print(f"RMSE: {mean_squared_error(y_test,y_prediction,squared=False)}")
print(f"RMSLE: : {mean_squared_log_error(y_test,y_prediction,squared=False)}")

# ^ RMSLE is not an option because there is a negative prediction.
# We may need to handle by logging the SalePrice before training.

# I apparently have an extra feature in my stest set..
review = [x for x in stest.columns.tolist() if x not in train.columns.tolist()]
# It is a MSSubClass that wasn't in the training set.
# For now, we will drop it.
stest.drop(columns='MSSubClass_150', inplace=True)

# Generate a kaggle submission
lr_pred = mod_lr.predict(stest)
sub = otest[['Id']]
sub['SalePrice'] = lr_pred
sub.to_csv(r'~/Documents/projects/kaggle-housing-prices/data/submission_lr_1.csv', index=False)

# Let's try Random Forest before reworking the codebase.
mod_rf = RandomForestRegressor(n_estimators=100, random_state=23)
  
# Fit the model.
mod_rf.fit(x_train, y_train)  

# Evaluate.
y_prediction =  mod_rf.predict(x_test)
y_prediction

# Scoring the model
print(f"MSE: {mean_squared_error(y_test,y_prediction)}")
print(f"RMSE: {mean_squared_error(y_test,y_prediction,squared=False)}")
print(f"RMSLE: : {mean_squared_log_error(y_test,y_prediction,squared=False)}")

# Generate a kaggle submission
rf_pred = mod_rf.predict(stest)
sub = otest[['Id']]
sub['SalePrice'] = rf_pred
sub.to_csv(r'~/Documents/projects/kaggle-housing-prices/data/submission_rf_1.csv', index=False)

# Next going to try Ridge Regression (as I suspect we do not need all of the
# variables that are currenlty included).