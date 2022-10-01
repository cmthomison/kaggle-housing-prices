import pandas as pd
import numpy as np
import os
import sys


# Load data.
train = pd.read_csv(r'~/Documents/projects/kaggle-housing-prices/data/train_fe.csv')

# Let's start with linear regression.
# At this time, I'm not going to transform any variables- but I may at a later
# time.