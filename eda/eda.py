"""
Exploratory data analysis for Kaggle housing prices.
"""

import sys
import os
from pathlib import Path
import pandas
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.append('..')
from support import data_functions as df


# Load data.
data = df.load_train()

# Ran into several errors testing out pandas profiling, so going back to
# manual visualizations.

# What I'm looking for:
# [ ] Distribution of dependent variable (housing prices)
# [ ] Distribution of independent variables
# [ ] Correlation of independent variables with dependent variable
# [ ] Missing values in independent variables
# [ ] Ideas for feature engineering

# Distribution of housing prices
sns.histplot(data, x='SalePrice')
# Right-skewed- a long tail for the fancy houses.

# I'm also going to manually categorize the features into a few thematic
# groups: size, age, quality, amenities, and sale details
size = [
    'BsmtFinSF1',
    'BsmtFinSF2',
    'BsmtUnfSF',
    'TotalBsmtSF',
    '1stFlrSF',
    '2ndFlrSF',
    'LowQualFinSF',
    'WoodDeckSF',
    'OpenPorchSF',
    'LotArea',
    'LotFrontage',
    'MasVnrArea', 
    'GrLivArea', 
    'GarageArea',
    'GarageCars',
    'PoolArea',
    'BedroomAbvGr',
    'KitchenAbvGr',
    'TotRmsAbvGrd',
    'BsmtFullBath',
    'BsmtHalfBath',
    'FullBath',
    'HalfBath'
]

age = [
    'YearBuilt',
    'YearRemodAdd',
    'GarageYrBlt'
]

quality = [
    'OverallQual',
    'OverallCond',
    'KitchenQual',
    'GarageQual',
    'GarageCond',
    'HeatingQC',
    'PoolQC',
    'Condition1',
    'Condition2',
    'ExterQual',
    'ExterCond',
    'BsmtQual',
    'BsmtCond',
    'Functional',
    'FireplaceQu'
]

amenities = [
    'Heating',
    'CentralAir',
    'Electrical',
    'GarageType',
    'Fence',
    'MiscFeature',
    'MiscVal',
    'EnclosedPorch',
    '3SsnPorch',
    'ScreenPorch',
    'Fireplaces',
    'BsmtExposure',
    'PavedDrive'
]

attributes = [
    'MSZoning',
    'MSSubClass',
    'Street',
    'LotShape',
    'LandContour',
    'Utilities',
    'LotConfig',
    'LandSlope',
    'Neighborhood',
    'BldgType',
    'HouseStyle',
    'RoofStyle',
    'RoofMatl',
    'Exterior1st',
    'Exterior2nd',
    'MasVnrType',
    'Foundation',
    'BsmtFinType1',
    'BsmtFinType2',
    'GarageFinish',
    'Alley'
]

sale_deets = [
    'SaleType',
    'SaleCondition',
    'MoSold',
    'YrSold'
]

cat_cols = size + age + quality + amenities + attributes + sale_deets
to_categorize = [x for x in data.columns.tolist() if x not in cat_cols]

# There is a bit of feature engineering I want to do right off the bat to make
# a few of these features a little more useful initially.
# [ ] Total bathrooms
# [ ] First and second floor square footage and total square footage (inc bas)

# Total bathrooms
def fe_total_baths(row):

    full_baths = row['FullBath'] + row['BsmtFullBath']
    half_baths = (row['HalfBath'] + row['BsmtHalfBath'])/2

    baths = full_baths + half_baths

    return baths

data['total_baths'] = data.apply(fe_total_baths, axis=1)

# Square footage
data['sf_above_grade'] = data['1stFlrSF'] + data['2ndFlrSF']
data['sf_total'] = data['sf_above_grade'] + data['TotalBsmtSF']

new_feats = ['total_baths', 'sf_above_grade', 'sf_total']

# Distribution of independent variables.
num_cols, text_cols = df.col_types(data)

# Review value counts and null values for all features.
all_feat = size + age + quality + amenities + attributes + sale_deets + new_feats

# Check to see if a feature is duplicated.
# Street was duplicated, but is now corrected.
review = [x for x in all_feat if all_feat.count(x) > 1]

# Loop through features to review value counts and null values.

review = quality
for col in review:

    # Get some summary info.
    print(col)
    print(data[col].dtypes)
    print(data[col].value_counts())
    print(data[col].describe())
    print(f'Null values: {data[col].isnull().sum()}')

    # We can also look at a bar chart of values.
    hist = sns.histplot(data=data, x=col).set(title=col)
    plt.show()

    # Generate a box plot to compare to the depenedent variable.
    box = sns.boxplot(data=data, x=col, y='SalePrice').set(title=col)
    plt.show()

    # Take a look at a scatterplot as well.
    scatter = sns.scatterplot(data=data, x=col, y='SalePrice').set(title=col)
    plt.show()

# Review results and jot down some notes; I may transfer to a juptyer notebook
# eventually.

"""
QUALITY features
OverallQual *
- There is definitely some relationship here- especially evident on the boxplot.
- As quality increases, so does sale price.
- No null values.

OverallCond
- There is a little something here, but not anything like OverallQual.
- Part of the issue might be that over half of the records have a rating of 5,
and the distribution for records with this ranking actually has a higher median
sales price than a ranking of 9.
- Possibly a data collection issue- some people thought the top score was 5?
- Not sure this one is going to be super useful...

KitchenQual *
KitchenQual: Kitchen quality

       Ex	Excellent
       Gd	Good
       TA	Typical/Average
       Fa	Fair
       Po	Poor

- Nearly all are Good or Typica/Average, but it there does seem to be a
relationship.
- For models where multicollinearity is an issue, we'll definitely want to
compare quality features.
- No null values!

GarageQual
- Nearly all values are Typical/Average, though Fair does have a lower median
Sale Price.

GarageCond
- Looks very similar to GarageQual.
- Might be able to use this with some other fields to create a 'rough shape'/
'investor special' flag.
- Both GarageQual and Garage Cond have 81 null values.

HeatingQC *
- The medians are fairly close, though 'Excellent' is visibly above the rest.
I'm guessing it is being pulled up by the very high value homes.
- I'm curious if this can help explain a lower than expected sales price- like
everything is good, but you'll need to replace the furnace next year, so you
agree on 10k less than asking/expected.

PoolQC
- Only 7 data points here.
- I think the objective with the pools will be to determine how fancy of a pool
it is, possibly in ground/above ground.
- Only one of the 7 pools is in a crazy expensive house.

Condition1 & Condition2
- These are interesting- really not how I would like to handle proximity, but
we can work with it.
- Nothing major jumps out to me with the box plots.
- I think what might be interesting would be to do a bit of feature engineering
to create count fields for 'nearby_positive' and 'nearby_negative'.

ExterQual *
- This looks like a good one- we can see a relationship in the box plots,
though it is clear that this feature alone could not be used.

ExterCond
- This one looks a little less useful than ExterQual- most of the values are
Typical/Average.
- We may want to compare to ExterCond/ExterQual against each other.

BsmtQual *
- We do have 37 null values, but there seems to be some relationship here in
the boxplots.

BsmtCond
- Similar thoughts to ExterQual/ExterCond.

Functional *
- Okay so perhaps this is our 'investor special' field.
- I'm not seeing anything immediately in the boxplots, BUT these types of
deductions can happen to any type of home with any base value. A 10k furnace
may need to be replaced in a 500k home just as it might in 250k home.
- I'm flagging this for further testing/review.

FireplaceQu
- I'm thinking maybe we can do some FE to flag 'problematic fireplaces' with
Poor/Fair values to use in conjunction with the fireplace count.
"""