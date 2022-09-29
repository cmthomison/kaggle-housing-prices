"""
Exploratory data analysis for Kaggle housing prices.
"""

import sys
import os
from pathlib import Path
import pandas
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder

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

review = sale_deets
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

ATTRIBUTES

MSZoning *
- RL (low density res) contains most of the homes and has the largest spread of
sale price values.
- RM has a tighter IQR, though a lower median. Perhaps we'll want to look at
this field in conjunction with neighborhood?
- Floating Village what! Highest median and few outliers. Perhaps a condo or 
houseboat situation where all units are fairly close in value?

MSSubClass *
- A lot going on here- we may want to group some of these classes, possibly
by age.
- This one also may be interesting to look at by neighborhood.

Street
- Paved vs. gravel
- Only a few gravel street homes, though median is slightly lower.
- Probalby won't end up using this one, at least initially.

LotShape
- Irregularly shaped lots tends to have a higher median home value?
- My guess is that inner suburb/urban lots are on a grid and the fancier
suburbs have slightly irregular lots.
- Again, something to compare against neighborhoods.

LandContour
- Not sure that we'll get too much out of this one with the overwhelming 
majority falling into Lvl.
- I'm wondering if the hillside could be nice houses along the river?

Utilities
- Only one without sewer- probably not going to be very useful.

LotConfig *
- Interesting here is cul-d-sac vs no cul-d-sac- flags 'subdivision' and
sale prices that come along with it

LandSlope
- Not a huge difference in the box plots- probably will not use initially.

Neighborhood *
- I think this will probably be a big one, especially paired with LotConfig,
Total Square Footage, and MSZoning.

BldgType
- There might be something here- I would actually expect a bit more of a 
difference between these groups.

HouseStyle *
- Looks like there is something here, though I would expect this to correlate
to total SF (2 Story, 1 Story, etc.).
- This does include if some of the area is unfininished- we may want to adjust
the total SF calc to account for finished/unfinished basements.

RoofStyle
- Hip vs Gable may be interesting since there are quite a few data points in
each and hip roofs cost more than gable roofs.

RoofMatl
- Very few records that are not CompShg.

Exterior1st & Exterior2nd
- May be something here, but probably will layer this one on after starting
with neighborhood, size, etc.

MasVnrType *
- ACTUALLY- it may make sense to do a a bit of FE to create a feature with
stone/brick/vinyl/other values.

Foundation *
- Could foundation type indicate base quality of the home?

BsmtFinType1 & BsmtFinType2
- Good quality living quarters vs. not perhaps? This one has a higher median
than all other options that are fairly close.

GarageFinish *
- This might be a feature that helps to split some of the higher end homes
(rough finish vs finished)

Alley
- Lots of null values- probably will not use, at least initially.

AMENITIES
- To move through this a bit quicker, I'm going to list those featues I'm
interested in using only.

Might be useful
- CentralAir
    - mostly Y, but those generally have a higher SalePrice.
- GarageType
    - May combine CarPort, Basment, and 2Types as 'Other'
    - NA means no garage
- Fireplaces
- There were a few others that could be useful, but didn't stand out as much
as those above.

SIZE
- TotalBsmtSF
- 1stFlrSF
- 2ndFlrSF
- GrLivArea * (visually strong correlation)
- GarageArea & GarageCars
- BedroomAbvGr (maybe paired with neighborhood)
- TotRmsAbvGrd
- FullBath

NEW FEATS
- total_baths
- sf_above_grade
- sf_total * (visually strong correlation)
    - appear to be two outliers- perhaps commercial or other funky zoning?

AGE
- YearBuilt
    - May be something here, definitely not the only factor though
- YearRemodAdd & GarageYrBlt
    - Similar- looks like there could be something

SALE DEETS
- SaleType
    - Possibly make a 'new home' flag
- Weirdly not seeing huge differences across month and year sold- perhaps
we can try grouping into quarters.
"""

keep = [
    'OverallQual', 'KitchenQual', 'HeatingQC', 'ExterQual', 'BsmtQual',
    'Functional', 'MSZoning', 'MSSubClass', 'LotConfig', 'Neighborhood',
    'MasVnrType', 'Foundation', 'GarageFinish', 'CentralAir',
    'GarageType', 'Fireplaces', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
    'GrLivArea', 'GarageArea', 'GarageCars', 'BedroomAbvGr', 'TotRmsAbvGrd',
    'FullBath', 'total_baths', 'sf_above_grade', 'sf_total', 'YearBuilt',
    'YearRemodAdd', 'GarageYrBlt', 'SaleType', 'SalePrice'
]

sns.heatmap(data[keep].corr())

# Some light FE
int_data = data[keep].reset_index(drop=True)

# New house flag
int_data['new_house'] = np.where(data['SaleType']=='New', True, False)
int_data.drop(columns='SaleType', inplace=True)

# Ordinal Encoding (via pd.factorize)
# KitchenQual
cat_kitchen_qual = pd.Categorical(
    int_data['KitchenQual'], 
    categories=['Po','Fa','TA','Gd','Ex'],
    ordered=True
)

labels, unique = pd.factorize(cat_kitchen_qual, sort=True)
int_data['KitchenQual_e'] = labels

# HeatingQC
cat_heating_qual = pd.Categorical(
    int_data['HeatingQC'], 
    categories=['Po','Fa','TA','Gd','Ex'],
    ordered=True
)

labels, unique = pd.factorize(cat_heating_qual, sort=True)
int_data['HeatingQC_e'] = labels

# ExterQual
cat_exter_qual = pd.Categorical(
    int_data['ExterQual'], 
    categories=['Po','Fa','TA','Gd','Ex'],
    ordered=True
)

labels, unique = pd.factorize(cat_exter_qual, sort=True)
int_data['ExterQual_e'] = labels

# BsmtQual
cat_bsmt_qual = pd.Categorical(
    int_data['BsmtQual'], 
    categories=['Po','Fa','TA','Gd','Ex'],
    ordered=True
)

labels, unique = pd.factorize(cat_bsmt_qual, sort=True)
int_data['BsmtQual_e'] = labels

# Functional
cat_functional = pd.Categorical(
    int_data['Functional'], 
    categories=['Sal','Sev','Maj2','Maj1','Mod','Min2','Min1','Typ'],
    ordered=True
)

labels, unique = pd.factorize(cat_functional, sort=True)
int_data['Functional_e'] = labels

# CentralAir
cat_centralair = pd.Categorical(
    int_data['CentralAir'], 
    categories=['N', 'Y'],
    ordered=True
)

labels, unique = pd.factorize(cat_centralair, sort=True)
int_data['CentralAir_e'] = labels

# Foundation- combine some values.
def fe_foundation(row):
    if row['Foundation'] == 'PConc':
        return 'PConc'
    elif row['Foundation'] == 'CBlock':
        return 'CBlock'
    else:
        return 'Other'

int_data['Foundation_e'] = int_data.apply(fe_foundation, axis=1)

# Garage Type- combine some values.
def fe_garagetype(row):
    if row['GarageType'] in ['Attchd','BuiltIn']:
        return 'Attached'
    elif row['GarageType'] == 'Detchd':
        return 'Detchd'
    else:
        return 'Other'

int_data['GarageType_e'] = int_data.apply(fe_garagetype, axis=1)

# NEXT 'TotalBsmtSF'

# One Hot Encoding
ohe = [
    'MSZoning', 'MSSubClass', 'LotConfig', 'Neighborhood', 
    'MasVnrType', 'LotConfig', 'GarageFinish', 'CentralAir', 'Foundation_e',
    'GarageType_e'
]

drop = [
    'KitchenQual', 'HeatingQC', 'ExterQual', 'BsmtQual', 'Functional',
    'Foundation', 'CentralAir', 'GarageType'
]