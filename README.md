# Advance-House-Price-Prediction
Machine Learning Project

## Steps in Data Science projects
1. Data Analysis or Data processing
2. Feature Engineering
3. Feature Selection
4. Model Building
5. Model Deployment

## Data Analysis Phase

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
dataset = pd.read_csv('train.csv')
dataset.head(15)
dataset.shape

Dividing Data Into Two categories: Nummerical_col(having numeric features) and data(having categorical features)


### Aim is understand about the data
### Splitting The Data Into Numerical & Categorical 
## Exploratory Data Analysis
IN Data Analysis What All Things We do

1. Missing values
2. All The numerical variables
3. Explore about the char variables
4. Finding the relationship about the features
5. Relationship Between the independent and dependent feature(sales price)

# Data has Categorical Feature Only
data = dataset.select_dtypes(include=[np.object])
data['SalePrice'] = dataset.SalePrice
print(data.shape)
data
data.head()
data.shape

## Missing Values
#Here we will check percentage of all nan values present in each feature
1. Print List of the features which has missing values
2. Print the features names and percentage of missing values

data.isnull().mean() #here data refere to chategorical features

#This Loop Shows Us The Unique Values Present In Each Feature As Well As Number Of Total Values 
#And Number Of Missing Values In Each Feature
for i in data.columns:
    print(data.groupby(data[i])[i].count())
    print('Total Number of Missing Values : ',data[i].isnull().sum())
    print('Total Number of Values Present : ',data[i].count())
    print('_________________________________________')
    
#As we Observed That Columns Like ['Alley','PoolQC','MiscFeature','Fence'] 
#Has More Than 70 % missing Value So the Only Way we Handle it is to drop it
drop = data.loc[:,['Alley','PoolQC','MiscFeature','Fence']]
data = data.drop(drop,axis =1)
data.isnull().sum()


## Replacing Missing Values By Most_Frequent_Value
#THis We actually Do First Take Out all The Unique values In Every Paticular Feature,
#then Arrange them in decending Order, The Values Comes First(The most occuring unique Value In That Feature).
#We Replace That Missing Values With That Most Occuring Value, 
#We Do same Technique With for all Other Feature Having Missing Value 
#We Create Function For This

data.groupby(['BsmtQual'])['BsmtQual'].count()    
data['BsmtQual'].value_counts().sort_values(ascending = False).index[0]

def impute_nan(data,variable):
    most_frequent_value = data[variable].value_counts().sort_values(ascending = False).index[0]
    data[variable].fillna(most_frequent_value,inplace = True)
    
impute_nan(data,'MasVnrType')
impute_nan(data,'Electrical')
impute_nan(data,'BsmtQual')
impute_nan(data,'BsmtCond')
impute_nan(data,'BsmtExposure')
impute_nan(data,'BsmtFinType1')
impute_nan(data,'BsmtFinType2')
impute_nan(data,'FireplaceQu')
impute_nan(data,'GarageType')
impute_nan(data,'GarageFinish')
impute_nan(data,'GarageQual')
impute_nan(data,'GarageCond')

# Counting The Values Graphically 

figure, ax = plt.subplots(3,2, figsize=(20,20))
sns.countplot(data['MSZoning'],ax=ax[0,0])
sns.countplot(data['Street'],ax= ax[0,1])
sns.countplot(data['LotShape'],ax= ax[1,0])
sns.countplot(data['LandContour'],ax=ax[1,1])
sns.countplot(data['Utilities'],ax= ax[2,0])
sns.countplot(data['LotConfig'],ax= ax[2,1])



























