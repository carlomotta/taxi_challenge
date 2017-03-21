import pandas as pd
import matplotlib  
import matplotlib.pyplot as plt  
import matplotlib.path as mplPath
from scipy.stats import ttest_ind, f_oneway, lognorm, levy, skew, chisquare
import numpy as np
import seaborn as sns
import datetime as dt
from clean_data import clean_data
from engineer_features import *
from functions_exploratory_data_analysis import *
from cross_validation_param_optimization import *

df = pd.read_csv('2015_Green_Taxi_Trip_Data.csv') 

#print df.describe()
# List unique values in the df['pickup_datetime'] column
# df.pickup_datetime.unique()
df = clean_data(df)

print "size before feature engineering:", df.shape
df = engineer_features(df)
print "size after feature engineering:", df.shape



## code to compare the two Tip_percentage identified groups
# split data in the two groups
df1 = df[df.Tip_percentage>0]
df2 = df[df.Tip_percentage==0]

"""
# generate histograms to compare
fs = 14 # fontsize
fig,ax=plt.subplots(2,1,figsize=(8,10))
plt.tick_params(labelsize=fs)
df.Tip_percentage.hist(bins = 20,normed=True,ax=ax[0])
ax[0].set_xlabel('Tip (%)', fontsize=fs)
ax[0].set_title('Distribution of Tip (%) - All transactions', fontsize=fs)
ax[0].set_ylabel('Group normed count', fontsize=fs)

plt.tick_params(labelsize=fs)
df1.Tip_percentage.hist(bins = 20,normed=True,ax=ax[1])
ax[1].set_xlabel('Tip (%)', fontsize=fs)
ax[1].set_title('Distribution of Tip (%) - Transaction with tips', fontsize=fs)
ax[1].set_ylabel('Group normed count', fontsize=fs)
plt.savefig('Question4_target_varc.jpeg',format='jpeg')
plt.show()
"""


#visualize_continuous(df1,'Fare_amount',outlier='on')
#test_classification(df,'Fare_amount',[0,25])

"""
df = df[abs(df.Speed_mph-12.807543)< 3*6.562044]
df = df[df.Speed_mph > 0]
visualize_continuous(df1,'Speed_mph',outlier='on')
test_classification(df,'Speed_mph',[0,25])
"""

################################
# adding weather
################################
weather = pd.read_csv('weather_NY_JFK.csv',usecols=['DATE', 'PRCP', 'DATE'])
weather.DATE = weather.DATE.astype(str)
df['date_dt'] = df.Pickup_dt.apply(lambda x:x.date())
weather['date_dt'] = weather.DATE.apply(lambda x:dt.datetime.strptime(x,"%Y%m%d").date())
df = pd.merge(df, weather, how='left', on='date_dt')
# Create 'with_rain' variable
df['with_rain'] = (df.PRCP>0)*1
#df = df.drop('Extra',1)


################################
## OPTIMIZATION & TRAINING OF THE CLASSIFIER
################################

from sklearn.ensemble import GradientBoostingClassifier
print "Optimizing the classifier..."

train = df.copy() # make a copy of the training set
# since the dataset is too big for my system, select a small sample size to carry on training and 5 folds cross validation
train = train.loc[np.random.choice(train.index,size=10000,replace=False)]
target = 'With_tip' # set target variable - it will be used later in optimization

tic = dt.datetime.now() # initiate the timing
# for predictors start with candidates identified during the EDA
#predictors = ['Payment_type','Total_amount','Trip_duration','Speed_mph','MTA_tax',
#              'Extra','Hour','Direction_NS', 'Direction_EW','U_manhattan']
predictors = ['Payment_type','Total_amount','Trip_duration','Speed_mph','MTA_tax',
              'Extra','Hour','Direction_NS', 'Direction_EW','with_rain']

# optimize n_estimator through grid search
param_test = {'n_estimators':range(30,151,20)} # define range over which number of trees is to be optimized


# initiate classification model
model_cls = GradientBoostingClassifier(
    learning_rate=0.1, # use default
    min_samples_split=2,# use default
    max_depth=5,
    max_features='auto',
    subsample=0.8, # try <1 to decrease variance and increase bias
    random_state = 10)

# get results of the search grid
gs_cls = optimize_num_trees(model_cls,param_test,'roc_auc',train,predictors,target)
print gs_cls.grid_scores_, gs_cls.best_params_, gs_cls.best_score_

# cross validate the best model with optimized number of estimators
modelfit(gs_cls.best_estimator_,train,predictors,target,'roc_auc')

# save the best estimator on disk as pickle for a later use
with open('my_classifier.pkl','wb') as fid:
    pickle.dump(gs_cls.best_estimator_,fid)
    fid.close()
    
print "Processing time:", dt.datetime.now()-tic



