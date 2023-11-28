#!/usr/bin/env python
# coding: utf-8

# # Prediction of total COVID-19 infections in December 2023

# #### 1. Nazwa zespołu:
# <br> The Overfitting Outliers
# #### 2. Członkowie zespołu – imiona, nazwiska i numery indeksu:
# <br> Jaźwiński Michał - 124246
# <br> Kozłowska Emilia - 123036
# <br> Warpas Martyna - 124290

# ### Packages

# In[345]:


import re
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from datetime import datetime
import lightgbm

from sklearn.metrics import mean_squared_error


# ### Functions used in further analysis

# In[346]:


def features(df):
    
    df_with_feat = df.copy()
    df_with_feat['day_of_week'] = df_with_feat.index.dayofweek
    df_with_feat['quarter'] = df_with_feat.index.quarter
    df_with_feat['month'] = df_with_feat.index.month
    df_with_feat['year'] = df_with_feat.index.year
    df_with_feat['day_of_year'] = df_with_feat.index.dayofyear
    
    return df_with_feat


# ### Data preparation

# In[347]:


# 1. Unzipping file

import zipfile
with zipfile.ZipFile('danehistorycznewojewodztwa.zip', 'r') as zip_ref:
    zip_ref.extractall()


# In[348]:


# 2. Scanning folder with data

import os

def scan_folder(parent):
    
    files_list = []
    # iterate over all the files in directory 'parent'
    for file_name in os.listdir(parent):
        if file_name.endswith(".csv"):
            # if it's a txt file, print its name (or do whatever you want)
            files_list.append(file_name)
        else:
            current_path = "".join((parent, "/", file_name))
            if os.path.isdir(current_path):
                # if we're checking a sub-directory, recursively call this method
                scan_folder(current_path)
                
    return files_list


files_list = scan_folder("C:/Users/lenovo/Desktop/MIESI/III semestr/DS_Python")  # Insert parent directory's path


# merging these files into one dataframe

merged_df = pd.DataFrame()

for f in files_list:
        
    day_df = pd.read_csv(f, sep=';', encoding='latin')
    day_df = pd.DataFrame(day_df.head(1))
    day_df['data'] = f[:8]
                
    merged_df = pd.concat([merged_df, day_df])


# In[349]:


merged_df[['stan_rekordu_na', 'data']]


# In[350]:


# 3. Preprocessing dataframe with basic data

merged_df = merged_df[['data'
                       , 'liczba_przypadkow'
                       , 'liczba_wszystkich_zakazen']]

merged_df = merged_df.fillna(value=0)
merged_df['liczba_zakazen_total'] = merged_df['liczba_przypadkow'] + merged_df['liczba_wszystkich_zakazen']
merged_df = merged_df[['data', 'liczba_zakazen_total']]

y = merged_df.set_index('data')
y.index = pd.to_datetime(y.index)


# In[351]:


y


# ### Data exploration

# In[352]:


y.plot()


# In[353]:


expl_df = features(y)


# In[354]:


expl_df


# In[355]:


expl_df.groupby(by=['year', 'month']).agg({'liczba_zakazen_total' : 'sum'})


# ### Train/test split

# In[356]:


# dataframes with one column - target
train = y.loc[y.index <= datetime(2023, 11, 15)]
test = y.loc[y.index > datetime(2023, 11, 15)]


# In[357]:


train


# ### Creating some features

# In[358]:


# dataframes with target and features
train_df = features(train)
test_df = features(test)


# In[359]:


train_df


# ### Building model

# In[360]:


# Lists of variables names
X_names = ['day_of_week', 'quarter', 'month', 'year', 'day_of_year']
y_name = ['liczba_zakazen_total']


# In[361]:


# Train and test sets - features and target column
X_train = train_df[X_names]
y_train = train_df[y_name]

X_test = test_df[X_names]
y_test = test_df[y_name]


# In[362]:


# Looking for best parameters

import numpy as np
from tqdm import tqdm

# Objective function to minimalize mse
def objective(params):
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse

# Parameters space
param_space = {
    'max_depth': range(1, 10),
    'learning_rate': [0.001, 0.01, 0.1, 0.5, 1.0],
    'gamma': [0, 0.1, 0.2, 0.3],
    'min_child_weight' : range(5, 50, 5),
    'subsample': [0.5, 0.7, 0.9, 1.0],
    'colsample_bytree': [0.5, 0.7, 0.9, 1.0],
    'reg_alpha': range(1, 30, 1),
    'reg_lambda': range(1, 30, 1),
    'min_split_loss': range(1, 10)
}

# Number of iterations Random Search
num_iterations = 200

# Initialization of best parameters and best mse
best_params = None
best_mse = np.inf

# Random Search Loop
for iteration in tqdm(range(num_iterations)):
    params = {param: np.random.choice(values) for param, values in param_space.items()}
    mse = objective(params)

    if mse < best_mse:
        best_mse = mse
        best_params = params

# Best parameters for model:
print("Najlepsze parametry:", best_params)
print("Najlepszy logloss:", best_loss)


# In[363]:


# v1: building model after parameteres optimization

reg_opt = xgb.XGBRegressor(max_depth=5
                           , learning_rate=0.5
                           , gamma=0.1
                           , min_child_weight=10
                           , subsample=0.7
                           , colsample_bytree=0.9
                           , reg_alpha=13
                           , reg_lambda=29
                           , min_split_loss=2)

reg_opt.fit(X_train
            , y_train
            , eval_set=[(X_train, y_train), (X_test, y_test)])


# In[364]:


# v2: building model with different parameters
reg = xgb.XGBRegressor(n_estimators=1000
                       , booster='gbtree'
                       , max_depth=15
                       , min_child_weight=0.5
                       , learning_rate=0.01
                       , early_stopping_rounds=50)

reg.fit(X_train
        , y_train
        , eval_set=[(X_train, y_train), (X_test, y_test)])


# ### Feature Importance

# In[365]:


fi = pd.DataFrame(data=reg.feature_importances_
                 , index=reg.feature_names_in_
                 , columns=['fi'])


# In[366]:


fi.sort_values(by='fi').plot(kind='barh')
plt.show()


# ### Prediction on train and test sets

# In[367]:


# Prediction on train set
train_df['prediction'] = reg.predict(X_train)


# In[368]:


ax = train_df[['liczba_zakazen_total']].plot(figsize=(15, 5))
train_df['prediction'].plot(ax=ax, style='.')
plt.legend(['Truth Data', 'Predictions'])
ax.set_title('Raw Dat and Prediction')
plt.show()


# In[369]:


# Prediction on test set
test_df['prediction'] = reg.predict(X_test)


# In[370]:


ax = test_df[['liczba_zakazen_total']].plot(figsize=(15, 5))
test_df['prediction'].plot(ax=ax, style='.')
plt.legend(['Truth Data', 'Predictions'])
ax.set_title('Raw Dat and Prediction')
plt.show()


# ### Prediction on December 2023

# In[371]:


dec_date_df = pd.DataFrame({'data' : pd.date_range(start='20231201', end='20231231')})
dec_date_df = dec_date_df.set_index('data')
dec_feat_df = features(dec_date_df)


# In[372]:


dec_feat_df['prediction'] = reg.predict(dec_feat_df)


# In[373]:


dec_feat_df


# In[374]:


dec_feat_df['prediction'].plot(style='-')
plt.legend(['Truth Data', 'Predictions'])
plt.show()


# ### Sum of predictions in December 2023:

# In[375]:


dec_feat_df['prediction'].sum().round()

