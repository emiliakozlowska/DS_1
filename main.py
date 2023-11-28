import re
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from datetime import datetime

from sklearn.metrics import mean_squared_error

# ### Data preparation
# 1. Unzipping file

import zipfile
with zipfile.ZipFile('danehistorycznewojewodztwa.zip', 'r') as zip_ref:
    zip_ref.extractall()

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

merged_df[['stan_rekordu_na', 'data']]

# 3. Preprocessing dataframe with basic data

merged_df = merged_df[['data'
                       , 'liczba_przypadkow'
                       , 'liczba_wszystkich_zakazen']]

merged_df = merged_df.fillna(value=0)
merged_df['liczba_zakazen_total'] = merged_df['liczba_przypadkow'] + merged_df['liczba_wszystkich_zakazen']
merged_df = merged_df[['data', 'liczba_zakazen_total']]

y = merged_df.set_index('data')
y.index = pd.to_datetime(y.index)


# ### Data exploration

y.plot()

# ### Train/test split

train = y.loc[y.index <= datetime(2022, 12, 31)]
test = y.loc[y.index > datetime(2022, 12, 31)]

# ### Creating some features

def features(df):
    
    df_with_feat = df.copy()
    df_with_feat['day_of_week'] = df_with_feat.index.dayofweek
    df_with_feat['quarter'] = df_with_feat.index.quarter
    df_with_feat['month'] = df_with_feat.index.month
    df_with_feat['year'] = df_with_feat.index.year
    df_with_feat['day_of_year'] = df_with_feat.index.dayofyear
    
    return df_with_feat

train_df = features(train)
test_df = features(test)

train_df.columns

# ### Building model

X_names = ['day_of_week', 'quarter', 'month', 'year', 'day_of_year']
y_name = ['liczba_zakazen_total']

X_train = train_df[X_names]
y_train = train_df[y_name]

X_test = test_df[X_names]
y_test = test_df[y_name]

reg = xgb.XGBRegressor(n_estimators=1000
                       , early_stopping_rounds=50)

reg.fit(X_train, y_train
        , eval_set=[(X_train, y_train), (X_test, y_test)]
        , verbose=100)


# ### Feature Importance

fi = pd.DataFrame(data=reg.feature_importances_
                 , index=reg.feature_names_in_
                 , columns=['fi'])

fi.sort_values(by='fi').plot(kind='barh')
plt.show()

# ### Prediction
