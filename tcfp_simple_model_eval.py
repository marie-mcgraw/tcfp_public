#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy as np
import os, glob
import matplotlib.pyplot as plt
from netCDF4 import Dataset, MFDataset
from datetime import datetime, timedelta
import pandas as pd
import xarray as xr
import cftime
from sklearn.linear_model import LinearRegression
import warnings
from sklearn.metrics import r2_score
import time
import dask
from distributed import Client


# In[4]:


warnings.simplefilter("ignore")


# In[5]:


foreday = 5
start_date = '2000-01-01'
end_date = '2012-01-05'
end_date2 = '2011-01-05'
end_date3 = '2010-01-05'
forecast_date = pd.date_range(start=start_date,end=end_date3,freq='6H')


# In[6]:


X_feat = xr.open_dataset('features_{n}day_full_predictors_{start}-{end}_SIMPLE_PRED.nc'.format(n=foreday,start=start_date,
                                                                  end=end_date),chunks={'latitude':1,'longitude':10,'feature':1})


# In[7]:


X_target = xr.open_dataset('targets_{n}day_full_predictors_{start}-{end}_SIMPLE_PRED.nc'.format(n=foreday,start=start_date,
                                                                end=end_date2),chunks={'latitude':1,'longitude':10})


# In[8]:


X_feat = X_feat.sel(forecast_date=forecast_date)
X_target = X_target.sel(forecast_date=forecast_date)


# In[9]:


ndates = len(X_feat.forecast_date)
year_test = [2002,2008]
dates = X_feat.forecast_date.values


# In[10]:


X_train = X_feat.x.sel(forecast_date=~X_feat.forecast_date.dt.year.isin([year_test]))
y_train = X_target.y.sel(forecast_date=~X_target.forecast_date.dt.year.isin([year_test]))
#
X_test = X_feat.x.sel(forecast_date=X_feat.forecast_date.dt.year.isin([year_test]))
y_test = X_target.y.sel(forecast_date=X_target.forecast_date.dt.year.isin([year_test]))


# In[11]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold,GridSearchCV,cross_val_score
import joblib
from joblib import parallel_backend


# In[13]:


targets = y_train.stack(sp=("latitude","longitude"))
print(targets.shape)


# In[14]:


features = X_train.stack(sp=("feature","latitude","longitude"))
print(features.shape)


# In[ ]:


kfolds = 10
n_repeats = 3
pipe = Pipeline([('scaler',StandardScaler()),('clf',RandomForestClassifier(class_weight='balanced'))])
params_RF = {'clf__max_features':[3,4],
            'clf__n_estimators':[50,100],
            'clf__max_depth':[5,6,8],
            'clf__min_samples_leaf':[10,20]}
#cv = RepeatedStratifiedKFold(n_splits=kfolds,n_repeats=n_repeats)
grid = GridSearchCV(pipe,param_grid=params_RF,n_jobs=30,scoring='f1_weighted')


# In[ ]:


#c = Client(n_workers=os.cpu_count()-2, threads_per_worker=1)
with parallel_backend('threading',n_jobs=10):
    # model = RandomForestRegressor(n_jobs=10,n_estimators=10)
    model = grid
    start_time = time.time()
    print('start time ',start_time)
    model.fit(features,targets)
    end_time = time.time()

print(end_time - start_time)


# In[17]:


joblib.dump(model, "rf_model_{nday}day_class_{start}-{end}_SIMPLE_PRED.pkl".format(nday=foreday,
                                                start=start_date,end=end_date3) )

print('saved ','rf_model_{nday}day_class_{start}-{end}_SIMPLE_PRED.pkl'.format(nday=foreday,
                                                start=start_date,end=end_date3))
# In[18]:





# In[ ]:




