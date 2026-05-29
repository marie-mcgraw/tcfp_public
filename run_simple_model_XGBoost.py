#!/usr/bin/env python
# coding: utf-8

# In[1]:


#! pip install xgboost
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
import random
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold,GridSearchCV,cross_val_score
import joblib
#! pip install imblearn
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline


# In[2]:


start_date = '2000-01-01'
end_date = '2020-12-31'
days_back = 5 # how long are our forecasts? 
ifname_train_save = '/mnt/mlnas01/mmcgraw/tcfp/data/{nday}day_full_predictors_{start}-{end}_{n}D_freq_SIMPLE_PRED'.format(nday=days_back,
                                                            start=start_date,end=end_date,n=1)


# In[3]:


X_read = xr.open_dataset(ifname_train_save+'_FEATURES.nc')
years_all = np.unique(X_read.forecast_date.dt.year.values)
y_read = xr.open_dataset(ifname_train_save+'_TARGETS.nc')
y_read = y_read.transpose("latitude","longitude","forecast_date")


# In[ ]:


## Set train and test years
years_test = [2010,2011,2012]
years_train = [yr for yr in years_all if yr not in years_test]
# years_train = [2000,2001,2002,2003]
# years_test = np.arange(2010,2013)
## years_train = [np.arange(2000,2010),
# Trim data to train and test data
X_train = X_read.sel(forecast_date = X_read.forecast_date.dt.year.isin(years_train)).x.values
X_test = X_read.sel(forecast_date = X_read.forecast_date.dt.year.isin(years_test)).x.values
#
y_train = y_read.sel(forecast_date = y_read.forecast_date.dt.year.isin(years_train)).y.values
y_test = y_read.isel(forecast_date = y_read.forecast_date.dt.year.isin(years_test)).y.values
# For now, keep this a simple classification problem
y_train[y_train > 0] = 1
y_test[y_test > 0] = 1


# In[ ]:


X_train = X_train[:,::2,::2,:]
X_test = X_test[:,::2,::2,:]
y_train = y_train[::2,::2,:]
y_test = y_test[::2,::2,:]


# In[ ]:


#features = X_train.reshape(X_train.shape[0]*X_train.shape[1]*X_train.shape[2],X_train.shape[3])
#print(features.transpose().shape)
#
#labels = y_train.reshape(y_train.shape[0],y_train.shape[1]*y_train.shape[2])
#print(labels.shape)


# In[ ]:


features = X_train.reshape(X_train.shape[0],X_train.shape[1]*X_train.shape[2]*X_train.shape[3]).transpose()
print(features.shape)
labels = y_train.reshape(y_train.shape[0]*y_train.shape[1]*y_train.shape[2]).transpose()
print(labels.shape)


# In[ ]:


classif = XGBClassifier()
pipeline = make_pipeline(SMOTE(sampling_strategy = 0.2), StandardScaler(), classif)
#
kfolds = 5
n_repeats = 3
params_xgb = {
            'xgbclassifier__n_estimators':[50,200],
            'xgbclassifier__max_depth':[5,7],
            'xgbclassifier__min_child_weight':[30,50],
            'xgbclassifier__learning_rate':[0.001,0.01],
            'xgbclassifier__subsample':[0.5,0.7],
            'xgbclassifier__booster':["gbtree"],
            'xgbclassifier__eval_metric':["logloss"]}
cv = RepeatedStratifiedKFold(n_splits=kfolds,n_repeats=n_repeats)
grid = GridSearchCV(pipeline,param_grid=params_xgb,cv=cv,n_jobs=70,scoring='f1_weighted')


# In[ ]:


memmap_path = "/mnt/mlnas01/mmcgraw/output_memmap_xgboost"
joblib.dump(np.asarray(features, dtype=np.float32, order='F'), memmap_path)


# In[ ]:


features_mmap = joblib.load(memmap_path,mmap_mode='c')


# In[ ]:





# In[ ]:


from joblib import parallel_backend

with parallel_backend('threading', n_jobs=70):
    #Pipe = ['scaler',Stand
    #model = RandomForestClassifier(n_jobs=6,n_estimators=10,max_features=3,max_depth=5,min_samples_leaf=10)
    model = grid
    start_time = time.time()
    #print('start time ',start_time)
    model.fit(features_mmap,labels)
    end_time = time.time()

print(end_time - start_time)


# In[ ]:





# In[ ]:


filename = 'saved_models/XGB_SMOTE_{nday}day_full_predictors_{start}-{end}_{n}D_freq_test_{yr1}-{yr2}.joblib'.format(nday=days_back,
                        start=start_date,end=end_date,n=1,yr1=np.min(years_test),yr2=np.max(years_test))
joblib.dump(model, filename)



# In[ ]:





# In[ ]:


np.min(years_test)


# In[ ]:


features_test = X_test.reshape(X_test.shape[0],X_test.shape[1]*X_test.shape[2]*X_test.shape[3]).transpose()
print(features.shape)
labels_test = y_test.reshape(y_test.shape[0]*y_test.shape[1]*y_test.shape[2]).transpose()


# In[ ]:


y_predict = model.predict(features_test)


# In[ ]:


y_predict_prob = model.predict_proba(features_test)


# In[ ]:


from sklearn.metrics import r2_score, accuracy_score
#foo = r2_score(labels_test,y_predict)
acc = accuracy_score(labels_test,y_predict)
acc


# In[ ]:


threshold = 0.1
predicted = y_predict_prob


# In[ ]:


predicted[:,0] = (predicted[:,0] < threshold)#.astype('int')
predicted[:,1] = (predicted[:,1] >= threshold)#.astype('int')

accuracy1 = accuracy_score(labels_test, predicted[:,1])
accuracy0 = accuracy_score(labels_test,predicted[:,0])


# In[ ]:


print('accuracy for genesis ',accuracy1)
print('accuracy for no genesis ',accuracy0)


# In[ ]:


from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(labels_test,y_predict)


# In[ ]:


cm


# In[ ]:


labels = model.classes_
cm_stats = pd.DataFrame(index=[0,1],columns=['Category','Category Names','N_predicted','N_actual','Hits','False Alarms',
                                                         'Misses','Correct Negs','POD','FAR','PFOD','SR','Threat'])
label_names = ['no genesis','genesis']
for i in [0,1]:
    # 
    cm_stats.loc[i,'Category Names'] = label_names[i]
    cm_stats.loc[i,'Category'] = i
    cm_stats.loc[i,'N_predicted'] = cm.sum(axis=0)[i]
    cm_stats.loc[i,'N_actual'] = cm.sum(axis=1)[i]
    cm_stats.loc[i,'Hits'] = np.diag(cm)[i]
    cm_stats.loc[i,'False Alarms'] = np.sum(cm[:,i]) - cm[i,i]
    cm_stats.loc[i,'Misses'] = np.sum(cm[i,:]) - cm[i,i]
    cm_negs = np.delete(cm,i,0)
    cm_negs = np.delete(cm_negs,i,1)
    cm_stats.loc[i,'Correct Negs'] = np.sum(cm_negs)
    cm_stats.loc[i,'POD'] = cm_stats.loc[i,'Hits']/(cm_stats.loc[i,'Hits'] + cm_stats.loc[i,'Misses'])
    cm_stats.loc[i,'FAR'] = cm_stats.loc[i,'False Alarms']/(cm_stats.loc[i,'Hits'] + cm_stats.loc[i,'False Alarms'])
    cm_stats.loc[i,'PFOD'] = cm_stats.loc[i,'False Alarms']/(cm_stats.loc[i,'Correct Negs'] + cm_stats.loc[i,'Misses'])
    cm_stats.loc[i,'Threat'] = cm_stats.loc[i,'Hits']/(cm_stats.loc[i,'Hits'] + cm_stats.loc[i,'Misses'] + cm_stats.loc[i,'False Alarms'])
    cm_stats.loc[i,'SR'] = cm_stats.loc[i,'Hits']/(cm_stats.loc[i,'Hits'] + cm_stats.loc[i,'False Alarms'])
    cm_stats.loc[i,'BIAS'] = (cm_stats.loc[i,'Hits'] + cm_stats.loc[i,'False Alarms'])/(cm_stats.loc[i,'Hits'] + cm_stats.loc[i,'Misses'])


# In[ ]:


cm_stats.to_csv('CM_XGB_SMOTE_{nday}day_full_predictors_{start}-{end}_{n}D_freq_test_{yr1}-{yr2}.joblib'.format(nday=days_back,
                        start=start_date,end=end_date,n=1,yr1=np.min(years_test),yr2=np.max(years_test)))


# In[ ]:




