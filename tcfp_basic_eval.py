#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold,GridSearchCV,cross_val_score
import joblib
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix, classification_report, f1_score


# In[2]:


start_date = '2000-01-01'
end_date = '2020-12-31'
days_back = 5 # how long are our forecasts? 


# In[3]:


ifname_train_save = '/mnt/mlnas01/mmcgraw/tcfp/data/{nday}day_full_predictors_{start}-{end}_{n}D_freq_SIMPLE_PRED'.format(nday=days_back,
                                                            start=start_date,end=end_date,n=1)
X_read = xr.open_dataset(ifname_train_save+'_FEATURES.nc')
years_all = np.unique(X_read.forecast_date.dt.year.values)
y_read = xr.open_dataset(ifname_train_save+'_TARGETS.nc')
y_read = y_read.transpose("latitude","longitude","forecast_date")


# In[4]:


## Set train and test years
years_test = [2010,2011,2012]
years_train = [yr for yr in years_all if yr not in years_test]
# years_test = np.arange(2010,2013)
## years_train = [np.arange(2000,2010),
# Trim data to train and test data
# X_train = X_read.sel(forecast_date = X_read.forecast_date.dt.year.isin(years_train)).x.values
X_test = X_read.sel(forecast_date = X_read.forecast_date.dt.year.isin(years_test))#.x.values
#
# y_train = y_read.sel(forecast_date = y_read.forecast_date.dt.year.isin(years_train)).y.values
y_test = y_read.sel(forecast_date = y_read.forecast_date.dt.year.isin(years_test))#.y.values
# For now, keep this a simple classification problem
#y_train[y_train > 0] = 1


# In[5]:


X_test = X_test.x.values
y_test = y_test.y.values
y_test[y_test > 0] = 1


# In[6]:


filename = 'saved_models/{nday}day_full_predictors_{start}-{end}_{n}D_freq_test_{yr1}-{yr2}.joblib'.format(nday=days_back,
                        start=start_date,end=end_date,n=1,yr1=np.min(years_test),yr2=years_test[-2])
model = joblib.load(filename)


# In[7]:


features_test = X_test.reshape(X_test.shape[0],X_test.shape[1]*X_test.shape[2]*X_test.shape[3]).transpose()
print(features_test.shape)
labels_test = y_test.reshape(y_test.shape[0]*y_test.shape[1]*y_test.shape[2]).transpose()


# In[8]:


y_predict = model.predict(features_test)
y_predict_prob = model.predict_proba(features_test)


# In[9]:


acc = accuracy_score(labels_test,y_predict)
acc


# In[29]:


threshold = 0.5
predicted = y_predict_prob
predicted[:,0] = (predicted[:,0] < threshold)#.astype('int')
predicted[:,1] = (predicted[:,1] >= threshold)#.astype('int')

accuracy1 = accuracy_score(labels_test, predicted[:,1])
accuracy0 = accuracy_score(labels_test,predicted[:,0])
print('accuracy for genesis ',accuracy1)
print('accuracy for no genesis ',accuracy0)


# In[ ]:





# In[30]:


cm = confusion_matrix(labels_test,y_predict)
cm


# In[31]:


labels = model.classes_
cm_stats = pd.DataFrame(index=[0,1],columns=['Category','Category Names','N_predicted','N_actual','Hits','False Alarms',
                                                         'Misses','Correct Negs','POD','FAR','POFD','SR','Threat'])
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
    cm_stats.loc[i,'POFD'] = cm_stats.loc[i,'False Alarms']/(cm_stats.loc[i,'Correct Negs'] + cm_stats.loc[i,'Misses'])
    cm_stats.loc[i,'Threat'] = cm_stats.loc[i,'Hits']/(cm_stats.loc[i,'Hits'] + cm_stats.loc[i,'Misses'] + cm_stats.loc[i,'False Alarms'])
    cm_stats.loc[i,'SR'] = cm_stats.loc[i,'Hits']/(cm_stats.loc[i,'Hits'] + cm_stats.loc[i,'False Alarms'])
    cm_stats.loc[i,'BIAS'] = (cm_stats.loc[i,'Hits'] + cm_stats.loc[i,'False Alarms'])/(cm_stats.loc[i,'Hits'] + cm_stats.loc[i,'Misses'])
    


# In[32]:


cm_stats.to_csv('inital_conf_matrix_stats_threshold_{thresh}pct.csv'.format(thresh=threshold*10))


# In[33]:


cm_stats
cm_stats.loc[1]['POD'] - cm_stats.loc[1]['POFD']


# In[34]:


cm_stats.loc[1]['POD']


# In[35]:


f1score = f1_score(labels_test,y_predict)
print('f1 score: ',f1score)


# In[46]:





# In[36]:


from sklearn.metrics import brier_score_loss


# In[37]:


bs_model = brier_score_loss(labels_test, y_predict_prob[:,1])
y_ref_true = labels_test
# Use climatology for reference
y_ref_clim = np.full(len(y_ref_true), np.mean(y_ref_true))
bs_clim = brier_score_loss(y_ref_true,y_ref_clim)
#
bss = 1 - (bs_model/bs_clim)
print('Brier skill score: ',bss)


# In[ ]:


from sklearn.inspection import permutation_importance
fi_pred = pd.DataFrame(index=X_read.feature.values,columns=['mean importance','std(importance)'])
r = permutation_importance(model,features_test,labels_test,scoring='f1_weighted',n_repeats=10)


# In[ ]:


for i in r.importances_mean.argsort()[::-1]:
    fi_pred.loc[features_test.columns[i],'mean importance'] = r.importances_mean[i]
    fi_pred.loc[features_test.columns[i],'std(importance)'] = r.importances_std[i]
    if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
        print(f"{features_test.columns[i]:<8}"
              f"{r.importances_mean[i]:.3f}"
              f" +/- {r.importances_std[i]:.3f}")


# In[ ]:


fi_pred.to_csv('inital_conf_matrix_stats.csv')

