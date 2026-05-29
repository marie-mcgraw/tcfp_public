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


# In[2]:


warnings.simplefilter("ignore")


# In[3]:


def get_satellite_predictors(X,var_sat):
    sat_vars = {'tbright':'brightness_temperature_clear',
                'pct_below':'percent_pixel_below_threshold'}
    #
    loadvar = np.empty([len(var_sat),len(X.variables['latitude'][:].data),
                           len(X.variables['longitude'][:].data)])
    #
    for i in np.arange(0,len(var_sat)):
        # print(var_sel)
        var_sel = var_sat[i]
        varname_sel = sat_vars[var_sel]
        ivar = X[varname_sel].squeeze()
        loadvar[i,:,:] = ivar
    return loadvar


# In[4]:


def get_sst_predictors(X,var_sst):
    sst_vars = {'sst':'sea_surface_temperature'}
    loadvar = np.empty([len(sst_vars),len(X.variables['latitude'][:].data),
                        len(X.variables['longitude'][:].data)])
    for i in np.arange(0,len(var_sst)):
        var_sel = var_sst[i]
        varname_sel = sst_vars[var_sel]
        ivar = X[varname_sel].squeeze()
        loadvar[i,:,:] = ivar
    return loadvar


# In[5]:


def get_derived_predictors(X,var_derived):
    derived_vars = {'shrg':'shear_generalized',
                'tanom':'temperature_anomaly',
                'hmax':'height_max',
                'wvel':'vertical_velocity_average'}
    derived_var_levels = {'shrg':[],
               'tanom':[300,400],
               'hmax':[],
               'wvel':[]}
    # 
    loadvar = np.empty([len(var_derived),len(X.variables['latitude'][:].data),
                           len(X.variables['longitude'][:].data)])
    #
    for i in np.arange(0,len(var_derived)):
        # print(var_sel)
        var_sel = var_derived[i]
        varname_sel = derived_vars[var_sel]
        lev_sel = derived_var_levels[var_sel]
        #print(varname_sel,lev_sel)
        # 
        # levels = boop.variables['level']
        if lev_sel:
            ivar = X[varname_sel].sel(level=lev_sel).mean(dim="level").squeeze()
        else:
            ivar = X[varname_sel].squeeze()
        loadvar[i,:,:] = ivar
    return loadvar


# In[6]:


def get_model_predictors(X,var_model):
    model_vars = {'RHmd':'relative_humidity',
              'vortl':'vorticity',
              'divl':'divergence',
              'divu':'divergence'}
    model_var_levels = {'RHmd':[500,700],
                    'vortl':[850],
                    'divl':[850],
                    'divu':[200]}
    model_var_rad = {'RHmd':[0,1000],
                 'vortl':[0,1000],
                 'divl':[0,1000],
                 'divu':[0,1000]}
    # 
    loadvar = np.empty([len(var_model),len(X.variables['latitude'][:].data),
                           len(X.variables['longitude'][:].data)])
    #
    for i in np.arange(0,len(var_model)):
        # print(var_sel)
        var_sel = var_model[i]
        varname_sel = model_vars[var_sel]
        lev_sel = model_var_levels[var_sel]
        reg_sel = model_var_rad[var_sel]
        #print(varname_sel,lev_sel)
        # 
        # levels = boop.variables['level']
        if lev_sel:
            if reg_sel:
                ivar = X[varname_sel].sel(level=lev_sel,region=reg_sel).mean(dim=["level","region"]).squeeze()
            else:
                ivar = X[varname_sel].sel(level=lev_sel).mean(dim="level").squeeze()
        else:
            if reg_sel:
                ivar = X[varname_sel].sel(region=reg_sel).mean(dim="region").squeeze()
            else:
                ivar = X[varname_sel].squeeze()
            
        loadvar[i,:,:] = ivar
    return loadvar


# In[7]:


def get_ground_truth(forecast_date):
    stormtest = '/mnt/tcnas08/cslocum/TCFPv4/devdat/v1p0/storm/1p00/'
    yr_sel = forecast_date.year
    mon_sel = forecast_date.month
    day_sel = forecast_date.day
    hr_sel = forecast_date.hour
    #
    mon_sel_str = str(mon_sel) if mon_sel > 9 else '0'+str(mon_sel)
    day_sel_str = str(day_sel) if day_sel > 9 else '0'+str(day_sel)
    hr_sel_str = str(hr_sel) if hr_sel > 9 else '0'+str(hr_sel)
    #
    fpath = stormtest+'{year}/{mon}{day}/'.format(year=yr_sel,
                                          mon = mon_sel_str,
                                          day = day_sel_str)
    fname = 'TCFP-devdat-verif-storm_v4r0_blend_s{year}{mon}{day}{hr}00000_e{year}{mon}{day}{hr}00000_*.nc'.format(year=yr_sel,
                                                            mon=mon_sel_str,day=day_sel_str,hr=hr_sel_str)
    #
    xdata = xr.open_mfdataset(fpath+fname)
    atcfids = xdata.atcfid
    ground_truth = xdata.storm_classifier
    return atcfids,ground_truth


# In[8]:


datacat_dict = {'satellite':'sat',
                'storm':'storm',
                'derived':'deriv',
                'model':'model',
                'sst':'sst'}
#
datacat_vars = {'satellite':['tbright','pct_below'],
                'derived':['shrg','tanom','hmax','wvel'],
                'model':['RHmd','vortl','divl','divu'],
                'sst':['sst']}
#
data_cat = ['model','derived','satellite','sst']


# In[9]:


start_date = '2000-01-01'
end_date = '2020-12-31'


# In[10]:


days_back = 5
forecast_date = pd.date_range(start=start_date,end=end_date,freq='6H')


# In[11]:


for iforecast in forecast_date:
    data_dates = iforecast - pd.Timedelta(days_back,'D')
    if np.mod(iforecast.day,7) == 0:
        print('forecasting for ',iforecast,'; data from ',data_dates)
    #
    yr_sel = data_dates.year
    mon_sel = data_dates.month
    day_sel = data_dates.day
    hr_sel = data_dates.hour
    #
    mon_sel_str = str(mon_sel) if mon_sel > 9 else '0'+str(mon_sel)
    day_sel_str = str(day_sel) if day_sel > 9 else '0'+str(day_sel)
    hr_sel_str = str(hr_sel) if hr_sel > 9 else '0'+str(hr_sel)
    # Get ground truth
    atcfids,ground_truth = get_ground_truth(iforecast)
    atcfid_df = atcfids.to_dataframe()
    atcfid_df['Date'] = iforecast
    if iforecast == forecast_date[0]:
        ground_truths_ALL = ground_truth.values
        atcfid_ALL = atcfid_df
    else:
        ground_truths_ALL = np.concatenate([ground_truths_ALL,ground_truth.values],axis=0)
        atcfid_ALL = pd.concat([atcfid_ALL,atcfid_df])
    ## If we are missing any predictors for a forecast time, skip all predictors for that time
    data_dir_allfiles = '/mnt/tcnas08/cslocum/TCFPv4/devdat/v1p0/*/1p00/'
    fpath_allfiles = data_dir_allfiles+'{year}/{mon}{day}/'.format(year=yr_sel,mon=mon_sel_str,day=day_sel_str)
    fname_allfiles = 'TCFP-devdat-pred-*_v4r0_blend_s{year}{mon}{day}{hr}00000_e{year}{mon}{day}{hr}00000_*.nc'.format(year=yr_sel,
                                                        mon=mon_sel_str,day=day_sel_str,hr=hr_sel_str)
    checkfiles = len(glob.glob(fpath_allfiles+fname_allfiles))
    if checkfiles < 4:
        print('missing some data')
        continue
    ## Now get predictors
    variables_list = []
    full_predictors = []
    for idata in data_cat:
        #print(idata)
        data_dir = '/mnt/tcnas08/cslocum/TCFPv4/devdat/v1p0/{datatype}/1p00/'.format(datatype=idata)
        fpath = data_dir+'{year}/{mon}{day}/'.format(year=yr_sel,mon = mon_sel_str,day = day_sel_str)
        #
        fname = 'TCFP-devdat-pred-{dcat}_v4r0_blend_s{year}{mon}{day}{hr}00000_e{year}{mon}{day}{hr}00000_*.nc'.format(dcat=datacat_dict[idata],
        year=yr_sel,mon=mon_sel_str,day=day_sel_str,hr=hr_sel_str)
        # 
        xdata = xr.open_mfdataset(fpath+fname)
        var_ALL = datacat_vars[idata]
        if idata == 'model':
            X_model = get_model_predictors(xdata,var_ALL)
        elif idata == 'derived':
            X_model = get_derived_predictors(xdata,var_ALL)
        elif idata == 'satellite':
            X_model = get_satellite_predictors(xdata,var_ALL)
        #
        elif idata == 'sst':
            X_model = get_sst_predictors(xdata,var_ALL)
        lat = xdata.variables['latitude'][:]
        lon = xdata.variables['longitude'][:]
        #
        variables_list = variables_list+var_ALL
        if idata == data_cat[0]:
            full_predictors = X_model
        else:
            full_predictors = np.append(full_predictors,X_model,axis=0)
    ##
    if iforecast == forecast_date[0]:
        full_predictors_ALL = full_predictors
    elif iforecast == forecast_date[1]:
        full_predictors_ALL = np.stack([full_predictors_ALL,full_predictors],axis=3)
    else:
        full_predictors_ALL = np.append(full_predictors_ALL,np.expand_dims(full_predictors,axis=3),axis=3)
    # Save predcitor files for every year
    if (mon_sel == 12) & (day_sel == 31) & (hr_sel == 18):
        print('saving predictors for now')
        ifname_train_save = '{nday}day_full_predictors_{start}-{end}_SIMPLE_PRED.nc'.format(nday=days_back,start=start_date,
                                                                                            end=iforecast.strftime('%Y-%m-%d'))
        X_train = full_predictors_ALL.copy()
        y_train = ground_truths_ALL.copy()
        dtv = forecast_date.where(forecast_date<=iforecast).dropna(how='all')
        data_xr = xr.DataArray(X_train,
                       coords={'feature':variables_list,
                               'latitude':lat,
                               'longitude':lon,
                               'forecast_date':dtv},
                       dims=["feature","latitude","longitude","forecast_date"],name='x')
        data_yr = xr.DataArray(y_train,
                       coords={'forecast_date':dtv,
                               'latitude':lat,
                               'longitude':lon},
                       dims=["forecast_date","latitude","longitude"],
                      name='y')
        data_xr.to_netcdf('features_'+ifname_train_save)
        data_yr.to_netcdf('targets_'+ifname_train_save)

# In[ ]:


ground_truths_ALL[ground_truths_ALL > 0] = 1


# In[ ]:


yrs_inc = pd.to_datetime(forecast_date).year.unique().tolist()
nyrs = len(yrs_inc)
ntest = int(np.ceil(0.2*nyrs))
yrs_test = random.sample(yrs_inc,ntest)
date_ser = forecast_date.to_series()
test_sel = forecast_date[pd.to_datetime(forecast_date).year.isin(yrs_test)]
inds = [forecast_date.get_loc(test_sel[i]) for i in np.arange(0,len(test_sel))]


# In[ ]:


inds2 = [10,5,35,59]


# In[ ]:


#print(full_predictors_ALL[:,:,:,:-int(ntest)].shape)
#X_train = full_predictors_ALL[:,:,:,~inds2]
#y_train = ground_truths_ALL[~inds2,:,:]
mask = np.zeros(full_predictors_ALL.shape, dtype=bool)
mask[:,:,:,inds2] = True
X_train = np.ma.masked_array(full_predictors_ALL,mask)
mask_y = np.zeros(ground_truths_ALL.shape, dtype=bool)
mask_y[inds2,:,:] = True
y_train = np.ma.masked_array(ground_truths_ALL,mask_y)


# In[ ]:


X_test = full_predictors_ALL[:,:,:,inds2]
y_test = ground_truths_ALL[inds2,:,:]
X_test.reshape(X_test.shape[0]*X_test.shape[1]*X_test.shape[2],X_test.shape[3]).shape
#X_train_rs.shape


# In[ ]:


fname_train_save = '{nday}day_training_sample_{start}-{end}_SIMPLE_PRED.nc'.format(nday=days_back,start=start_date,end=end_date)
data_xr = xr.DataArray(X_train,
                       coords={'feature':variables_list,
                               'latitude':lat,
                               'longitude':lon,
                               'forecast_date':forecast_date[:-int(ntest)]},
                       dims=["feature","latitude","longitude","forecast_date"],name='x')


# In[ ]:


data_yr = xr.DataArray(y_train,
                       coords={'forecast_date':forecast_date[:-int(ntest)],
                               'latitude':lat,
                               'longitude':lon},
                       dims=["forecast_date","latitude","longitude"],
                      name='y')
#
data_ytestr = xr.DataArray(y_test,
                       coords={'forecast_date':forecast_date[-int(ntest):],
                               'latitude':lat,
                               'longitude':lon},
                       dims=["forecast_date","latitude","longitude"])


# In[ ]:


fname_test_save = '{nday}day_forecasts_{start}-{end}_SIMPLE_PRED.nc'.format(nday=days_back,start=start_date,end=end_date)
data_test_xr = xr.DataArray(X_test,
                       coords={'feature':variables_list,
                               'latitude':lat,
                               'longitude':lon,
                               'forecast_date':forecast_date[-int(ntest):]},
                       dims=["feature","latitude","longitude","forecast_date"])
##
data_xr.to_netcdf('features_'+fname_train_save)
data_test_xr.to_netcdf('features_'+fname_test_save)
data_yr.to_netcdf('targets_'+fname_train_save)
data_ytestr.to_netcdf('targets_'+fname_test_save)
print('saved data')


# In[ ]:


X_train_calc = X_train.copy()
y_train_bin = y_train.copy()


# In[ ]:


features = X_train_calc.reshape(X_train_calc.shape[0]*X_train_calc.shape[1]*X_train_calc.shape[2],X_train_calc.shape[3])
print(features.transpose().shape)
#
labels = y_train_bin.reshape(y_train_bin.shape[0],y_train_bin.shape[1]*y_train_bin.shape[2])
print(labels.shape)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold,GridSearchCV,cross_val_score
import joblib


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


#labels.shape
#features.transpose().shape


# In[ ]:


from joblib import parallel_backend

with parallel_backend('threading', n_jobs=30):
    #Pipe = ['scaler',Stand
    #model = RandomForestClassifier(n_jobs=6,n_estimators=10,max_features=3,max_depth=5,min_samples_leaf=10)
    model = grid
    start_time = time.time()
    #print('start time ',start_time)
    model.fit(features.transpose(),labels)
    end_time = time.time()

print(end_time - start_time)


# In[ ]:


joblib.dump(model, "rf_model_{nday}day_class_{start}-{end}_SIMPLE_PRED.pkl".format(nday=days_back,start=start_date,end=end_date))