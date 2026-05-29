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


# #### `get_satellite_predictors(X,var_sat)`
# 
# Retrieves satellite predictors (brightness temperature and cold pixel count) from TCFPv4 satellite predictor files. 
# 
# <b>Inputs</b>: 
# * `X`: DataArray containing satellite variables
# * `var_sat`: dictionary containing names of satellite variables we want to use as predictors
# 
# <b>Outputs</b>: 
# * `loadvar`: numpy array with desired satellite data, dimensions of `[n,lat,lon]`, where `n` is the number of variables we extract. 

# In[3]:


def get_satellite_predictors(X,var_sat):
    # Dictionary containing names of satellite predictors we're interested in
    sat_vars = {'tbright':'brightness_temperature_clear',
                'pct_below':'percent_pixel_below_threshold'}
    # Pre-allocate empty numpy array
    loadvar = np.empty([len(var_sat),len(X.variables['latitude'][:].data),
                           len(X.variables['longitude'][:].data)])
    # Loop through each variable, read in, and save to `loadvar`
    for i in np.arange(0,len(var_sat)):
        # print(var_sel)
        var_sel = var_sat[i]
        varname_sel = sat_vars[var_sel]
        ivar = X[varname_sel].squeeze()
        loadvar[i,:,:] = ivar
    return loadvar


# #### `get_sst_predictors(X,var_sst)`
# 
# Retrieves sst predictors (just sea surface temperature) from TCFPv4 sst predictor files. 
# 
# <b>Inputs</b>: 
# * `X`: DataArray containing sst variables
# * `var_sst`: dictionary containing names of sst variables we want to use as predictors
# 
# <b>Outputs</b>: 
# * `loadvar`: numpy array with desired sst data, dimensions of `[n,lat,lon]`, where `n` is the number of variables we extract. For SST there is only one variable so `n` should always just be 1. 

# In[4]:


def get_sst_predictors(X,var_sst):
    # Dictionary containing SST variable names
    sst_vars = {'sst':'sea_surface_temperature'}
    # Pre-allocate array
    loadvar = np.empty([len(sst_vars),len(X.variables['latitude'][:].data),
                        len(X.variables['longitude'][:].data)])
    # Loop through and save sst variables to `loadvar`
    for i in np.arange(0,len(var_sst)):
        var_sel = var_sst[i]
        varname_sel = sst_vars[var_sel]
        ivar = X[varname_sel].squeeze()
        loadvar[i,:,:] = ivar
    return loadvar


# #### `get_derived_predictors(X,var_derived)`
# 
# Retrieves derived predictors (shear, temperature anomaly, max height, vertical velocity) from TCFPv4 derived predictor files. Note that unlike the satellite and the sst predictors, the derived and model predictors are available at many different pressure levels, so those must also be specified. 
# 
# <b>Inputs</b>: 
# * `X`: DataArray containing derived variables
# * `var_derived`: dictionary containing names of derived variables we want to use as predictors
# 
# <b>Outputs</b>: 
# * `loadvar`: numpy array with desired derived data, dimensions of `[n,lat,lon]`, where `n` is the number of variables we extract. 

# In[5]:


def get_derived_predictors(X,var_derived):
    # Dictionary containing derived variables we are interested in
    derived_vars = {'shrg':'shear_generalized',
                'tanom':'temperature_anomaly',
                'hmax':'height_max',
                'wvel':'vertical_velocity_average'}
    # Dictionary containing the corresponding height levels for the derived variables. At this point, if we select multiple 
    # pressure levels we will average over them. 
    derived_var_levels = {'shrg':[],
               'tanom':[300,400],
               'hmax':[],
               'wvel':[]}
    # Pre-allocate variable
    loadvar = np.empty([len(var_derived),len(X.variables['latitude'][:].data),
                           len(X.variables['longitude'][:].data)])
    # Loop through each derived variable and get desired variables
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


# #### `get_model_predictors(X,var_model)`
# 
# Retrieves model predictors (RH, vorticity, divergence) from TCFPv4 model predictor files. Note that unlike the satellite and the sst predictors, the derived and model predictors are available at many different pressure levels, so those must also be specified. For the model  variables, we also specifiy the radius (from the center) over which we want to avearge our model variables. Note that in TCFPv4 parlance, the radius dimension is called "region".  
# 
# <b>Inputs</b>: 
# * `X`: DataArray containing model variables
# * `var_model`: dictionary containing names of model variables we want to use as predictors
# 
# <b>Outputs</b>: 
# * `loadvar`: numpy array with desired model data, dimensions of `[n,lat,lon]`, where `n` is the number of variables we extract. 

# In[6]:


def get_model_predictors(X,var_model):
    # Dictionary containing model variables we are interested in
    model_vars = {'RHmd':'relative_humidity',
              'vortl':'vorticity',
              'divl':'divergence',
              'divu':'divergence'}
    # Dictionary containing desired pressure levels to average each model variable over
    model_var_levels = {'RHmd':[500,700],
                    'vortl':[850],
                    'divl':[850],
                    'divu':[200]}
    # Dictionary containing desired radius ("region") to average each model variable over
    model_var_rad = {'RHmd':[0,1000],
                 'vortl':[0,1000],
                 'divl':[0,1000],
                 'divu':[0,1000]}
    # Pre-allocate array
    loadvar = np.empty([len(var_model),len(X.variables['latitude'][:].data),
                           len(X.variables['longitude'][:].data)])
    # Reach in each variable, average over both pressure level and radius where appropriate, and save
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


# #### `get_ground_truth(forecast_date)`
# 
# Gets the ground truth TCFPv4 forcast for the specified forecast date.   
# 
# <b>Inputs</b>: 
# * `forecast_date`: datetime object containing the desired forecast date (`YYYY-MM-DD HH:MM:SS`)
# 
# <b>Outputs</b>: 
# * `atcfids`: list of `atcfid`s for each storm that has an active TCFP forecast for `forecast_date`
# * `ground_truth`: `storm_classifier` object that identifies any existing storms for forecast date. This object is a classifier, but rather than just being a binary classifier, each different storm object has a different number (e.g., 1, 2, 3, 4)

# In[7]:


def get_ground_truth(forecast_date):
    # Load TCFPv4 verification data
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
    # Get atcfids and labels
    xdata = xr.open_mfdataset(fpath+fname)
    atcfids = xdata.atcfid
    ground_truth = xdata.storm_classifier
    return atcfids,ground_truth


# In[8]:


# Dictionary containing abbreviated names for each type of TCFP input data
datacat_dict = {'satellite':'sat',
                'storm':'storm',
                'derived':'deriv',
                'model':'model',
                'sst':'sst'}
# Dictionary containing the predictors we are interested in from each input data category
datacat_vars = {'satellite':['tbright','pct_below'],
                'derived':['shrg','tanom','hmax','wvel'],
                'model':['RHmd','vortl','divl','divu'],
                'sst':['sst']}
# Array containing all the data categories we want to include
data_cat = ['model','derived','satellite','sst']


# In[29]:


start_date = '2000-01-01'
end_date = '2020-12-31'


# In[30]:


days_back = 5 # how long are our forecasts? 
forecast_date = pd.date_range(start=start_date,end=end_date,freq='1D')


# For each `forecast_date`:
# 1. Identify the date corresponding to `days_back` days back and get correctly formatted date string
# 2. Load ground truth corresonding to `forecast_date`
# 3. Check to make sure we have data files corresponding to `forecast_date` for all of our different data categories (e.g., derived, model, satellite, sst). If not, skip to next date
# 4. Load desired predictors for each data category

# In[31]:


for iforecast in forecast_date:
    data_dates = iforecast - pd.Timedelta(days_back,'D')
    if np.mod(iforecast.day,7) == 0:
        print('forecasting for ',iforecast,'; data from ',data_dates)


# In[32]:


for iforecast in forecast_date:
    data_dates = iforecast - pd.Timedelta(days_back,'D')
    if np.mod(iforecast.day,7) == 0:
        print('forecasting for ',iforecast,'; data from ',data_dates)
    #
    yr_sel = data_dates.year
    mon_sel = data_dates.month
    day_sel = data_dates.day
    hr_sel = data_dates.hour
    # Add 0s in front of single digits where needed
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
    ## If we are missing any predictors for a forecast time, skip all
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
        # Model
        if idata == 'model':
            X_model = get_model_predictors(xdata,var_ALL)
        # Derived
        elif idata == 'derived':
            X_model = get_derived_predictors(xdata,var_ALL)
        # Satellite
        elif idata == 'satellite':
            X_model = get_satellite_predictors(xdata,var_ALL)
        # SST
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
        ifname_train_save = '/mnt/mlnas01/mmcgraw/tcfp/data/{nday}day_full_predictors_{start}-{end}_{n}D_freq_SIMPLE_PRED'.format(nday=days_back,
                                                            start=start_date,end=end_date,n=1)
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
        data_xr.to_netcdf(ifname_train_save+'_FEATURES.nc')
        data_yr.to_netcdf(ifname_train_save+'_TARGETS.nc')


# In[34]:


ifname_train_save = '/mnt/mlnas01/mmcgraw/tcfp/data/{nday}day_full_predictors_{start}-{end}_{n}D_freq_SIMPLE_PRED'.format(nday=days_back,
                                                            start=start_date,end=end_date,n=1)
X_train = full_predictors_ALL.copy()
y_train = ground_truths_ALL.copy()
dtv = forecast_date.where(forecast_date<=iforecast).dropna(how='all')
#dtv = dtv[0:-1]
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


# In[35]:


data_xr.to_netcdf(ifname_train_save+'_FEATURES.nc')


# In[36]:


data_yr.to_netcdf(ifname_train_save+'_TARGETS.nc')


# In[42]:





# In[ ]:




