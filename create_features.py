#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import time
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import os, glob
import seaborn as sns
from datetime import datetime, timedelta
import pandas as pd


# `get_dates_list(day0,dates_back)`:
# 
# Get a list of dates between the forecast date (`day0`) and `dates_back` (default is 1). Dates are given in `%Y%m%d%H` format to match TCFP formatting. 
# 
# <b>Inputs:</b>
# * `day0`: Target date [string, `%Y%m%d%H`]
# * `dates_back`: How many days back are we looking? [int, default is 1]
# 
# <b>Outputs:</b>
# * `list_dates`: List of dates (in `%Y%m%d%H` format) between `day0` and `dates_back` days earlier [list of strings]

# In[2]:


def get_dates_list(day0,dates_back=1):
    date1 = datetime.strptime(day0,'%Y%m%d%H')
    date2 = date1 - timedelta(days=dates_back)
    hrs_back = dates_back*24
    # print(hrs_back)
    # Get list of dates for every 6 hours
    list_dates = []
    for singledate in (date1 - timedelta(hours=n) for n in range(0,hrs_back+1,6)):
        list_dates.append(datetime.strftime(singledate,'%Y%m%d%H'))
    return list_dates   


# `load_feature_dicts()`:
# 
# Load the dictionaries that correspond to our desired features. We divided features into 4 categories: `model` (from NWP models), `derived` (derived quantities from NWP and cloud models), `sst` (from satellite), and `satellite` (from geostationary satellites). `model` and `derived` predictors can have dimensions of `[time, level, longitude, latitude, region]`, where `level` refers to vertical pressure levels, and `region` refers to distance from storm radius. `satellite` and `sst` predictors will only have dimensions of `[time, longitude, latitude]`. 
# 
# <b>Inputs:</b>
# * `None`
# 
# <b>Outputs:</b>
# * `model_vars`: long-form `model` variable names
# * `model_var_levels`: desired levels for each `model` variable
# * `model_var_rad`: desired regions for each `model` variable
# * `derived_vars`: long-form `derived` variable names
# * `derived_var_rad`: desired regions for each `derived` variable
# * `sst_vars`: long-form `sst` variable names
# * `satellite_vars`: long-form `satellite` variable names

# In[3]:


def load_feature_dicts():
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
    derived_vars = {'shrg':'shear_generalized',
                'tanom':'temperature_anomaly',
                'hmax':'height_max',
                'wvel':'vertical_velocity_average'}
    derived_var_levels = {'shrg':[],
                   'tanom':[300,400],
                   'hmax':[],
                   'wvel':[]}
#
    sst_vars = {'sst':'sea_surface_temperature'}
    satellite_vars = {'tb':'brightness_temperature',
                  'cpc':'percent_pixel_below_threshold'}
    return model_vars,model_var_levels,model_var_rad,derived_vars,derived_var_levels,sst_vars,satellite_vars


# In[4]:


def create_derived_vars(ncfname,matching_dates,target_date,variables,derived_vars,derived_var_levels,dateslist,
                     labels_list):
    ncfile_test_der = Dataset(ncfname,"w")
    latitude = ncfile_test_der.createDimension("latitude",len(Dataset(matching_dates[0]).variables['latitude']))
    longitude = ncfile_test_der.createDimension("longitude",len(Dataset(matching_dates[0]).variables['longitude']))
    time = ncfile_test_der.createDimension("time",None)
    lat = ncfile_test_der.createVariable("latitude","f4",("latitude",))
    lon = ncfile_test_der.createVariable("longitude","f4",("longitude",))
    lat.units = 'N'
    lon.units = 'E'
    time = ncfile_test_der.createVariable("time","f4",("time",))
    time.units = "hours since {date}".format(date=target_date)
    lat[:] = Dataset(matching_dates[0]).variables['latitude'][:].data
    lon[:] = Dataset(matching_dates[0]).variables['longitude'][:].data
    #
    for var_sel in variables:
        # print(var_sel)
        ncfile_test_der.createVariable("{var}".format(var=var_sel),"f8",("time","latitude","longitude",))
    #
    for var_sel in variables:
       # print(var_sel)
        varname_sel = derived_vars[var_sel]
        lev_sel = derived_var_levels[var_sel]
        # rad_sel = derived_var_rad[var_sel]
        # 
        loadvar = np.empty([len(dateslist),len(Dataset(matching_dates[0]).variables['latitude'][:].data),
                           len(Dataset(matching_dates[0]).variables['longitude'][:].data)])
        forecast_time = []
        for ii in np.arange(0,len(dateslist)):
        #idate = list_of_dates[10]
            idate = dateslist[ii]
            matching = [x for x in labels_list if idate in x]
            if not matching:
                continue
            # print(matching[0][ex.find("_s")+2:ex.find("_s")+12])
            time_in_hrs = (datetime.strptime(idate,'%Y%m%d%H') - date1).total_seconds()/(60*60)
            forecast_time.append(time_in_hrs)
            #
            if lev_sel:
            # rad = Dataset(matching[0]).variables['region']
            # rad_ind = np.isin(rad,np.arange(np.min(rad_sel),np.max(rad_sel)+1))
                levels = Dataset(matching[0]).variables['level']
                level_ind = np.isin(levels,np.arange(np.min(lev_sel),np.max(lev_sel)+1))
                vartest = Dataset(matching[0]).variables[varname_sel][:,level_ind,:,:]
                var_mean = np.nanmean(vartest,axis=1)
            elif not lev_sel:
                vartest = Dataset(matching[0]).variables[varname_sel]
                var_mean = np.nanmean(vartest,axis=3)
            # print('radius: ',rad_ind)
            # print('levels: ',level_ind)
            # print(vartest.shape)
            
            # 
            loadvar[ii,:,:] = var_mean
            #write_variable_to_netcdf(ncfile_test,var_sel,var_mean,ii)
            #time[ii] = time_in_hrs
        #
        time[:] = forecast_time
        ncfile_test_der.variables[var_sel][:] = loadvar
        # Description
    return ncfile_test_der


# In[5]:


def create_model_vars(ncfname,matching_dates,target_date,variables,model_vars,model_var_rad,model_var_levels,dateslist,
                     labels_list):
    ncfile_test = Dataset(ncfname,"w")
    latitude = ncfile_test.createDimension("latitude",len(Dataset(matching_dates[0]).variables['latitude']))
    longitude = ncfile_test.createDimension("longitude",len(Dataset(matching_dates[0]).variables['longitude']))
    time = ncfile_test.createDimension("time",None)
    lat = ncfile_test.createVariable("latitude","f4",("latitude",))
    lon = ncfile_test.createVariable("longitude","f4",("longitude",))
    lat.units = 'N'
    lon.units = 'E'
    time = ncfile_test.createVariable("time","f4",("time",))
    time.units = "hours since {date}".format(date=target_date)
    lat[:] = Dataset(matching_dates[0]).variables['latitude'][:].data
    lon[:] = Dataset(matching_dates[0]).variables['longitude'][:].data
    #
    for var_sel in variables:
        # print(var_sel)
        ncfile_test.createVariable("{var}".format(var=var_sel),"f8",("time","latitude","longitude",))
    #
    for var_sel in variables:
        # print(var_sel)
        varname_sel = model_vars[var_sel]
        lev_sel = model_var_levels[var_sel]
        rad_sel = model_var_rad[var_sel]
        # 
        loadvar = np.empty([len(dateslist),len(Dataset(matching_dates[0]).variables['latitude'][:].data),
                           len(Dataset(matching_dates[0]).variables['longitude'][:].data)])
        forecast_time = []
        for ii in np.arange(0,len(dateslist)):
        #idate = list_of_dates[10]
            idate = dateslist[ii]
            matching = [x for x in labels_list if idate in x]
            if not matching:
                continue
            # print(matching[0][ex.find("_s")+2:ex.find("_s")+12])
            time_in_hrs = (datetime.strptime(idate,'%Y%m%d%H') - date1).total_seconds()/(60*60)
            forecast_time.append(time_in_hrs)
            #
            rad = Dataset(matching[0]).variables['region']
            rad_ind = np.isin(rad,np.arange(np.min(rad_sel),np.max(rad_sel)+1))
            levels = Dataset(matching[0]).variables['level']
            level_ind = np.isin(levels,np.arange(np.min(lev_sel),np.max(lev_sel)+1))
            # print('radius: ',rad_ind)
            # print('levels: ',level_ind)
            vartest = Dataset(matching[0]).variables[varname_sel][:,level_ind,:,:,rad_ind]
            # print(vartest.shape)
            var_mean = np.nanmean(vartest,axis=(1,4))
            # 
            loadvar[ii,:,:] = var_mean
            #write_variable_to_netcdf(ncfile_test,var_sel,var_mean,ii)
            #time[ii] = time_in_hrs
        #
        time[:] = forecast_time
        ncfile_test.variables[var_sel][:] = loadvar
    return ncfile_test   


# In[6]:


model_vars,model_var_levels,model_var_rad,derived_vars,derived_var_levels,sst_vars,satellite_vars = load_feature_dicts()


# In[ ]:





# In[13]:


label_dir = ('/mnt/tcnas08/cslocum/TCFPv4/devdat/v1p0/')
yr_ALL = [2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023]
#year_sel = 2020
# yr_ALL = [2000]
nday = 5
features_5day = pd.DataFrame()
feature_sel_all = ['model','derived','sst','satellite']
feature_sel = 'model'
label_dir_full = label_dir+'{feature_sel}/1p00/'.format(feature_sel=feature_sel)


# ### OUTLINE
# 1. Load dictionaries for features
# 2. For each subset of features:
#     *  For each year:
#         *   Get list of files for each TCFP forecast for that year (should be 4x/day, 00Z, 06Z, 12Z, 18Z)
#         *   For each file/TCFP forecast:
#             *   Create netCDF file with dimensions `[forecast time x longitude x latitude]`
#             *   Get a list of files that correspond to the past `D` days (default is `D = 5`) [`get_dates_list`]
#             *   For each file in that list:
#                 *   Load file and extract desired features, averaging over `level` and `region`  dimensions as necessary
#                 *   Features should be in `[time x longitude x latitude]` format
#                 *   Write feature to netCDF file
#             *   Save netCDF file with all predictors that fall into desired forecast range
# 
# 
# End result should be for each TCFP forecast (1460 forecasts/yr), we have 4 netCDF files [`model`, `derived`, `sst`, `satellite`] that contain features going back `D` days for each forecast. 

# In[12]:


for yr_sel in yr_ALL:
    print('running files for ',yr_sel)
    ncfilepath = 'tcfp_predictors/marshalls_simple_model/model/{yr_sel}/'.format(yr_sel=yr_sel)
    if not os.path.exists(ncfilepath):
        os.makedirs(ncfilepath)
    fnames_labels = glob.glob(label_dir_full+'{yr}/*/*.nc'.format(yr=yr_sel))
    # genesis_pct = pd.DataFrame(columns=['Forecast Date','No. Positive','No. Total'],index=np.arange(0,len(fnames_labels)))
        #i = 21
    for i in np.arange(0,len(fnames_labels)):
        ex = fnames_labels[i]
        exdate = ex[ex.find("_s")+2:ex.find("_s")+12]
     #   print(exdate)
        date1 = datetime.strptime(exdate,'%Y%m%d%H')
        list_of_dates = get_dates_list(exdate,nday)
        #
        matching0 = [xx for xx in fnames_labels if list_of_dates[0] in xx]
        #
        # Variables
        #
        var_ALL = ['RHmd','vortl','divl','divu']
        
        ncfname = ncfilepath+'model_predictors_{exdate}_{nday}_forecasts.nc'.format(exdate=exdate,nday=nday)
        ncfile_model = create_model_vars(ncfname,matching0,exdate,var_ALL,model_vars,model_var_rad,model_var_levels,list_of_dates,
                         fnames_labels)
        ncfile_model.description = "Simple TCFP model, NWP variables from TCFPv4 dataset. Previous {nday} day features for forecast on {exdate}.".format(nday=nday,
                                                                                            exdate=exdate)
        ncfile_model.history = "Updated "+time.ctime(time.time())
        ncfile_model.close()
    # Derived
    ncfilepath_derived = 'tcfp_predictors/marshalls_simple_model/derived/{yr_sel}/'.format(yr_sel=yr_sel)
    label_dir_derived = label_dir+'{feature_sel}/1p00/'.format(feature_sel='derived')
    if not os.path.exists(ncfilepath_derived):
       os.makedirs(ncfilepath_derived)
    fnames_labels_derived = glob.glob(label_dir_derived+'{yr}/*/*.nc'.format(yr=yr_sel))
    #
    for j in np.arange(0,len(fnames_labels_derived)):
        exj = fnames_labels_derived[j]
        exdatej = exj[exj.find("_s")+2:exj.find("_s")+12]
        # print(exdatej)
        date1j = datetime.strptime(exdatej,'%Y%m%d%H')
        list_of_dates_j = get_dates_list(exdatej,nday)
        #
        matching0j = [xx for xx in fnames_labels_derived if list_of_dates_j[0] in xx]
        #
        # Variables
        #
        var_ALL_derived = ['shrg','tanom','hmax','wvel']
        #filename_derived = 
        ncfname_derived = ncfilepath_derived+'derived_predictors_{exdate}_{nday}_forecasts.nc'.format(exdate=exdatej,nday=nday)
        ncfile_derived = create_derived_vars(ncfname_derived,matching0j,exdatej,var_ALL_derived,derived_vars,derived_var_levels,list_of_dates_j,
                         fnames_labels_derived)
        ncfile_derived.description = "Simple TCFP model, dervied variables from TCFPv4 dataset. Previous {nday} day features for forecast on {exdate}.".format(nday=nday,
                                                                                            exdate=exdatej)
        ncfile_derived.history = "Updated "+time.ctime(time.time())
        ncfile_derived.close()


# Create feature set: for each TCFP forecast date, create a netCDF file of predictors with `[lon x lat x forecast time]`. 

# In[2]:





# In[ ]:




