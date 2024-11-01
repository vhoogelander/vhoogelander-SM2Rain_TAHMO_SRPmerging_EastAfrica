import os
import sys
import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
import geopandas as gpd
import matplotlib.pyplot as plt
import ipywidgets as widgets
import pickle
import warnings
import sm2r_tahmo_algorithm
import extract_ascat_rzsm
import utils
# Ignore the specific RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning)

###################################### SETTINGS ##############################################
path_to_sm_data = 'data/soilmoisture_subsets/' #Path to ASCAT soil moisutre subsets   
path_to_tahmo = 'data/tahmo/'  # Path to TAHMO data
path_to_metadata = 'data/tahmo/metadata.csv' #Path to TAHMO metadata file

[MinLon,MaxLon,MinLat,MaxLat] =  [28.45,43.55,-5.05,5.45] # AOI

tol = 0.2 # Fraction of NaN data in TAHMO data (if nan_count/datapoints <tol, than data is included)
nclasses = 7 #Number of Rainfall classes
min_value = 0 # Min value in rainfall classification
max_value = 1800 # Max value in rainfall classification

sigma=5.0 #Smoothing parameter for parameter fields

ID = 'E' # ID of SM2Rain generation
num_stations_to_select = 'All'  # Number of stations to select in SM2Rain calibration 'All' #'All' or int

start_time = datetime(2018, 1, 1) #Start time of SM2Rain generation
end_time = datetime(2022, 12, 31) #End time of SM2Rain generation

start_cal = datetime(2022, 1, 1) #Start time of SM2Rain calibration
end_cal = datetime(2022, 12, 31) #End time of SM2Rain calibration

data_flags_included = True #True if data quality flags are used for removing dubious TAHMO data
path_to_qualityflags = 'data/quality_flags.csv'

timeid = f"{start_time.strftime('%Y%m%d')}_{end_time.strftime('%Y%m%d')}"

#################################################################################################################


years = list(range(start_time.year, end_time.year + 1)) 

datasets_path = path_to_sm_data+ "ascat_*_sm_*_R01_subset.nc"

sm_data_years = xr.open_mfdataset(datasets_path)
sm_data_years = sm_data_years['var40'].sel(time=slice(start_time, end_time))[:,:,:]
sm_data = sm_data_years
sm_data_values = sm_data_years.values 

new_var = xr.Variable(dims=sm_data_years.dims, data=sm_data_values)

# Add the new variable to the dataset
sm_data_years['var40'] = new_var 

sm_data_years['var40'].to_netcdf(f'smdata_{ID}.nc') # Generate temporally file


if data_flags_included:
    flags = pd.read_csv(path_to_qualityflags, index_col=0)  #Remove all flagged TAHMO data
    mask = flags.iloc[:,-5:] > 0
    
    stacked_df = mask.stack()
    flagged_data = stacked_df[stacked_df].index.tolist()
    
    flagged_data


list_stations = []
data_folder = path_to_tahmo

### Create list of all tahmo stations
for i in range(len(os.listdir(data_folder))):
    data_dir = os.path.join(data_folder, os.listdir(data_folder)[i])
    if os.path.basename(data_dir).startswith('TA'):
        station_name = os.path.basename(data_dir).split('.')[0]
        list_stations.append(station_name)

# Select the stations that are located within the subset
list_stations_subset = utils.SelectStationsSubset(list_stations, path_to_metadata, MinLat, MaxLat, MinLon, MaxLon) 


# #### Create dictionary with rainfall data per station
start_time = start_cal 
end_time = end_cal

data_folder = path_to_tahmo
prec_ts_stations = {}

## Create dictionary of rainfall data per station
for i in range(len(os.listdir(data_folder))):
    data_dir = os.path.join(data_folder, os.listdir(data_folder)[i])
    if os.path.basename(data_dir).startswith('TA'):
        station_name = os.path.basename(data_dir).split('.')[0]
        if station_name in list_stations_subset:
            station_data = pd.read_csv(data_dir, index_col=0, parse_dates=True)
            station_data = station_data.loc[(station_data.index <=end_cal) & (station_data.index >= start_cal)]
            nan_count = np.count_nonzero(np.isnan(station_data.precip.loc[(station_data.index >= start_time) & (station_data.index <= end_time)]))
            if (nan_count / len(station_data.precip.loc[(station_data.index >= start_time) & (station_data.index <= end_time)])) < tol:
                prec_ts_stations[station_name] = station_data.precip.loc[(station_data.index >= start_time) & (station_data.index <= end_time)]
                
if data_flags_included:
    for i in range(len(flagged_data)):
        if flagged_data[i][0] in prec_ts_stations.keys():
            prec_ts_stations[flagged_data[i][0]].loc[prec_ts_stations[flagged_data[i][0]].index.year == float(flagged_data[i][1])] = np.nan


# ### Create dictionary with SM data at stations

sm_ts_stations = utils.timeSeriesAllTahmoFromxArray(sm_data['var40'][:,:,:,0], path_to_metadata, list_stations_subset, 'var40') #SM timeseries per TAHMO station (0-7cm depth)


sm_ts_stations = {key: value for key, value in sm_ts_stations.items() if key in prec_ts_stations}
sm_ts_stations =  {key: value.loc[(value.index >= start_time) & (value.index <= end_time)] for key, value in sm_ts_stations.items()}


# # Create rainfall classes

lon_coords, lat_coords = sm_data.lon.values, sm_data.lat.values

rainfall_grid, rainclass_dic = utils.create_rainfall_class_grid(path_to_tahmo, path_to_metadata, lon_coords, lat_coords, list_stations_subset, min_value, max_value, nclasses)

rainclass_dic = {key: value for key, value in rainclass_dic.items() if key in prec_ts_stations} #Dictionary of Rainfall class, lon, and lat of every TAHMO statioin


# ### Select subset for calibration/validation
rainclass_dic_subset, unselected_stations = utils.select_subset_stations(rainclass_dic, num_stations_to_select)
len(rainclass_dic_subset)

# rainclass_dic_subset
file_path = f'data/selected_stations_{ID}.pkl' #File name

with open(file_path, 'wb') as file:
    pickle.dump(rainclass_dic_subset, file) #Export Pickle file with all selected stations for SM2Rain calibration


# # Calibration
#Calibrate the parameters per rainclass
param_dic = sm2r_tahmo_algorithm.RainfallClass_params(nclasses, rainclass_dic_subset, prec_ts_stations, sm_ts_stations, 400, x0=None, bounds=None, options=None, method='TNC')


# # SM2Rain

# ### Extrapolate SM2R parameters to grid

param_grid = sm2r_tahmo_algorithm.assign_sm2rparams_to_grid(lon_coords, lat_coords, path_to_metadata, param_dic_calibrated=param_dic, param_grid_rainclass=rainfall_grid, rainclass=True, sigma=sigma)


param_grid.to_netcdf(f'data/param_grid_{start_time.year}_{end_time.year}_{ID}.nc') #Export parameter grid


water_body_mask = np.isnan(sm_data_years['var40']) ## Create water body mask from ASCAT data
param_grid_regridded = param_grid.interp_like(water_body_mask)

for var_name in param_grid_regridded.variables:
    if var_name not in param_grid_regridded.coords:  # Exclude coordinate variables
        param_grid_regridded[var_name] = param_grid_regridded[var_name].where(~water_body_mask)


param_grid_regridded['smoothed_a'][:,:,0].plot(cmap='gnuplot')


# ### Generate precip timeseries

DS_subset_sm2r =  sm2r_tahmo_algorithm.ts_sm2rain_grid(param_grid, f'smdata_{ID}.nc' , smoothed=True)
sm_data.close()


# #### Remove soilmoisture dataset

os.remove(f'smdata_{ID}.nc')


# 
# #### Export results
DS_subset_sm2r['precip'].to_netcdf(f'data/ascat_sm2r_{timeid}_R01_rainclass_smoothed_{num_stations_to_select}_stations_{ID}.nc')



