import pandas as pd
import numpy as np
import xarray as xr
import os
import scipy.stats as stats
from scipy.interpolate import griddata
from sklearn.metrics import confusion_matrix
import random

## select stations in subset
def SelectStationsSubset(StationsList, MetaFilePath, lat_min, lat_max, lon_min, lon_max):
    """
    Selects stations that are located within a given subset (rectangle).
    
    StationsList: list of strings
            List of TAHMO station ID's
    MetaFilePath: str
            path to TAHMO metadata file
    lat_min: float
            Minimum latitude
    lat_max: float
            Maximum latitude
    lon_min: float
            Minimum longitude
    lon_max: float
            Maximum longitude
            
    Returns list containing all station ID's that are within the subset.
    """
    
    meta = pd.read_csv(MetaFilePath, index_col=0)
    meta = meta[meta.index.isin(StationsList)]
    lon_station = meta['longitude']
    lat_station = meta['latitude']

    list_stations_subset = []
    for i in range(len(StationsList)):
        if (lon_min <= lon_station.loc[StationsList[i]] <= lon_max) & (lat_min <= lat_station.loc[StationsList[i]] <= lat_max):
            list_stations_subset.append(StationsList[i])
    return list_stations_subset

def timeSeriesAtLatLonFromNetCDF(netCDFFilePath,variable,lat,lon):
    """
    Get timeseries from a NetCDF at a given lat lon coordinate.
    
    netCDFFilePath: str
            Path to NetCDF file
    variable: str
            Variable of NetCDF
    lat: float
            Latitude
    lon: float
            Longitude
    Returns Pandas Series 
    """
    
    netCDFDataFrame = xr.open_dataset(netCDFFilePath)  # NetCDF or OPeNDAP URL
    netCDFDataAtLocation = netCDFDataFrame.sel(lon=lon, lat=lat, method='nearest')
    return netCDFDataAtLocation[variable].to_series()

def timeSeriesAllTahmoFromNetCDF(netCDFFilePath, MetaFilePath, StationsList, variable):
    """
    Get timeseries from a NetCDF at all TAHMO stations.

    netCDFFilePath: str
            Path to NetCDF file
    MetaFilePath: str
            path to TAHMO metadata file
    StationsList: list of strings
            List of TAHMO station ID's
    variable: str
            Variable in xarray dataset
    Returns dictionary of all timeseries with TAHMO station ID's as keys 
    """
    meta = pd.read_csv(MetaFilePath, index_col=0)
    meta = meta[meta.index.isin(StationsList)]
    lon = meta['longitude']
    lat = meta['latitude']
    dic_ts = {}
    for i in range(len(meta)):
        key = StationsList[i]
        lon_station, lat_station = lon.loc[key], lat.loc[key]
        dic_ts[key] = timeSeriesAtLatLonFromNetCDF(netCDFFilePath,variable,lat_station,lon_station)
    return dic_ts

def timeSeriesAllTahmoFromxArray(dataset, MetaFilePath, StationsList, variable):
    """
    Get timeseries from a xArray at all TAHMO stations.

    dataset: xarray.Dataset
            xarray dataset
    MetaFilePath: str
            path to TAHMO metadata file
    StationsList: list of strings
            List of TAHMO station ID's
    variable: str
            Variable in xarray dataset
    Returns dictionary of all timeseries with TAHMO station ID's as keys 
    """

    meta = pd.read_csv(MetaFilePath, index_col=0)
    meta = meta[meta.index.isin(StationsList)]
    lon = meta['longitude']
    lat = meta['latitude']
    dic_ts = {}
    for i in range(len(meta)):
        key = StationsList[i]
        lon_station, lat_station = lon.loc[key], lat.loc[key]
        # print(dataset, lo
        ts_station = dataset.sel(lon=lon_station, lat=lat_station, method='nearest')
        dic_ts[key] = pd.Series(data=ts_station.values, index=ts_station.time.values)  
    return dic_ts

# def remove_multiindex(dictionary):
#     for key, series in dictionary.items():
#         if isinstance(series, pd.Series):
#             dictionary[key] = series.reset_index(level=1, drop=True)
#     return dictionary

def GetEvaluationStatisticsPerStation(PathToTahmoStationData, PathToQFlags, ts_prec, StationsList, p_data, correlation='spearman', only_raindays=False, getMean=True):
    
    """
    Calculate the evaluation statistics: Root Mean Squared Error (RMSE), Bias, Kling Gupta Efficiency (KGE), Correlation , Probability of Detection (POD), False Alarm Ratio (FAR), and Heidke Skill Score (HSS)
    
    PathToTahmoStationData: str
            Path to folder containing daily data of TAHMO stations
    PathToQFlags: str
            Path to TAHMO quality flags file
    ts_prec: str
            Dictionary of simulated precipitation timeseries at all TAHMO stations. (Can be created using timeSeriesAllTahmoFromNetCDF(netCDFFilePath, MetaFilePath, StationsList, variable))
    StationsList: list of strings
            List of TAHMO station ID's
    p_data: float
            Minimum percentage of "true data" at station
    correlation: str
            Correlation method ('spearman' or 'pearson')
    only_raindays: False (Default)
            If True, Calculate the RMSE based on prec>0 for both TAHMO and satellite rainfall product
    getMean: True (Default)
            If True, returns of mean statistics of all TAHMO stations (rmse:float, spc:float, pod:float, far:float, hss:float). If False, returns dictionary of statistics at all TAHMO stations            
    """
        
    rmse_dic = {} #root mean squared error
    bias_dic = {}
    spc_dic = {} #Spearman correlation coefficient
    kge_dic = {} #Kling Gupta Efficiency
    pod_dic = {} #Probability of detction
    far_dic = {}
    hss_dic = {}
    
    flags = pd.read_csv(PathToQFlags, index_col=0)
    mask = flags.iloc[:,-5:] > 0

    stacked_df = mask.stack()
    false_data = stacked_df[stacked_df].index.tolist()
    
    ## Create dictionary with rainfall data
    tahmo_data = {}
    for i in range(len(os.listdir(PathToTahmoStationData))):
        data_dir = os.path.join(PathToTahmoStationData, os.listdir(PathToTahmoStationData)[i])
        if os.path.basename(data_dir).startswith('TA'):
            station_name = os.path.basename(data_dir).split('.')[0]
            # if station_name in station_subset_list:
            station_data = pd.read_csv(data_dir, index_col=0, parse_dates=True)
            if station_name in StationsList:
                tahmo_data[station_name] = station_data.precip
    if PathToQFlags != None:
        for i in range(len(false_data)):
            if false_data[i][0] in StationsList:
                tahmo_data[false_data[i][0]].loc[tahmo_data[false_data[i][0]].index.year == float(false_data[i][1])] = np.nan
        
    
    
    for i in range(len(tahmo_data)):
        station_name = list(ts_prec.keys())[i]
            
        if (station_name in list(ts_prec.keys())) & (station_name in list(tahmo_data.keys())):
            station_data = tahmo_data[station_name]
            ts_station = ts_prec[station_name]
            # ts_station[ts_station < 0] = np.nan
            # ts_station.index = pd.to_datetime(ts_station.index)
            # if monthly == True:
            prec_station = station_data.loc[ts_station.index]

            if (np.count_nonzero(~np.isnan(prec_station)) / len(prec_station) > p_data):
                # RMSE
                # print(prec_station, ts_station)
                if only_raindays == True:
                    ts_station_rainmask = ts_station > 0
                    prec_station_rainmask = prec_station > 0

                    # Combine the masks using the logical AND operator
                    ts_station_masked = ts_station[ts_station_rainmask & prec_station_rainmask]
                    prec_station_masked = prec_station[ts_station_rainmask & prec_station_rainmask]

                    squared_diff = np.square(np.subtract(prec_station_masked,ts_station_masked)) 
                else: 
                    squared_diff = np.square(np.subtract(prec_station,ts_station)) 
                mse = np.nanmean(squared_diff) 
                rmse = np.sqrt(mse)

                #Mean bias
                daily_difference = ts_station - prec_station 
                bias = daily_difference.mean()

                #Spearman Correlation
                if correlation == 'spearman':
                    valid_indices = ~np.isnan(prec_station) & ~np.isnan(ts_station)
                    filtered_data1 = prec_station[valid_indices]
                    filtered_data2 = ts_station[valid_indices]
                    spcor, _ = stats.spearmanr(filtered_data1, filtered_data2)
                if correlation == 'pearson':
                    valid_indices = ~np.isnan(prec_station) & ~np.isnan(ts_station)
                    filtered_data1 = prec_station[valid_indices]
                    filtered_data2 = ts_station[valid_indices]
                    spcor, p_val = np.corrcoef(filtered_data1, filtered_data2)
                ## KGE
                valid_indices = ~np.isnan(prec_station) & ~np.isnan(ts_station)
                filtered_data1 = prec_station[valid_indices]
                filtered_data2 = ts_station[valid_indices]

                cor, p_val = np.corrcoef(filtered_data1, filtered_data2)    
                alpha = np.std(filtered_data2) / np.std(filtered_data1)
                beta = np.mean(filtered_data2) / np.mean(filtered_data1)
                kge = 1 - np.sqrt((cor[1] - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

                prec_station_nonan = prec_station[~np.isnan(prec_station)]
                ts_station_nonan = ts_station[~np.isnan(prec_station)]
                prec_station_nonan = prec_station_nonan[~np.isnan(ts_station)]
                ts_station_nonan = ts_station_nonan[~np.isnan(ts_station)]


                prec_station_nonan[prec_station_nonan>0] = 1
                ts_station_nonan[ts_station_nonan>0] = 1

                #POD, FAR, HSS
                labels=[0,1]
                conf_matrix = confusion_matrix(prec_station_nonan, ts_station_nonan, labels=labels)
                tn, fp, fn, tp  = conf_matrix.ravel()
                pod = tp / (tp + fn)
                far = fp / (fp + tp)
                num = 2 * ((tp * tn) - (fp * fn))
                den = ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
                hss = num / den    


                rmse_dic[station_name] = rmse
                bias_dic[station_name] = bias
                spc_dic[station_name] = spcor 
                kge_dic[station_name] = kge
                pod_dic[station_name] = pod
                far_dic[station_name] = far
                hss_dic[station_name] = hss
    if getMean == False:
        return rmse_dic, bias_dic, spc_dic, kge_dic, pod_dic, far_dic, hss_dic
    if getMean == True:
        return np.nanmean(list(rmse_dic.values())), np.nanmean(list(bias_dic.values())), np.nanmean(list(spc_dic.values())),   np.nanmean(list(kge_dic.values())), np.nanmean(list(pod_dic.values())), np.nanmean(list(far_dic.values())), np.nanmean(list(hss_dic.values()))


def classify_precip(precip_value, min_value, max_value, nclasses):
    """
    Classify precipitation into nclasses

    precip_value: float
            Precipitation amount 
    min_value: float
            Min value in rainfall classification
    max_value: float
            Max value in rainfall classification 
    nclasses: int
            Number of rainfall classes
    Returns dictionary with classification
    """

    if nclasses > 1:
        class_width = (max_value - min_value) / (nclasses-1)
        for i in range(1, nclasses):
            if min_value + (i - 1) * class_width <= precip_value <= min_value + i * class_width:
                return {'Class': i}
            if precip_value > max_value:
                return {'Class': nclasses}
    else:
        return {'Class': nclasses}  
        
def create_rainfall_class_grid(path_to_tahmo, path_to_metadata, lon_coords, lat_coords, StationsList, min_value, max_value, nclasses):
    """
    Create a rainfall class grid from TAHMO stations

    path_to_tahmo: str
        Path to folder containing daily data of TAHMO stations
    path_to_metadata:  str
            path to TAHMO metadata file
    lon_coords: Array
            Array of longitude coordinates 
    lat_coords: Array
            Array of latitude coordinates
    StationsList: list of strings
            List of TAHMO station ID's 
    min_value: float
            Min value in rainfall classification
    max_value: float
            Max value in rainfall classification  
    nclasses: int
            Number of rainfall classes
    Returns xarray.Dataset with rainfall classification on the grid and Dictionary of TAHMO stations with corresponding rainfall classes 
    """
    
    tahmo_rainfall_year = {}
    for i in range(len(os.listdir(path_to_tahmo))):
        data_dir = os.path.join(path_to_tahmo, os.listdir(path_to_tahmo)[i])
        if data_dir.endswith('.csv'):
            station_name = os.path.basename(data_dir).split('.')[0]
            if station_name in StationsList:
                station_data = pd.read_csv(data_dir, index_col=0, parse_dates=True)

                precip_by_year  = station_data.precip.groupby(station_data.index.year).sum()

                years_to_exclude = station_data.precip.isnull().groupby(station_data.index.year).mean() >0.1
                precip_by_year = precip_by_year[~precip_by_year.index.isin(years_to_exclude[years_to_exclude].index)]

                if not precip_by_year.empty:
                    tahmo_rainfall_year[station_name] = precip_by_year.mean()

    sorted_rainfall = dict(sorted(tahmo_rainfall_year.items(), key=lambda item: item[1]))

    classified_precip = {station_name: classify_precip(precip_value, 0, 1800, nclasses) for station_name, precip_value in sorted_rainfall.items()}
    
    stations = list(classified_precip.keys())
    meta = pd.read_csv(path_to_metadata, index_col=0)
    meta = meta.loc[stations]

    for station in stations:
        classified_precip[station]['lon'] = meta.loc[station]['longitude']
        classified_precip[station]['lat'] =  meta.loc[station]['latitude']


    param_grid = xr.Dataset({
        param: (['lat', 'lon'], np.nan*np.zeros((len(lat_coords), len(lon_coords))))
        for param in ['Class']
    }, coords={'lat': lat_coords, 'lon': lon_coords})

    lon_grid, lat_grid = np.meshgrid(param_grid.lon.values, param_grid.lat.values)

    for station, data in classified_precip.items():
        # Find the nearest grid point based on latitude and longitude
        lat_diff = lat_grid - data['lat']
        lon_diff = lon_grid - data['lon']
        distances = np.sqrt(lat_diff**2 + lon_diff**2) #.T
        nearest_indices = np.unravel_index(distances.argmin(axis=None), distances.shape)

    #    Assign the parameter values to the nearest grid point
        param_grid['Class'].values[nearest_indices] = data['Class']

    indices = np.argwhere(np.isfinite(param_grid['Class'].values))
    lon = param_grid.lon.values
    lat = param_grid.lat.values

    coords = lat[indices[:, 0]], lon[indices[:, 1]]
    grid_z = griddata(coords, param_grid['Class'].values[np.isfinite(param_grid['Class'].values)], (lat_grid, lon_grid), method='nearest')
    param_grid['Class'].values = grid_z 
    
    return param_grid, classified_precip

def select_subset_stations(station_dict, num_stations):
    """
    Randomly select a subset of stations. if the num_stations>7, a station in every rainfall class is selected in the subset.

    station_dict: dictionary
        Dictionary of TAHMO stations with corresponding rainfall classes
    num_stations:  int
            Number of stations in subset
            
    Returns dictionary of selected stations and dictionary of unselected stations
    """
    
    if num_stations == 'All':
        return station_dict, None
    else:
        
        # Create a list of station IDs
        station_ids = list(station_dict.keys())

        # Shuffle the stations
        random.shuffle(station_ids)

        
        selected_stations = {}
        unselected_stations = {}
        
        # Separate stations by rain class
        stations_by_class = {}
        for station_id in station_ids:
            station_data = station_dict[station_id]
            rain_class = station_data['Class']
            if rain_class not in stations_by_class:
                stations_by_class[rain_class] = []
            stations_by_class[rain_class].append((station_id, station_data))

        # Ensure that each class is represented in the selected stations
        if len(stations_by_class) >= 7:
            for rain_class, stations_list in stations_by_class.items():
                selected_station_id, selected_station_data = stations_list.pop()
                selected_stations[selected_station_id] = selected_station_data
                if len(selected_stations) == 7:
                    break

        #remaining number of stations to select
        remaining_stations = num_stations - len(selected_stations)

        # Randomly select additional stations with random class IDs
        while remaining_stations > 0:
            random_class = random.choice(list(stations_by_class.keys()))

            if len(stations_by_class[random_class]) > 0:
                selected_station_id, selected_station_data = stations_by_class[random_class].pop()
                selected_stations[selected_station_id] = selected_station_data
                remaining_stations -= 1

        for rain_class, stations_list in stations_by_class.items():
            for station_id, station_data in stations_list:
                unselected_stations[station_id] = station_data

        return selected_stations, unselected_stations
    
def select_ascat_product(year):
    if year >= 2022:
        product = 'h26'
    if 2019 <= year < 2022:
        product = 'h142'
    if 1992 <= year < 2019:
        product = 'h141'
    if year < 1992:
        print('No product for selected year')
    return product