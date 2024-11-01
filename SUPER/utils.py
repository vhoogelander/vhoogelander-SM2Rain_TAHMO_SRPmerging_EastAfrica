import pandas as pd
import numpy as np
import xarray as xr
import os
import scipy.stats as stats
from sklearn.metrics import confusion_matrix

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
    return pd.Series(data=netCDFDataAtLocation[variable].values, index=netCDFDataAtLocation.time.values) 

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


def regrid_dataset(ds, subset, resolution):
    """
    Regrid xarray dataset with a given subset and resolution

    ds: xarray.Dataset
            xarray dataset
    subset: list of floats
            [MinLon,MaxLon,MinLat,MaxLat]
    resolution: float
            
    Returns regridded xarray dataset
    """

    # Subset the dataset based on the provided lon/lat range
    [MinLon,MaxLon,MinLat,MaxLat] = subset
    
    if 'longitude' in ds.dims:
        ds = ds.rename(longitude='lon')
    if 'latitude' in ds.dims:
        ds = ds.rename(latitude='lat')
        
    ds = ds.sortby('lat')
    ds = ds.sortby('lon') 
    
    ds_subset = ds.sel(lat=slice(MinLat,MaxLat), lon=slice(MinLon,MaxLon))

    #Define the new lon/lat coordinates
    lon_new = np.arange(subset[0], subset[1] + resolution, resolution)
    lat_new = np.arange(subset[2], subset[3] + resolution, resolution)

    # Regrid the dataset to the new grid
    ds_regridded = ds_subset.interp(lon=lon_new, lat=lat_new, method='nearest') #,kwargs={"fill_value": "extrapolate"} 

    return ds_regridded

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


def GetDetectionAbilityPerStation(PathToTahmoStationData, ts_prec, StationsList, p_data, category='norain', getMean=True):
    
    """
    Calculate the detection ability of a rainfall category: Probability of Detection (POD), False Alarm Ratio (FAR), and Heidke Skill Score (HSS)
    
    PathToTahmoStationData: str
            Path to folder containing daily data of TAHMO stations
    ts_prec: str
            Dictionary of simulated precipitation timeseries at all TAHMO stations. (Can be created using timeSeriesAllTahmoFromNetCDF(netCDFFilePath, MetaFilePath, StationsList, variable))
    StationsList: list of strings
            List of TAHMO station ID's
    p_data: float
            Minimum percentage of "true data" at station
    category: str
        norain (Default) : rain < 1
        light: 1<= rain < 5
        moderate1: 5<= rain < 10
        moderate2: 10 <= rain < 20
        heavy: 20 <= rain < 40
        extreme: rain >= 40
    getMean: True (Default)
            If True, returns of mean statistics of all TAHMO stations (pod:float, far:float, hss:float). If False, returns dictionary of statistics at all TAHMO stations
            
    """

    pod_dic = {} #Probability of detction
    far_dic = {}
    hss_dic = {}
    for i in range(len(os.listdir(PathToTahmoStationData))):
        data_dir = os.path.join(PathToTahmoStationData, os.listdir(PathToTahmoStationData)[i])
        if os.path.basename(data_dir).startswith('TA'):
            station_name = os.path.basename(data_dir).split('.')[0]
            
            if station_name in list(ts_prec.keys()):
                station_data = pd.read_csv(data_dir, index_col=0, parse_dates=True)
                ts_station = ts_prec[station_name]
                ts_station[ts_station < 0] = np.nan
                ts_station.index = pd.to_datetime(ts_station.index)
                prec_station = station_data['precip'].loc[ts_station.index]
                if (np.count_nonzero(~np.isnan(prec_station)) / len(prec_station) > p_data):
 
                    prec_station_nonan = prec_station[~np.isnan(prec_station)]
                    ts_station_nonan = ts_station[~np.isnan(prec_station)]
                    prec_station_nonan = prec_station_nonan[~np.isnan(ts_station)]
                    ts_station_nonan = ts_station_nonan[~np.isnan(ts_station)
                    
                    if category == 'norain':
                        prec_station_nonan = np.where((prec_station_nonan < 0.5), 1, 0)
                        ts_station_nonan = np.where((ts_station_nonan < 1), 1, 0)
                        
                    if category == 'lightrain':
                        prec_station_nonan = np.where((prec_station_nonan >= 0.5) & (prec_station_nonan < 5), 1, 0)
                        ts_station_nonan = np.where((ts_station_nonan >= 0.3) & (ts_station_nonan < 8), 1, 0)
                    if category == 'moderaterain':
                        prec_station_nonan = np.where((prec_station_nonan >= 5) & (prec_station_nonan < 20), 1, 0)
                        ts_station_nonan = np.where((ts_station_nonan >= 4) & (ts_station_nonan < 22), 1, 0)
                    if category == 'heavyrain':
                        prec_station_nonan = np.where((prec_station_nonan >= 40), 1, 0)
                        ts_station_nonan = np.where((ts_station_nonan >= 40), 1, 0)                        
                        
                    if category == 'light':
                        prec_station_nonan = np.where((prec_station_nonan >= 0.5) & (prec_station_nonan < 5), 1, 0)
                        ts_station_nonan = np.where((ts_station_nonan >= 0.5) & (ts_station_nonan < 5), 1, 0)
                        
                    if category == 'moderate1':
                        prec_station_nonan = np.where((prec_station_nonan >= 5) & (prec_station_nonan < 10), 1, 0)
                        ts_station_nonan = np.where((ts_station_nonan >= 5) & (ts_station_nonan < 10), 1, 0)
                        
                    if category == 'moderate2':
                        prec_station_nonan = np.where((prec_station_nonan >= 10) & (prec_station_nonan < 20), 1, 0)
                        ts_station_nonan = np.where((ts_station_nonan >= 10) & (ts_station_nonan < 20), 1, 0)
                        
                    if category == 'heavy':
                        prec_station_nonan = np.where((prec_station_nonan >= 20) & (prec_station_nonan < 40), 1, 0)
                        ts_station_nonan = np.where((ts_station_nonan >= 20) & (ts_station_nonan < 40), 1, 0)
                        
                    if category == 'extreme':
                        prec_station_nonan = np.where((prec_station_nonan >= 40), 1, 0)
                        ts_station_nonan = np.where((ts_station_nonan >= 40), 1, 0)
                    
                    #POD, FAR, HSS
                    labels=[0,1]
                    conf_matrix = confusion_matrix(prec_station_nonan, ts_station_nonan, labels=labels)
                    tn, fp, fn, tp  = conf_matrix.ravel()
                    pod = tp / (tp + fn)
                    far = fp / (fp + tp)
                    num = 2 * ((tp * tn) - (fp * fn))
                    den = ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
                    hss = num / den    

                    pod_dic[station_name] = pod
                    far_dic[station_name] = far
                    hss_dic[station_name] = hss
    if getMean == False:
        return pod_dic, far_dic, hss_dic
    if getMean == True:
        return np.nanmean(list(pod_dic.values())), np.nanmean(list(far_dic.values())), np.nanmean(list(hss_dic.values()))    
