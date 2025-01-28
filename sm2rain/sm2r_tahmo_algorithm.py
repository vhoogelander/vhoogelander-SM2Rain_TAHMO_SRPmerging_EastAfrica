"""
 Some functions in this notebook are taken or adapted from the original SM2RAIN implementation, available at https://github.com/IRPIhydrology/sm2rain. 
These functions have been modified if needed and wrapped for calibration and computation using data from TAHMO stations.
"""
import os
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
import shutil
import requests
import ftplib
import calendar
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter


import numpy as np
from scipy.optimize import minimize

from pandas import Series


def ts_sm2rain_pd(smdf, sm2rainparams, thr=None, suffix='_SM2R'):
    """
    a wrapper for `ts_sm2rain_Tpot()` that allows using a pandas.Series
    directly as input

    Parameters
    ----------
    smdf : pandas.Series
        a soil-moisture timeseries with a datetime-index.
    sm2rainparams : dict
        a dict of the sm2rain parameters (a, b, z, T, c).
    thr : float, optional
        Upper threshold of p_sim (default: None).
    suffix : str, optional
        A suffix that will be added to the parameter-name in the returned
        Panas-Series (default: '_SM2R')

    Returns
    -------
    df : pandas.Series
        the estimated rainfall as a pandas.Series
        (named with the same name as smdf, but with a suffix "_SM2R")
    """
    assert isinstance(smdf, Series), 'smdf must be a pandas.Series!'
    df = Series(
        data=ts_sm2rain(jdates=smdf.index.to_julian_date().to_numpy(),
                        sm=smdf.to_numpy(),
                        **sm2rainparams,
                        thr=thr),
        index=smdf.index[:-1],
        name=smdf.name + suffix)

    return df


def calib_sm2rain_pd(smdf, rainfdf, NN, x0=None, bounds=None,
                     options=None, method='TNC'):
    """
    a wrapper for `calib_sm2rain_Tpot()` that allows using pandas.Series
    directly as input

    Parameters
    ----------
    smdf : pandas.Series
        a soil-moisture timeseries with a datetime-index.
    rainfdf : pandas.Series
        auxiliary rainfall timeseries with a datetime-index.
    NN : integer
        Data aggregation coefficient
    x0 : tuple, optional
        Initial guess of a, b, z, T, c
        (default: (8, 5.9, 49, 0.345, 0.12))
    bounds : tuple of tuples, optional
        Boundary values for a, b, z, T, c
        (default: ((0, 160), (1, 50), (10, 400), (0.05 3.00), (0.05 0.75)))
    options : dict, optional
        A dictionary of solver options
        (default: {'ftol': 1e-8, 'maxiter': 3000, 'disp': False}).
        For more explanation/options see scipy.minimize.
    method : str, optional
        Type of solver (default: 'TNC', i.e. Truncated Newton).
        For more explanation/options see scipy.minimize.

    Returns
    -------
    sm2rainparams : dict
        a dict of the estimated sm2rain parameters:
        (a [mm], b [-], z [mm], T [days], c [-])
    """

    assert isinstance(smdf, Series), 'smdf must be a pandas.Series!'
    assert isinstance(rainfdf, Series), 'rainfdf must be a pandas.Series!'

    jdates = smdf.index.to_julian_date().to_numpy()
    sm = smdf.to_numpy()
    p_obs = rainfdf.to_numpy()[:-1]

    sm2rain_params = calib_sm2rain_Tpot(jdates=jdates, sm=sm, p_obs=p_obs,
                                        NN=NN, x0=x0, bounds=bounds,
                                        options=options, method=method)

    return dict(zip(['a', 'b', 'z', 'T', 'c'], sm2rain_params))

def calib_sm2rain_rainclass(smdf_dic, rainfdf_dic, NN, x0=None, bounds=None,
                 options=None, method='TNC'):
    """
    a wrapper for `calib_sm2rain_Tpot_rainclass()` that allows using pandas.Series
    directly as input. This function is used for SM2Rain implementation with rainfall classes.

    Parameters
    ----------
    smdf_dic : pandas.Series
        a dictionary of soil-moisture timeseries with a datetime-index.
    rainfdf_dic : pandas.Series
        a dictionary of rainfall timeseries with a datetime-index.
    NN : integer
        Data aggregation coefficient
    x0 : tuple, optional
        Initial guess of a, b, z, T, c
        (default: (8, 5.9, 49, 0.345, 0.12))
    bounds : tuple of tuples, optional
        Boundary values for a, b, z, T, c
        (default: ((0, 160), (1, 50), (10, 400), (0.05 3.00), (0.05 0.75)))
    options : dict, optional
        A dictionary of solver options
        (default: {'ftol': 1e-8, 'maxiter': 3000, 'disp': False}).
        For more explanation/options see scipy.minimize.
    method : str, optional
        Type of solver (default: 'TNC', i.e. Truncated Newton).
        For more explanation/options see scipy.minimize.

    Returns
    -------
    sm2rainparams : dict
        a dict of the estimated sm2rain parameters:
        (a [mm], b [-], z [mm], T [days], c [-])
    """    

    
    assert isinstance(smdf_dic, dict), 'smdf must be a pandas.Series!'
    assert isinstance(rainfdf_dic, dict), 'rainfdf must be a pandas.Series!'
    jdates = smdf_dic[list(smdf_dic.keys())[0]].index.to_julian_date().to_numpy()

    sm_dic = smdf_dic
    p_obs_dic = rainfdf_dic 

    sm2rain_params = calib_sm2rain_Tpot_rainclass(jdates=jdates, sm_dic=sm_dic, p_obs_dic=p_obs_dic,
                                        NN=NN, x0=x0, bounds=bounds,
                                        options=options, method=method)

    return dict(zip(['a', 'b', 'z', 'T', 'c'], sm2rain_params)) 
    
def ts_sm2rain(sm, a, b, z, jdates=None, T=None, c=None, thr=None):
    """
    Retrieve rainfall from soil moisture.

    Parameters
    ----------
    sm : numpy.ndarray
        Single or multiple soil moisture time series.
    a : float, numpy.ndarray
        a parameter, units mm.
    b : float, numpy.ndarray
        b parameter, units -.
    z : float, numpy.ndarray
        Z parameter, units mm.
    jdates: numpy.ndarray
        Julian date time series.
    T : float, numpy.ndarray
        T parameter, units days.
    c : float, numpy.ndarray
        Tpot parameter, units days.
    thr : float, optional
        Upper threshold of p_sim (default: None).

    Returns
    -------
    p_sim : numpy.ndarray
        Single or multiple simulated precipitation time series.
    """
    if T is not None and c is None:
        swi = swicomp_nan(sm, jdates, T)
        swi = (swi-np.nanmin(swi))/(np.nanmax(swi)-np.nanmin(swi))
    elif c is not None:
        if not np.isnan(sm).all():
            swi = swi_pot_nan(sm, jdates, T, c)
            swi = (swi-np.nanmin(swi))/(np.nanmax(swi)-np.nanmin(swi))
        else: 
            swi = sm
            # swi[:] = np.nan
    else:
        swi = sm

    if jdates is None:
        jdates = np.arange(0, len(sm))

    p_sim = z * (swi[1:] - swi[:-1]) + \
        ((a * swi[1:]**b + a * swi[:-1]**b)*(jdates[1:]-jdates[:-1]) / 2.)

    p_sim[abs(np.diff(swi)) <= 0.0001] = 0
    p_sim[p_sim > 999999] = np.nan

    return np.clip(p_sim, 0, thr)


def swicomp_nan(in_data, in_jd, ctime):
    """
    Calculates exponentially smoothed time series using an
    iterative algorithm

    Parameters
    ----------
    in_data : double numpy.array
        input data
    in_jd : double numpy.array
        julian dates of input data
    ctime : int
        characteristic time used for calculating
        the weight
    """

    filtered = np.empty(len(in_data))
    gain = 1
    filtered.fill(np.nan)

    ID = np.where(~np.isnan(in_data))
    D = in_jd[ID]
    SWI = in_data[ID]
    tdiff = np.diff(D)

    # find the first non nan value in the time series

    for i in range(2, SWI.size):
        gain = gain / (gain + np.exp(- tdiff[i - 1] / ctime))
        SWI[i] = SWI[i - 1] + gain * (SWI[i] - SWI[i-1])

    filtered[ID] = SWI

    return filtered


def calib_sm2rain_Tpot(jdates, sm, p_obs, NN, x0=None, bounds=None,
                       options=None, method='TNC'):
    """
    Calibrate sm2rain parameters a, b, z, T, c.

    Parameters
    ----------
    jdates: numpy.ndarray
        Julian date time series.
    sm : numpy.ndarray
        Soil moisture time series.
    p_obs : numpy.ndarray
        Precipitation time series.
    NN : integer
        Data aggregation coefficient
    x0 : tuple, optional
        Initial guess of a, b, z, T, c
        (default: (10%, 5%, 10%, 10%, 10%) of bounds limits).
    bounds : tuple of tuples, optional
        Boundary values for a, b, z, T, c
        (default: ((0, 160), (1, 50), (10, 400), (0.05 3.00), (0.05 0.75))).
    options : dict, optional
        A dictionary of solver options
        (default: {'ftol': 1e-8, 'maxiter': 3000, 'disp': False}).
        For more explanation/options see scipy.minimize.
    method : str, optional
        Type of solver (default: 'TNC', i.e. Truncated Newton).
        For more explanation/options see scipy.minimize.

    Returns
    -------
    a : float
        a parameter, units mm.
    b : float
        b parameter, units -.
    z : float
        z parameter, units mm.
    T : float
        Tbase parameter, units days.
    c : float
        Tpot parameter, units -.
    """
    if bounds is None:
        bounds = ((0, 200), (0.01, 50), (1, 800), (0.05, 3.), (0.05, 0.75))

    if x0 is None:
        p = [0.1, 0.05, 0.1, 0.1, 0.1]
        x0 = np.array([(bounds[i][1] - bounds[i][0]) * p[i] + bounds[i][0]
                       for i in range(len(bounds))])

    if options is None:
        options = {'ftol': 1e-8, 'maxiter': 3000, 'disp': False}

    result = minimize(cost_func_Tpot, x0, args=(jdates, sm, p_obs, NN),
                      method=method, bounds=bounds, options=options)

    a, b, z, T, c = result.x

    return a, b, z, T, c



def calib_sm2rain_Tpot_rainclass(jdates, sm_dic, p_obs_dic, NN, x0=None, bounds=None,
                       options=None, method='TNC'):
    """
    Calibrate sm2rain parameters a, b, z, T, c.

    Parameters
    ----------
    jdates: numpy.ndarray
        Julian date time series.
    sm_dic : numpy.ndarray
        Dictionary of soil moisture time series.
    p_obs_dic : numpy.ndarray
        Dictionary of precipitation time series.
    NN : integer
        Data aggregation coefficient
    x0 : tuple, optional
        Initial guess of a, b, z, T, c
        (default: (10%, 5%, 10%, 10%, 10%) of bounds limits).
    bounds : tuple of tuples, optional
        Boundary values for a, b, z, T, c
        (default: ((0, 160), (1, 50), (10, 400), (0.05 3.00), (0.05 0.75))).
    options : dict, optional
        A dictionary of solver options
        (default: {'ftol': 1e-8, 'maxiter': 3000, 'disp': False}).
        For more explanation/options see scipy.minimize.
    method : str, optional
        Type of solver (default: 'TNC', i.e. Truncated Newton).
        For more explanation/options see scipy.minimize.

    Returns
    -------
    a : float
        a parameter, units mm.
    b : float
        b parameter, units -.
    z : float
        z parameter, units mm.
    T : float
        Tbase parameter, units days.
    c : float
        Tpot parameter, units -.
    """
    if bounds is None:
        bounds = ((0, 200), (0.01, 50), (1, 800), (0.05, 3.), (0.05, 0.75))

    if x0 is None:
        p = [0.1, 0.05, 0.1, 0.1, 0.1]
        x0 = np.array([(bounds[i][1] - bounds[i][0]) * p[i] + bounds[i][0]
                       for i in range(len(bounds))])

    if options is None:
        options = {'ftol': 1e-8, 'maxiter': 3000, 'disp': False}
    result = minimize(cost_func_Tpot_rainclass, x0, args=(jdates, sm_dic, p_obs_dic, NN),
                      method=method, bounds=bounds, options=options)

    a, b, z, T, c = result.x

    return a, b, z, T, c


def RainfallClass_params(nclasses, rainclass_dic, prec_ts_stations, sm_ts_stations, NN, x0=None, bounds=None, options=None, method='TNC'):
    """
    Wrapper function of calib_sm2rain_rainclass to for SM2Rain calibration per rainclass.

    Parameters
    ----------
    nclasses : integer
        Number of rainfall classes
    rainclass_dic : dictionary
        Dictionary of rainfall classification and coordinates of TAHMO stations.
    prec_ts_stations : dictionary
        Dictionary of precipitation series at TAHMO station locations.
    sm_ts_stations : dictionary
        Dictionary of soil moisture series at TAHMO station locations.
    NN : integer
        Data aggregation coefficient
    x0 : tuple, optional
        Initial guess of a, b, z, T, c
        (default: (10%, 5%, 10%, 10%, 10%) of bounds limits).
    bounds : tuple of tuples, optional
        Boundary values for a, b, z, T, c
        (default: ((0, 160), (1, 50), (10, 400), (0.05 3.00), (0.05 0.75))).
    options : dict, optional
        A dictionary of solver options
        (default: {'ftol': 1e-8, 'maxiter': 3000, 'disp': False}).
        For more explanation/options see scipy.minimize.
    method : str, optional
        Type of solver (default: 'TNC', i.e. Truncated Newton).
        For more explanation/options see scipy.minimize.


    Returns
    -------
    rmsd : float
        Root mean square difference between p_sim and p_obs.
    """
    param_dic = {}
    for Class in range(1, nclasses+1):
        rainfall_class = {key: value for key, value in rainclass_dic.items() if value['Class'] == Class}



        prec_ts_stations_class = {key: prec_ts_stations[key] for key in prec_ts_stations if key in rainfall_class}
        sm_ts_stations_class = {key: sm_ts_stations[key] for key in sm_ts_stations if key in rainfall_class}


        params = calib_sm2rain_rainclass(sm_ts_stations_class, prec_ts_stations_class, 400, x0=None, bounds=None,
                             options=None, method='TNC')
        
        param_dic[str(Class)] = params
    return param_dic


def cost_func_Tpot(x0, jdates, sm, p_obs, NN):
    """
    Cost function.

    Parameters
    ----------
    x0 : tuple
        Initial guess of parameters a, b, z, T, c.
    jdates: numpy.ndarray
        Julian date time series.
    sm : numpy.ndarray
        Soil moisture time series.
    p_obs : numpy.ndarray
        Observed precipitation time series.
    NN : integer
        Data aggregation coefficient

    Returns
    -------
    rmsd : float
        Root mean square difference between p_sim and p_obs.
    """
    p_sim = ts_sm2rain(sm, x0[0], x0[1], x0[2], jdates, x0[3], x0[4])

    p_sim1 = np.add.reduceat(p_sim, np.arange(0, len(p_sim), NN))
    p_obs1 = np.add.reduceat(p_obs, np.arange(0, len(p_obs), NN))
    rmsd = np.nanmean((p_obs1 - p_sim1)**2)**0.5

    return rmsd


def cost_func_Tpot_rainclass(x0, jdates, sm_dic, p_obs_dic, NN):
    """
    Cost function. This function is used for SM2Rain implementation with rainfall classes.

    Parameters
    ----------
    x0 : tuple
        Initial guess of parameters a, b, z, T, c.
    jdates: numpy.ndarray
        Julian date time series.
    sm_dic : numpy.ndarray
        Dictionary of soil moisture time series.
    p_obs_dic : numpy.ndarray
        Dictionary of observed precipitation time series.
    NN : integer
        Data aggregation coefficient

    Returns
    -------
    rmsd : float
        Root mean square difference between p_sim and p_obs.
    """
    station_list = list(sm_dic.keys())
    rmsd_list = []
    for station in station_list:
        sm = sm_dic[station].to_numpy()
        p_obs = p_obs_dic[station].to_numpy()[:-1]

        p_sim = ts_sm2rain(sm, x0[0], x0[1], x0[2], jdates, x0[3], x0[4])
        
        rmsd = np.nanmean((p_obs - p_sim)**2)**0.5
        rmsd_list.append(rmsd)
    return np.nanmean(rmsd_list)

def swi_pot_nan(sm, jd, t, POT):
    """
    Soil water index computation.

    Parameters
    ----------
    sm : numpy.ndarray
        Soil moisture time series.
    jd : numpy.ndarray
        Julian date time series.
    t : float, optional
        t parameter, the unit is fraction of days (default: 2).

    Returns
    -------
    swi : numpy.ndarray
        Soil water index time series.
    k : numpy.ndarray
        Gain parameter time series.
    """

    idx = np.where(~np.isnan(sm))[0]

    swi = np.empty(len(sm))
    swi[:] = np.nan
    swi[idx[0]] = sm[idx[0]]

    Tupd = t * sm[idx[0]] ** (- POT)
    gain = 1

    for i in range(1, idx.size):

        dt = jd[idx[i]] - jd[idx[i-1]]

        gain0 = gain / (gain + np.exp(- dt / Tupd))
        swi[idx[i]] = swi[idx[i - 1]] + gain0 * (sm[idx[i]] - swi[idx[i - 1]])
        Tupd = t * swi[idx[i]] ** (- POT)

        gain = gain0

    return swi  

def generate_sm2rainparams(sm_ts_stations, prec_ts_stations):
    """
    Wrapper of calib_sm2rain_pd to calibrate SM2Rain per station.

    Parameters
    ----------
    prec_ts_stations : dictionary
        Dictionary of precipitation series at TAHMO station locations.
    sm_ts_stations : dictionary
        Dictionary of soil moisture series at TAHMO station locations.
    Returns
    -------
    sm2rainparams : dictionary
        Dictionary of the SM2Rain parameters [a, b, z, T, c] per station

    """
    station_list = list(prec_ts_stations.keys())
    sm2rainparams = {}
    for station in station_list:
        sm_ts_station = sm_ts_stations[station]
        prec_ts_station = prec_ts_stations[station]

        if not np.isnan(prec_ts_station.values).any():

            sm2rainparams[station] = calib_sm2rain_pd(sm_ts_station, prec_ts_station, 400, x0=None, bounds=None,
                                 options=None, method='TNC')

    return sm2rainparams

def sm2rain_ts_station(sm_ts_stations, sm2rainparams_dic):
    """
    Generate SM2Rain timeseries per station

    Parameters
    ----------
    sm_ts_stations : dictionary
        Dictionary of soil moisture series at TAHMO station locations.
    sm2rainparams_dic : dictionary
        Dictionary of the SM2Rain parameters [a, b, z, T, c] per station
    Returns
    -------
    sm2rain : dictionary
        Dictionary of the SM2Rain timeseries per station

    """
    station_list = list(sm2rainparams_dic.keys())
    sm2rain = {}
    for station in station_list:
        if station in station_list:
            sm_ts_station = sm_ts_stations[station]
            sm2rainparams = sm2rainparams_dic[station]
            sm2rain[station] = ts_sm2rain_pd(sm_ts_station, sm2rainparams, thr=None, suffix='_SM2R')
    return sm2rain

def assign_sm2rparams_to_grid(lon_coords, lat_coords, path_to_metadata=None, param_dic_calibrated=None, param_grid_rainclass=None, rainclass=False, sigma = 5.0):
    """
    Assign SM2Rain parameters [a, b, z, T, c] to a xarray grid.

    Parameters
    ----------
    path_to_ascat_file : str
        Path to ASCAT SM file
    path_to_metadata : str
       Path to TAHMO metadata file
    param_dic_calibrated : dictionary
        Dictionary of calibrated SM2Rain parameters per rainclass or station
    param_grid_rainclass : xarray.Dataset
        xarray grid of rainclasses
    rainclass : True (default)
        If True, parameters are assigned according to the rainclass grid. If False, paramaters are assigned according to calibrated parameters per station.
    sigma : float 
        Gaussian filter parameter (default is 5.0)
    Returns
    -------
    param_grid : xarray dataset
        Dictionary of the SM2Rain timeseries per station

    """
    params = ['a', 'b', 'z', 'T', 'c']
    
    
    param_grid = xr.Dataset({
    param: (['lat', 'lon'], np.nan*np.zeros((len(lat_coords), len(lon_coords))))
    for param in params
}, coords={'lat': lat_coords, 'lon': lon_coords})
    
    if rainclass == False:
        stations = list(param_dic_calibrated.keys())
        meta = pd.read_csv(path_to_metadata, index_col=0)
        meta = meta.loc[stations]

        for station in stations:
            param_dic_calibrated[station]['lon'] = meta.loc[station]['longitude']
            param_dic_calibrated[station]['lat'] =  meta.loc[station]['latitude']

        lon_grid, lat_grid = np.meshgrid(param_grid.lon.values, param_grid.lat.values)


        for station, data in param_dic_calibrated.items():
            lat_diff = lat_grid - data['lat']
            lon_diff = lon_grid - data['lon']
            distances = np.sqrt(lat_diff**2 + lon_diff**2) #.T
            nearest_indices = np.unravel_index(distances.argmin(axis=None), distances.shape)

            for param in params:
                param_grid[param].values[nearest_indices] = data[param]
        for param in params:
            indices = np.argwhere(np.isfinite(param_grid[param].values))
            lon = param_grid.lon.values
            lat = param_grid.lat.values

            coords = lat[indices[:, 0]], lon[indices[:, 1]]
            grid_z = griddata(coords, param_grid[param].values[np.isfinite(param_grid[param].values)], (lat_grid, lon_grid), method='nearest')
            param_grid[param].values = grid_z  
            
    elif rainclass == True:
        param_grid['Class'] = param_grid_rainclass['Class']

        for class_id, params in param_dic_calibrated.items():
            param_grid['a'] = xr.where(param_grid['Class'] == int(class_id), param_dic_calibrated[str(class_id)]['a'], param_grid['a'])
            param_grid['b'] = xr.where(param_grid['Class'] == int(class_id), param_dic_calibrated[str(class_id)]['b'], param_grid['b'])
            param_grid['z'] = xr.where(param_grid['Class'] == int(class_id), param_dic_calibrated[str(class_id)]['z'], param_grid['z'])
            param_grid['T'] = xr.where(param_grid['Class'] == int(class_id), param_dic_calibrated[str(class_id)]['T'], param_grid['T'])
            param_grid['c'] = xr.where(param_grid['Class'] == int(class_id), param_dic_calibrated[str(class_id)]['c'], param_grid['c'])

        for param_name in param_grid.data_vars:
            param_data = param_grid[param_name]

            smoothed_param_data = gaussian_filter(param_data, sigma=sigma)

            smoothed_data_array = xr.DataArray(smoothed_param_data, dims=param_data.dims, coords=param_data.coords)

            param_grid[f'smoothed_{param_name}'] = smoothed_data_array
        
    return param_grid
def ts_sm2rain_grid(param_grid, path_to_ascat_file, smoothed=True):
    """
    Generate SM2Rain timeseries on a xarray grid.

    Parameters
    ----------
    param_grid : str
        Path to ASCAT SM file
    path_to_ascat_file : str
       Path to merged ASCAT SM timeseries file.
    smoothed : True (default)
        If True, SM2Rain timeseries are generated using a Gaussian-filtered parameter grid. 

    Returns
    -------
    DS_subset_sm2r : xarray.Dataset
        SM2Rain timeseries on a xarray grid

    """
    DS_subset = xr.open_dataset(path_to_ascat_file)
    lon_grid, lat_grid = np.meshgrid(param_grid.lon.values, param_grid.lat.values)

    lon_flat = lon_grid.flatten()
    lat_flat = lat_grid.flatten()

    Ncoords = len(lon_flat)
    Ntime = len(DS_subset.time)
    timeseries_gridcells = np.zeros((Ncoords, Ntime))

    for i, (lon, lat) in enumerate(zip(lon_flat, lat_flat)):
        if smoothed == True:
            a, b, z, T, c = param_grid.sel(lon=lon, lat=lat).smoothed_a.values, param_grid.sel(lon=lon, lat=lat).smoothed_b.values, param_grid.sel(lon=lon, lat=lat).smoothed_z.values, param_grid.sel(lon=lon, lat=lat).smoothed_T.values, param_grid.sel(lon=lon, lat=lat).smoothed_c.values
        else:
            a, b, z, T, c = param_grid.sel(lon=lon, lat=lat).a.values, param_grid.sel(lon=lon, lat=lat).b.values, param_grid.sel(lon=lon, lat=lat).z.values, param_grid.sel(lon=lon, lat=lat).T.values, param_grid.sel(lon=lon, lat=lat).c.values
            
        sm2rain_params = dict(zip(['a', 'b', 'z', 'T', 'c'], [a, b, z, T, c]))

        sm_ts = pd.Series(DS_subset.var40.sel(lon=lon, lat=lat).values.flatten())
        sm_ts.index = DS_subset.time.values
        sm_ts.name = str(f'{lon:.2f}')+'_'+str(f'{lat:.2f}')
        prec_ts = ts_sm2rain_pd(sm_ts, sm2rain_params, thr=None, suffix='_SM2R')
        prec_ts = np.append(prec_ts, np.nan)
        timeseries_gridcells[i] = prec_ts


    sm_ts_grid = []
    grid_dim = lat_grid.shape
    for i in range(len(timeseries_gridcells[0])):
        sm_T = timeseries_gridcells[:,i].reshape(grid_dim)
        sm_ts_grid.append(sm_T)

    sm_ts_grid_datarray = xr.DataArray(sm_ts_grid, dims=('time', 'lat', 'lon'))
    DS_subset_sm2r = DS_subset.assign(precip=sm_ts_grid_datarray)
    
    return DS_subset_sm2r
