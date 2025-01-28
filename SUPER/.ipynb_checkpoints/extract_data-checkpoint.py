import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
#matplotlib inline
from datetime import datetime, timedelta
import time
from mpl_toolkits.basemap import Basemap
from netCDF4 import Dataset
import pandas as pd
import requests
import xarray as xr


def extract_imerg_daily(start_time,end_time,subset,path_imerg_data):
       def build_imerg_url(start_time, Lat_index,Lon_index, run='L',version='06B', opendap=True):

        Lon_Minsub,Lon_Maxsub = [Lon_index.min(),Lon_index.max()]
        Lat_Minsub,Lat_Maxsub = [Lat_index.min(),Lat_index.max()]
        """Build URL to IMERG data"""

        end = start_time + pd.Timedelta(minutes=29, seconds=59)

        product = f'GPM_3IMERGD{run}.{version[:2]}'
        url = (f'{OPENDAP_URL}/'
                f'{product}/{start_time.year}/{start_time.month:02}/'
                f'3B-DAY-{run}.MS.MRG.3IMERG.'
                f'{start_time.year}{start_time.month:02}{start_time.day:02}-S000000-E235959.V{version}.nc4')

        url_subset = (f'{url}.nc4?precipitationCal%5B0:1:47%5D%5B0:{Lon_Minsub}:{Lon_Maxsub}%5D%5B0:{Lat_Minsub}:{Lat_Maxsub}%5D,'
                        f'time%5B0:1:47%5D,lat%5B0:{Lat_Minsub}:{Lat_Maxsub}%5D,lon%5B0:{Lon_Minsub}:{Lon_Maxsub}%5D')
        # url_subset2 = (f'{url}.nc4?precipitationCal[0:1:0][{Lon_Minsub}:1:{Lon_Maxsub}][{Lat_Minsub}:1:{Lat_Maxsub}],time[0:1:0],lat[{Lat_Minsub}:1:{Lat_Maxsub}],lon[{Lon_Minsub}:1:{Lon_Maxsub}]')
        url_subset2 = (f'{url}.nc4?precipitationCal[0:1:0][{Lon_Minsub}:1:{Lon_Maxsub}][{Lat_Minsub}:1:{Lat_Maxsub}],lat[{Lat_Minsub}:1:{Lat_Maxsub}],lon[{Lon_Minsub}:1:{Lon_Maxsub}],time[0:1:0]')
        # print(url_subset2)
        return url_subset2

    def download_imerg(url,username,password, path=path_imerg_data):
        """Downloads and saves IMERG file"""
        file_name_ = url[url.rfind('/')+1:len(url)]
        file_name = file_name_[0:file_name_.rfind('?')]
        file_path = os.path.join(path, file_name)
        if not os.path.exists(file_path):
            # r = requests.get(url)
            
            
                    # if not os.path.exists(file_path):
            
            with requests.Session() as session:
                req = session.request('get', url)
                r = session.get(req.url, auth=('USERNAME','PASSWORD'))
                if r.ok:
                    # imagename = url.split('.')

                    with open(file_path, 'wb') as f:
                        f.write(r.content)            
            
                else: 
                    raise RuntimeError(f'{r.status_code}: could not download {r.url}')
        return file_path
    
    [Xmin_lbm,Xmax_lbm,Ymin_lbm,Ymax_lbm] =  [-180, 180,-90, 90]
    [MinLon,MaxLon,MinLat,MaxLat] = subset
    
    Lon = np.arange(Xmin_lbm,Xmax_lbm,0.1)
    Lat = np.arange(Ymin_lbm,Ymax_lbm,0.1)
    Lon_index = np.asanyarray(np.where( (Lon >= MinLon ) & (Lon <= MaxLon) ) )
    Lat_index = np.asanyarray(np.where( (Lat >= MinLat ) & (Lat <= MaxLat) ) )
    OPENDAP_URL = 'https://gpm1.gesdisc.eosdis.nasa.gov/opendap/hyrax/GPM_L3' #'https://gpm1.gesdisc.eosdis.nasa.gov/opendap/hyrax/GPM_L3'
    freq = 'D'
    start_time = start_time.strftime("%m/%d/%Y/%H")
    end_time = end_time.strftime("%m/%d/%Y/%H")
    time_steps = pd.date_range(start_time, end_time, freq= freq)
    cache_dir=os.path.join(path_imerg_data)
    
    urls = [build_imerg_url(d, Lat_index,Lon_index, run='L',version='06') for d in time_steps]
    files = [download_imerg(u,'USERNAME','PASSWORD',path=cache_dir) for u in urls]
    
    imerg = xr.open_mfdataset(files, parallel=True)
    precipitation = imerg['precipitationCal']

    # imerge_hourly_numpy = imerg_hourly.values

    return precipitation


def extract_era5_prec_daily(start_time, end_time, subset, path_era5_data):
    def generate_era5_prec_url(start_time, end_time, subset):
        import cdsapi
        time_steps = pd.date_range(start_time, end_time, freq= 'D')
        [MinLon,MaxLon,MinLat,MaxLat] =  subset
        subset = [MinLat,MinLon,MaxLat,MaxLon]
        years =  list(map(str, set([d.year for d in time_steps])))
        months =  list(map(lambda x: f"{x:02}", set([d.month for d in time_steps])))
        days =  list(map(lambda x: f"{x:02}", set([d.day for d in time_steps])))
        c = cdsapi.Client()
        download = c.retrieve('reanalysis-era5-single-levels',
            {   'product_type': 'reanalysis',
                'area': subset,
                "format": "netcdf",
                'variable': 'total_precipitation',
                "year": years,
                "month": months,
                "day": days,
                "time": ["00:00","01:00","02:00",
                        "03:00","04:00","05:00",
                        "06:00","07:00","08:00",
                        "09:00","10:00","11:00",
                        "12:00","13:00","14:00",
                        "15:00","16:00","17:00",
                        "18:00","19:00","20:00",
                        "21:00","22:00","23:00",
                        ],

            }, "download.nc")

        url = download.location
        return url

    def download_era5_prec(start_time, end_time, subset, path_era5_data):
        [MinLon,MaxLon,MinLat,MaxLat] =  subset

        time_steps = pd.date_range(start_time, end_time, freq= 'D')

        years =  list(map(str, set([d.year for d in time_steps])))
        months =  list(map(lambda x: f"{x:02}", set([d.month for d in time_steps])))
        days =  list(map(lambda x: f"{x:02}", set([d.day for d in time_steps])))

        file_name = (
        'era5_' + 
        str(MinLon) + '_' + str(MaxLon) + '_' +
        str(MinLat) + '_' + str(MaxLat) + '_' +
        time_steps[0].strftime('%Y%m%d').zfill(6) + '_' +
        time_steps[-1].strftime('%Y%m%d').zfill(6) +
        '.nc'
    )
        file_path = os.path.join(path_era5_data, file_name)

        if not os.path.exists(file_path):
            url = generate_era5_prec_url(start_time, end_time, subset)
            r = requests.get(url)
            # r = requests.get(url, auth=(username,password))
            if r.ok:
                os.makedirs(path_era5_data, exist_ok=True)
                with open(file_path, 'wb') as f:
                    f.write(r.content)
            else: 
                raise RuntimeError(f'{r.status_code}: could not download {r.url}')
        # print(file_path)        

        era5 = xr.open_dataset(file_path)
        era5_daily = era5.resample({'time': 'D'}).sum().load() 
        return era5_daily
    
    [MinLon,MaxLon,MinLat,MaxLat] =  subset
    files = os.listdir(path_era5_data)
    coordinate_pattern = f"_{MinLon}_{MaxLon}_{MinLat}_{MaxLat}_"

    # Check if any file matches the coordinate pattern
    matching_files = [
        filename for filename in files if coordinate_pattern in filename
    ]
    years = [year for year in range(start_time.year , end_time.year+1)]
    unavailable_years = []
    if matching_files:

        filepaths = []

        for filename in matching_files:
            parts = filename.split('_')
            start_time_str = parts[-2]
            end_time_str = parts[-1].split('.')[0]


            # Get start/end time of the file
            start_time_file = datetime.strptime(start_time_str, "%Y%m%d")
            end_time_file = datetime.strptime(end_time_str, "%Y%m%d")
            years_files = [year for year in range(start_time_file.year , end_time_file.year+1)] 
            if any(year in years for year in years_files):
                filepaths.extend([path_era5_data + filename])
            unavailable_years.append([year for year in years if year not in years_files])

        flat_list = [year for sublist in unavailable_years for year in sublist]
        non_unique_list = [year for year in flat_list if flat_list.count(year) == len(matching_files)]
        unavailable_years = sorted(set(non_unique_list))

        if bool(filepaths):
            # ds = xr.open_mfdataset(filepaths, combine='nested', compat='override' ,join ='outer')
            datasets = [xr.open_dataset(fp) for fp in filepaths]

            ds = xr.concat(datasets, dim='time', coords='all', combine_attrs='override')

        else:
            print("no files with matching start or end time found.")
        era5_files = ds.sel(time=slice(str(start_time.year),str(end_time.year)))   


        for year in unavailable_years:
            print(f"Downloading ERA5 data of {year} in {subset}")
            first_day, last_day = datetime(year, 1, 1), datetime(year, 12, 31)
            era5_year = download_era5_prec(first_day, last_day, subset, path_era5_data)

            era5_files = xr.concat([era5_files, era5_year], dim='time')
        era5_daily = era5_files.sortby('time').resample({'time': 'D'}).sum().load() 
        era5_daily = era5_daily.sel(time=slice(start_time,end_time))
    else:
        print("No files with matching coordinates found.")   
        era5_daily = download_era5_prec(start_time, end_time, subset, path_era5_data)
    return era5_daily

def extract_chirps_daily(start_time, end_time, subset, path_chirps_data, delete_global_file=False):
    def download_chirps_global(start_time,end_time,path_chirps_data):
        years = list(range(start_time.year, end_time.year + 1))
        global_dir = path_chirps_data + '/global'
        if not os.path.exists(global_dir):
            os.makedirs(global_dir)
        for year in years:
            file_name = 'chirps_global_'+str(year)+'.nc'
            file_path = os.path.join(global_dir, file_name)
            if not os.path.exists(file_path):
                url = f'https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/netcdf/p05/chirps-v2.0.{year}.days_p05.nc'
                r = requests.get(url)

                if r.ok:
                    os.makedirs(global_dir, exist_ok=True)
                    with open(file_path, 'wb') as f:
                        f.write(r.content)
        return global_dir
    def chirps_subset(start_time, end_time, subset, path_chirps_data, path_chirps_global):
        [MinLon,MaxLon,MinLat,MaxLat] = subset
        time_steps = pd.date_range(start_time, end_time, freq= 'D')
        ds = xr.open_mfdataset(os.path.join(path_chirps_global, '*.nc'))
        ds_subset = ds.sel(latitude=slice(MinLat,MaxLat), longitude=slice(MinLon,MaxLon)).sel(time=slice(str(start_time), str(end_time)))
        file_name = 'chirps_' + str(MinLon)+'_'+str(MaxLon)+'_'+str(MinLat)+'_'+str(MaxLat)+'_'+str(time_steps[0].year)+str(time_steps[0].month)+'_'+str(time_steps[-1].year)+str(time_steps[-1].month)+'.nc'
        file_path = os.path.join(path_chirps_data, file_name)
        if not os.path.exists(file_path):
            ds_subset.to_netcdf(file_path)
        ds.close()
        return ds_subset
    
    path_chirps_global = download_chirps_global(start_time,end_time,path_chirps_data)
    chirps_subset = chirps_subset(start_time, end_time, subset, path_chirps_data, path_chirps_global)
    if delete_global_file == True:
        os.remove(file_path)
    return chirps_subset

def extract_chpclim(subset, path_chpclim_data, delete_global_file=False):
    def download_chpclim_global(path_chpclim_data):
        file_name = 'chpclim_global_monthly_9090.nc'
        file_path = os.path.join(path_chpclim_data, file_name)
        if not os.path.exists(file_path):
            url = f'http://data.chc.ucsb.edu/products/CHPclim/netcdf/chpclim.9090.monthly.nc'
            r = requests.get(url)

            if r.ok:
                os.makedirs(path_chpclim_data, exist_ok=True)
                with open(file_path, 'wb') as f:
                    f.write(r.content)
        return file_path

    def chpclim_subset(subset, path_chpclim_data, chpclim_file_path_global):
        [MinLon,MaxLon,MinLat,MaxLat] = subset
        ds = xr.open_dataset(chpclim_file_path_global)
        ds_subset = ds.sel(latitude=slice(MinLat,MaxLat), longitude=slice(MinLon,MaxLon))
        file_name = 'chpclim' + str(MinLon)+'_'+str(MaxLon)+'_'+str(MinLat)+'_'+str(MaxLat)+'.nc'
        file_path = os.path.join(path_chpclim_data, file_name)
        if not os.path.exists(file_path):
            ds_subset.to_netcdf(file_path)
        ds.close()
        return ds_subset
    
    chpclim_file_path_global = download_chpclim_global(path_chpclim_data)
    chpclim_subset = chpclim_subset(subset, path_chpclim_data, chpclim_file_path_global)
    if delete_global_file == True:
        os.remove(chpclim_file_path_global)
    return chpclim_subset