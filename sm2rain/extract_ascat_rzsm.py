import os
import numpy as np
import eumdac
import datetime
import shutil
import requests
import ftplib
import calendar

def download_ascat_rzsm(local_path, days, months, year, login, password):
    """
    Download ASCAT Root Zone Soil Moisture data using a FTP connection to H SAF data if this data is not in given the folder.
    
    local_path: str
            Local path to ASCAT RZSM data folder. The folder should contain subfolders per year. Within the subfolders the downloaded files will be saved, or daily rzsm data is already stored.
    days: list of integers
            Selected days in month
    months: list of integers
            Selected months in year
    year: int
            year. TODO: allow for multiple years
    login: str
            H SAF account login
    password: str
            H SAF account password
    Returns filename, filepath 
    """
    if year >= 2022:
        path = 'h26/h26_cur_mon_nc'
        ftp = ftplib.FTP("ftphsaf.meteoam.it")
        ftp.login(login, password)  # Use your own H SAF credentials here (requires account)
        ftp.cwd(path)

        for month in months:
            _, num_days = calendar.monthrange(year, month)  # Get the number of days in the current month

            for day in range(1, num_days + 1):
                filename = f"h26_{year:04d}{month:02d}{day:02d}00_R01.nc"  ## Data format: h26_YYYYMMDD00_R01.nc
                file_path = local_path + "/" + filename
                if not os.path.exists(file_path):
                    with open(file_path, 'wb') as file:
                        ftp.retrbinary("RETR " + filename, file.write)
                    # ftp.retrbinary("RETR " + filename, open(filename, 'wb').write)

    elif 2019 <= year <= 2021:            
        path = f'h142/h142/netCDF4/{year}'
        ftp = ftplib.FTP("ftphsaf.meteoam.it")
        ftp.login(login, password)  # Use your own H SAF credentials here (requires account)
        ftp.cwd(path)

        for month in months:
            _, num_days = calendar.monthrange(year, month)  # Get the number of days in the current month

            for day in range(1, num_days + 1):
                filename = f"h142_{year:04d}{month:02d}{day:02d}00_R01.nc"  ## Data format: h142_YYYYMMDD00_R01.nc
                file_path = local_path + "/" + filename
                if not os.path.exists(file_path):
                    with open(file_path, 'wb') as file:
                        ftp.retrbinary("RETR " + filename, file.write)
                    # ftp.retrbinary("RETR " + filename, open(filename, 'wb').write)

    elif year <= 2018:
        path = f'h141/h141/netCDF4/{year}'
        ftp = ftplib.FTP("ftphsaf.meteoam.it")
        ftp.login(login, password)  # Use your own H SAF credentials here (requires account)
        ftp.cwd(path)

        for month in months:
            _, num_days = calendar.monthrange(year, month)  # Get the number of days in the current month

            for day in range(1, num_days + 1):
                filename = f"h141_{year:04d}{month:02d}{day:02d}00_R01.nc"  ## Data format: h141_YYYYMMDD00_R01.nc
                file_path = local_path + "/" + filename
                if not os.path.exists(file_path):
                    with open(file_path, 'wb') as file:
                        ftp.retrbinary("RETR " + filename, file.write)
                    # ftp.retrbinary("RETR " + filename, open(filename, 'wb').write)

    ftp.quit()
    return filename, file_path