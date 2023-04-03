import numpy as np
import numpy.ma as ma

# Remove warning and live a happy life
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Paths

import netCDF4
from netCDF4 import Dataset
def importNetcdf(path,variable_name):
    nc_fid = Dataset(path, 'r')
    field = nc_fid.variables[variable_name][:]
    return field

# Load lats and lons for these time series
def importNetcdf_lat_lon(path,variable_name):
    nc_fid = Dataset(path, 'r')
    lat = nc_fid.variables[variable_name].latitude
    lon = nc_fid.variables[variable_name].longitude
    coords = [lat,lon]
    return coords

# From masked array to np.array
def masked_array_to_numpy(data):
    return np.ma.filled(data.astype(np.float32), np.nan);

