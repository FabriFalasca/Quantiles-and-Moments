#Libraries and stuff

import numpy as np
import numpy.ma as ma

import matplotlib.pyplot as plt
from matplotlib import ticker

import netCDF4
from netCDF4 import Dataset

# Remove warning and live a happy life
import warnings
warnings.filterwarnings("ignore")

from matplotlib import dates
from datetime import date, timedelta
import pandas as pd

# Parallelization
from joblib import Parallel, delayed
import multiprocessing

########################### Define parameters

# quantiles considered
qs = np.arange(0.05, 1.00, 0.05)
# np.arange(0.02, 1.00, 0.02)
# Number of bootstrap
n = 1000
n_cores = multiprocessing.cpu_count()
#n_cores = 4
# 95% significance
alpha = 0.05

import time
start = time.time()

########################### Loading the data

import utils_load_data
import utils

path = './zos_plus_ib_CM4_100yrs_0p5_deg_nohl.nc'

print('Load data')

# Load the data
data = utils_load_data.importNetcdf(path,'SLA')
data = utils_load_data.masked_array_to_numpy(data)

# Time vector to use
time = utils_load_data.importNetcdf(path, 'TIME1')
time = utils_load_data.masked_array_to_numpy(time)

# longitude and latitude
lon = utils_load_data.importNetcdf(path,'lon')
lon = utils_load_data.masked_array_to_numpy(lon)

lat = utils_load_data.importNetcdf(path,'lat')
lat = utils_load_data.masked_array_to_numpy(lat)

lat_long_array = []

for i in range(len(lat)):
    for j in range(len(lon)):     
        lat_long_array.append([lat[i],lon[j]])

lat_long_array = np.array(lat_long_array)  
########################### Preprocessing

dimT = np.shape(data)[0]
dim_lat = np.shape(data)[1]
dim_lon = np.shape(data)[2]

print('Consider coast only')

mask = utils.mask_coast_only(data[0])
only_coast_field = data * mask

print('# of coastal points: '+str(np.nansum(mask)))
# From a [dimT,dimY,dimX] array
# to a list of (dimY x dimX) spectra of length dimV
flat_data_masked = only_coast_field.reshape(dimT,dim_lat*dim_lon).transpose()
# Consider only the points that are not masked
flat_data = flat_data_masked[~np.isnan(np.sum(flat_data_masked,axis=1))]

########################### Longitude and latitudes for coastal indices

print('Consider only longitudes and latitudes of coastal points')
# to a list of (dimY x dimX) 
flat_mask = mask.reshape(dim_lat*dim_lon).transpose()
# Indices wih coastal grid points
coastal_indices = np.where(~np.isnan(flat_mask))[0]
# longitudes and latitides for coastal points
lat_long_array_coast = lat_long_array[coastal_indices]

########################### Running the model

# Parallelization
from joblib import Parallel, delayed
import utils

print('Running changes in moments in parallel')
print('Number of bootstrap is n = '+str(n))

# all_coeffs, all_sigs, all_bootstrapped_PDF =
results = Parallel(n_jobs=n_cores)(delayed(utils.changes_in_moments)(time,i,n,qs,tau_range,alpha) for i in flat_data)
results = np.array(results,dtype=object)


print('Saving results')
np.save('./zos_ib_B1000_block_1_season.npy',results)
np.save('./long_and_lats_coast_zos_ib_blocks_1season.npy',lat_long_array_coast)

end = time.time()
print('Finished in '+str(round(end - start, 2))+' seconds')
