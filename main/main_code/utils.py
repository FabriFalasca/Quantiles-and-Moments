# Libraries and stuff

import numpy as np
import numpy.ma as ma

# Remove warning and live a happy life
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import pandas as pd

import sys
import os
import time
import numpy as np
import scipy.stats

# library for bootstrapping
import recombinator
from recombinator.block_bootstrap import moving_block_bootstrap

def masked_array_to_numpy(data):
    return np.ma.filled(data.astype(np.float32), np.nan);

def get_nonmask_indices(data):
    return np.argwhere(~np.isnan(np.sum(data,axis=0)));

'''

MAIN CODE for quantifying (linear) changes in distributions from time series using a fast implementation 
of quantile regression and projection onto Hermite polynomials.

- The code to compute quantile regression is based on the following paper: 
D. Chernozhukov, I. Fernàndez-Val, and B. Melly. Fast algorithms for the quantile regression process. Empir Econ., 62:7–33,
2022. doi: https://doi.org/10.1007/s00181-020-01898-0

- The implementation of the method of Chernozhukov et al. (2022), can be found in the quantile regression R package: 
R. Koenker. quantreg: Quantile Regression. R package version 5.94, 2022. URL http://CRAN.R-project.org/package=quantreg

- These R functions are called in our Python environment through the rpy2 package available here https://rpy2.github.io/

- The projection over Hermite polynomials is proposed in our paper and mainly comes from this paper
E. A. Cornish and R. A. Fisher. Moments and Cumulants in the Specification of Distributions. Revue De L’Institut International
De Statistique / Review of the International Statistical Institute, 5(4):307–320, 1937. doi: https://doi.org/10.2307/1400905

Importantly, when working with models' outputs we would like to focus on coastal sea level only. 
The function mask_coast_only takes a spatial field (e.g., sea surface temperature at time t = 0). 
The land is supposed to be "nan". The function outputs a new spatial field where coastal points 
(points in the ocean, adjacent to the coast) have value 1. Everyting else is nan.

Contact
Fabrizio Falasca, fabrifalasca@gmail.com

'''

################## What's in here

# (a) Code for quantile regression + statistical significance

'''
Functions

- q_reg_analysis (MAIN FUNCTION USED FOR QUANTILE REGRESSION):
Give a time series yt compute the quantile regression for a set of quantiles q in [0,1].
'''

# (b) Code quantifying changes in the first four statistical moments

'''

- basis_functions:
the 4 basis functions derived from the Cornish-Fisher expansion

- bootstrap_time_series:
it block-bootstrapps a time series n time and returns the result

- changes_in_moments (MAIN FUNCTION TO QUANTIFY CHANGES IN MOMENTS):
for a given time series it computes the slopes in mean, variance, skewness and kurtosis of
the distribution plus their significance. Crucially, here we quantify independent sources of 
changes in the first four moments.

'''

################# Step (a): QUANTILE REGRESSION

from sklearn.utils import resample

################ Quantile regression in R
import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

# Function that takes values in y (e.g., sea level) and x (e.g., time) and transform it
# in into a dataframe to be read by R
def r_object(xt, yt):

    # input:
    # - xt: time array
    # - yt: climate observable

    # output:
    # r dataframe

    x = xt
    x = x - x[0]
    y = yt
    y_not_nans = y[~np.isnan(y)]
    x_not_nans = x[~np.isnan(y)]
    # Storing in dataframe
    df = pd.DataFrame({'x': x_not_nans, 'y': y_not_nans})

    #converting the dataframe into r object for passing into r function
    with localconverter(robjects.default_converter + pandas2ri.converter):
      df_r = robjects.conversion.py2rpy(df)

    return df_r

################################################################################
# Defining the R script and loading the instance in Python
r = robjects.r
r['source']('utils_R.R')

print('Loading the function we have defined in R.')
# Loading the function we have defined in R.
quantile_reg_R = robjects.globalenv['qr_function']
multiple_quantile_reg_R = robjects.globalenv['multiple_qr']
################################################################################

def q_regressions(xt,yt):

    # Input:
    #       - xt: is time
    #       - yt: is the variable of interest (sea level for us)

    # We have nans in the data: we do not consider them in the analysis

    # Output:
    #       - slope

    # From xt and yt to a R dataframe
    df_r = r_object(xt, yt)
    # Compute slopes
    slopes_r = multiple_quantile_reg_R(df_r)
    # Finally we convert back the results into Python
    with localconverter(robjects.default_converter + pandas2ri.converter):
      slopes = robjects.conversion.rpy2py(slopes_r)

    return slopes

####################### Here we add a code for quantile regression in Python
# This is OK in case of 1 or few quantile regressions
# It is prohibitevely slow in case of multiple ones

import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd

def q_regression_Python(xt,yt,q):

    # Input:
    #       - xt: is time
    #       - yt: is the variable of interest (sea level for us)
    #       - q:  quantile of interest (i.e., q = 0.05, 0.5, 0.95)

    # We have nans in the data: we do not consider them in the analysis

    # Output:
    #       - trend
    #       - slope

    # Set the time vector going from 0 to T
    xt = xt - xt[0]

    x = xt.copy()
    y = yt.copy()
    # Do not consider nan
    y_not_nans = y[~np.isnan(y)]
    x_not_nans = x[~np.isnan(y)]

    df = pd.DataFrame({'time': x_not_nans, 'timeseries': y_not_nans})
    model_q = smf.quantreg('timeseries ~ time', df).fit(q=q,max_iter = 10000,p_tol = 1e-6)

    # regression line
    get_y = lambda a, b: a * x + b

    # y_predicted
    y_predicted = get_y(model_q.params['time'],model_q.params['Intercept'])

    # slope
    slope = model_q.params['time']

    return y_predicted, slope

################# Step (b): PROJECTION ONTO BASIS

from scipy.stats import norm
from scipy import linalg
import pandas as pd

# Defining basis

def basis_functions(qs):

    # input: array of qs. E.g., qs = np.arange(0.02, 1.00, 0.02)
    # the four basis functions
    b1 = np.array([1] * len(qs))
    b2 = (1/2)*norm.ppf(qs, loc=0, scale=1)
    b3 = (1/6)*( (norm.ppf(qs, loc=0, scale=1)**2 ) - 1)
    b4 = (1/24)*( (norm.ppf(qs, loc=0, scale=1)**3 ) - 3*norm.ppf(qs, loc=0, scale=1))

    return np.array([b1,b2,b3,b4])

# Block bootstrap a time series (trends included)
# Recombinator package: https://github.com/InvestmentSystems/recombinator/blob/master/notebooks/Block%20Bootstrap.ipynb

# Block bootstrap a time series (trends included)

def bootstrap_time_series(yt,n,block_length):

    # Input:
    #       - yt: is the variable of interest (e.g., sea level)
    #       - n:  number of blockboostrapps
    #       - block_length: values inside a block size are NOT i.i.d.

    bootstrapped_yt = moving_block_bootstrap(yt,
                                   block_length=block_length,
                                   replications=n,
                                   replace=True)

    return bootstrapped_yt

# This computes the changes in moments for a single time series

def changes_in_moments(xt,yt,block_length,n,qs,alpha):

    # Input:
    # xt: time array
    # yt: time series
    # n: number of bootstrap samples
    # qs: quantiles
    # block_length: values inside a block size are NOT i.i.d.
    # alpha: significance level

    # Output:
    # coeffs. 4 numbers: each one tells you the slope in [mean, variance, skewness, kurtosis]
    # sigs. 4 numbers: each one tells you if the slope in [mean, variance, skewness, kurtosis]
    # is significant or not

    bootstrapped_ts =  bootstrap_time_series(yt,n,block_length)

    B = np.empty((n, len(qs)))    # Initialize matrix

    # Populate quantile regression matrix
    for i in range(n):
        ys = bootstrapped_ts[i]
        slopes_bootstrap = q_regressions(xt,ys)
        B[i,:] = slopes_bootstrap

     # From bootstrapped matrix, compute coefficient matrix using the basis functions
    C = np.zeros((n, 4))

    basis = basis_functions(qs)

    for i in range(n):
        C[i, :] = linalg.lstsq(np.transpose(basis), B[i])[0]

    # Create significance intervals at alpha level:
    # significance_intervals[i] is the significance interval for the ith moment coefficient.
    # Here we look at two-tailed tests

    significance_intervals = [
        pd.Interval(np.quantile(C[:, j], alpha/2), np.quantile(C[:, j], 1-(alpha/2)),)
        for j in range(4)
    ]

    # Quantile regression of the original time series
    slopes = q_regressions(xt,yt)

    # Compute also the coefficients for the main quantile regressions
    coeffs = linalg.lstsq(np.transpose(basis), slopes)[0]
    # Compute significances
    sigs = [coeffs[i] not in significance_intervals[i] for i in range(4)]

    return coeffs, sigs, C, slopes

################# EXTRACT COASTAL POINTS FROM OCEAN MODEL.

# The function mask_coast_only takes a spatial field (e.g., sea surface temperature at time t = 0). The land is supposed to be "nan". The function outputs a new spatial field where the only non-nan points are points on the coast.

from sklearn.neighbors import NearestNeighbors

def mask_coast_only(field):

    # input:
    # spatial field [latxlon] (1 time step)
    # land is nan

    # output:
    # new mask: coastal grid points are the only non-nan points. Such points have value 1.

    ############################################# Step (a)
    # Store all indices and the one with just nan

    # Consider all indices
    indices = []
    for i in range(np.shape(field)[0]):
        for j in range(np.shape(field)[1]):
            indices.append([i,j]);
    indices = np.asarray(indices,dtype=int)
    # Consider the indices with nans
    nan_indices = np.transpose(np.where(np.isnan(field)))

    ############################################# Step (b)
    # Find the 8 nearest neighbors for each nan

    k = 8+1 # because we are counting ourselves too
    nearest_neighbors = NearestNeighbors(n_neighbors=k)
    nearest_neighbors.fit(indices)
    # coordinate of the closest nearest neighbor to a index with nan
    _,k_neighborhood_indices = nearest_neighbors.kneighbors(nan_indices)
    # Remove yourself
    k_neighborhood_indices = k_neighborhood_indices[:,1:]
    # OK, for each nan, we have information of its 8 nearest neighbors
    # nearest_neighb_indices[0] tells me nearest neighbors of nan_indices[0]
    nearest_neighb_indices = indices[k_neighborhood_indices]

    # values_nearest_neighb contains the values of data for every nearest neighbor of 
    # a masked (nan) grid point. A masked grid point is a point on land.
    # Some of these nearest neighbors will be masked too, other will actually be on the ocean (so not masked)

    values_nearest_neighb = []
    for i in range(len(nearest_neighb_indices)):
        for j in range(8): # yes, 8 nearest neighbors
            values_nearest_neighb.append(field[nearest_neighb_indices[i,j,0],nearest_neighb_indices[i,j,1]])
    values_nearest_neighb = np.asarray(values_nearest_neighb)
    values_nearest_neighb = np.reshape(values_nearest_neighb,(np.shape(nearest_neighb_indices)[0],np.shape(nearest_neighb_indices)[1]))

    ############################################# Step (c)
    # Define the mask

    # Now I define every point equal to nan
    # I will assign a value of 1 only to points at the coast
    mask = np.empty((np.shape(field)[0],np.shape(field)[1]))
    mask[:] = np.nan

    # If a land point has at least one nearest neighbor that is non-nan (ocean) 
    # I set that point equal to 1

    for i in range(len(values_nearest_neighb)):
        if np.nansum(np.abs(values_nearest_neighb[i])) > 0:
            for j in range(8):
                if ~np.isnan(values_nearest_neighb[i,j]):
                    mask[nearest_neighb_indices[i,j,0],nearest_neighb_indices[i,j,1]] = 1

    return mask

