# Script to compute Quantile Regression of a time series

# Load the library
#library(quantreg)
suppressPackageStartupMessages(library(quantreg))

# Quantile regression for a single quantile
qr_function <- function(df,q){
  # df is the dataframe
  # q is the respective quantile

  slope = coef(rq(formula = y ~ x, tau = q, data = df))[2]

  return(slope)
}

# Quantile regression for a multiple quantiles using a for loop
multiple_qr_for_loop <- function(df){
  # df is the dataframe
  # These are the quantiles considered
  qs <- c(0.01, 0.05, 0.1 , 0.15, 0.2 , 0.25, 0.3 , 0.35, 0.4 , 0.45, 0.5 ,
       0.55, 0.6 , 0.65, 0.7 , 0.75, 0.8 , 0.85, 0.9 , 0.95, 0.99)

  # Create a vector where we'll store the slopes
  slopes_quantiles = 1:length(qs)
  for( i in 1:length(qs)){
    fit = rq(formula = y ~ x, tau = qs[i], data = df, method = 'pfn')
    coefficients <- coef(fit)
    slope = coefficients[2]
    slopes_quantiles[i] = slope
    }

  return(slopes_quantiles)
}

# Quantile regression for multiple quantiles using built-in function (no loops)
# This is the PREFERRED WAY.
multiple_qr <- function(df){
  # df is the dataframe
  # These are the quantiles considered
  qs <- c(0.05, 0.1 , 0.15, 0.2 , 0.25, 0.3 , 0.35, 0.4 , 0.45, 0.5 ,
       0.55, 0.6 , 0.65, 0.7 , 0.75, 0.8 , 0.85, 0.9 , 0.95)
  
  # Create a vector where we'll store the slopes
  slopes_quantiles = 1:length(qs)
  intercepts_quantiles = 1:length(qs)
  
  # Create a vector where we'll store the slopes
  fit = rq(formula = y ~ x, tau = c(0.05, 0.1 , 0.15, 0.2 , 0.25, 0.3 , 0.35, 0.4 , 0.45, 0.5 ,
                                      0.55, 0.6 , 0.65, 0.7 , 0.75, 0.8 , 0.85, 0.9 , 0.95), data = df, method = 'pfnb')

  coefficients <- coef(fit)
  slopes = coefficients[2,]
  # Save the slopes coefficients
  slopes_quantiles[] = slopes[]
  
  return(slopes_quantiles)
}

# Resampling methods in R
# Give a look at the Boot Package
# https://cran.r-project.org/doc/Rnews/Rnews_2002-3.pdf


