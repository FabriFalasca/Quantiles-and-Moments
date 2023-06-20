# Exploring the non-stationarity of coastal sea level probability distributions

Fabrizio Falasca, Andrew Brettin, Laure Zanna, Stephen M. Griffies, Jianjun Yin, Ming Zhao. Paper in https://doi.org/10.1017/eds.2023.10

Contact: Fabrizio Falasca (fabri.falasca@nyu.edu)

We propose a framework to study of changes in shapes of probability distributions directly from time series data. The framework first quantifies linear trends in quantiles through quantile regression. Quantile slopes are then projected onto a set of four orthogonal polynomials quantifying how such changes can be explained by independent shifts in the first four statistical moments. The framework proposed is theoretically founded, general and can be applied to any climate observable with close-to-linear changes in distributions.

Observational dataset. We focus on daily-resolved, Tide Gauges records downloaded from the University of Hawaii Sea Level Center (https://uhslc.soest.hawaii.edu/). Model dataset. We focus on the fully coupled GFDL-CM4 model. 

## Main package 

- The quantile regression step is coded in R and called by the Python script through the rpy2 package (https://rpy2.github.io/). 
Coding the quantile regression in R allows us to leverage new results in quantile regression estimation (see https://link.springer.com/content/pdf/10.1007/s00181-020-01898-0.pdf). 

This results in a huge speed up compared to the python code. Do not install the last version of rpy2. We installed by: pip install rpy2==3.4.5

- Quantile regression in R: https://cran.r-project.org/web/packages/quantreg/vignettes/rq.pdf
To speed up the computation we adopt the method "pfnb" described in https://cran.r-project.org/web/packages/quantreg/quantreg.pdf and proposed in https://link.springer.com/content/pdf/10.1007/s00181-020-01898-0.pdf

- You can download R here: https://cran.r-project.org/
Packages to install: quantreg
  - To install a package type "R" to open a R-terminal. Then:
    install.packages("quantreg")

## In this repository:

- main/main_code

Main framework with examples on 1 time series.  In the notebook main.ipynb we present step by step each component of the proposed framework. Given a time series generated from a nonstationary Beta distribution we show how to (a) compute and visualize trends of few quantiles (e.g., q = 0.05, 0.5 and 0.95), (b) quantify independent changes in moments by (b.1) quantile regression and (b.2) projection onto Hermite polynomials, (c) analyze the statistical significance of such changes.

This code allows to infer and study changes in probability for any time series.

- main/GFDL-CM4

When considering many time series we face a multiple testing problem and more and more false positives are expected in the statistical inference. To control the ratio of false positives we adopt the False Discovery Rate (FDR) test proposed by Benjamini and Hochberg [1995]. We show how to perform this simple step in the case of time series in the GFDL-CM4 model

-- main/GFDL-CM4/code_model

In main.py we show the code used to infer changes in moments in the GFDL-CM4 model in the case of a climate change experiment. Importantly, in utils.py we present a simple algorithm to extract only the ocean points neighoboring the coast. This allows to consider only coastal sea level and remove the open ocean from the analysis.

-- main/GFDL-CM4/analysis

Given the slopes in moments and their bootstrapped distribution computed in main/GFDL-CM4/code_model, we show the main steps to analyze such data. This mainly involve (a) computation of p-values from the bootstrapped distributions, (b) inference of statistical significance given the False Discovery Rate proposed in Benjamini and Hochberg [1995] and (c) plotting the results.

The codes in folder "main" allows to reproduce all results in the paper. Moreover, they can be used to study changes in probability distributions for general climate variables.

- Exploring_CF_Expansion

Here we explore the Cornish Fisher expansion. Importantly, we show how shifts in quantiles using the polynomials in Eq. 8 in https://arxiv.org/abs/2211.04608 results in changes in independent changes in the first moments. This analysis is shown in Section "From a Quantile function to PDF".



