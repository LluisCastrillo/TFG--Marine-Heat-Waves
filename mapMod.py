'''
  Software which uses the MHW definition
  of Hobday et al. (2015) applied to 
  select SST time series around the globe
'''


# Load required modules

import numpy as np
from scipy import io
from scipy import linalg
from scipy import stats
from scipy import interpolate
from scipy import signal
from datetime import date
from netCDF4 import Dataset
import xarray as xr

from matplotlib import pyplot as plt
import mpl_toolkits.basemap as bm

import marineHeatWaves as mhw
import trendSimAR1

import ecoliver as ecj

#
# observations
#

pathroot = '/home/ocean/tfg/'
file0 = pathroot + 'rec81-91.nc'
ds = xr.open_dataset(file0)
t, dates, T, year, month, day, doy = ecj.timevector([1981, 10, 16], [1991, 12, 31])

#
# lat and lons of obs
#

fileobj = Dataset(file0, mode='r')
lon = fileobj.variables['lon'][:].astype(float)
lat = fileobj.variables['lat'][:].astype(float)
fill_value = fileobj.variables['sst']._FillValue.astype(float)
scale = fileobj.variables['sst'].scale_factor.astype(float)
offset = fileobj.variables['sst'].add_offset.astype(float)
fileobj.close()

#
# Size of mhwBlock variable
#

pathroot = '/home/ocean/tfg/'
file1 = pathroot + 'recmean_correct.nc'
dz = xr.open_dataset(file1)
st = dz['sst']
st = dz.sst

# Generate synthetic temperature time series
sst = np.zeros(len(t))
sst[0] = st.values[0] # Initial condition

for i in range(1,len(t)):
    sst[i] = st.values[i]

sst
#
# Apply Marine Heat Wave definition
#
mhws, clim = mhw.detect(t, sst)
mhwBlock = mhw.blockAverage(t, mhws)
years = mhwBlock['years_centre']
NB = len(years)
#
#
# initialize some variables
#

pctile = 90 # 90 # Percentile for calculation of MHWs
alpha = 0.05
X = len(lon)
Y = len(lat)

i_which = range(0,X)
j_which = range(0,Y)
DIM = (len(j_which), len(i_which))
SST_mean = np.NaN*np.zeros(DIM)
MHW_total = np.NaN*np.zeros(DIM)
MHW_cnt = np.NaN*np.zeros(DIM)
MHW_dur = np.NaN*np.zeros(DIM)
MHW_max = np.NaN*np.zeros(DIM)
MHW_mean = np.NaN*np.zeros(DIM)
MHW_cum = np.NaN*np.zeros(DIM)
MHW_var = np.NaN*np.zeros(DIM)
MHW_td = np.NaN*np.zeros(DIM)
MHW_tc = np.NaN*np.zeros(DIM)
SST_tr = np.NaN*np.zeros(DIM)
MHW_cnt_tr = np.NaN*np.zeros(DIM)
MHW_dur_tr = np.NaN*np.zeros(DIM)
MHW_max_tr = np.NaN*np.zeros(DIM)
MHW_mean_tr = np.NaN*np.zeros(DIM)
MHW_cum_tr = np.NaN*np.zeros(DIM)
MHW_var_tr = np.NaN*np.zeros(DIM)
MHW_td_tr = np.NaN*np.zeros(DIM)
MHW_tc_tr = np.NaN*np.zeros(DIM)
DIM2 = (len(j_which), len(i_which), 2)
SST_dtr = np.NaN*np.zeros(DIM2)
MHW_cnt_dtr = np.NaN*np.zeros(DIM2)
MHW_dur_dtr = np.NaN*np.zeros(DIM2)
MHW_max_dtr = np.NaN*np.zeros(DIM2)
MHW_mean_dtr = np.NaN*np.zeros(DIM2)
MHW_cum_dtr = np.NaN*np.zeros(DIM2)
MHW_var_dtr = np.NaN*np.zeros(DIM2)
MHW_td_dtr = np.NaN*np.zeros(DIM2)
MHW_tc_dtr = np.NaN*np.zeros(DIM2)
N_ts = np.zeros((len(j_which), len(i_which), NB))
SST_ts = np.zeros((len(j_which), len(i_which), NB))
MHW_cnt_ts = np.zeros((len(j_which), len(i_which), NB))
MHW_dur_ts = np.zeros((len(j_which), len(i_which), NB))
MHW_max_ts = np.zeros((len(j_which), len(i_which), NB))
MHW_mean_ts = np.zeros((len(j_which), len(i_which), NB))
MHW_cum_ts = np.zeros((len(j_which), len(i_which), NB))
MHW_var_ts = np.zeros((len(j_which), len(i_which), NB))
MHW_td_ts = np.zeros((len(j_which), len(i_which), NB))
MHW_tc_ts = np.zeros((len(j_which), len(i_which), NB))
lon_map =  np.NaN*np.zeros(len(i_which))
lat_map =  np.NaN*np.zeros(len(j_which))



# Theil-Sen trend function
def meanTrend_TS(mhwBlock, alpha=0.05):
    # Initialize mean and trend dictionaries
    mean = {}
    trend = {}
    dtrend = {}
#
    # Construct matrix of predictors, first column is all ones to estimate the mean,
    # second column is the time vector, equal to zero at mid-point.
    t = mhwBlock['years_centre']
    X = t-t.mean()
#
    # Loop over all keys in mhwBlock
    for key in mhwBlock.keys():
        # Skip time-vector keys of mhwBlock
        if (key == 'years_centre') + (key == 'years_end') + (key == 'years_start'):
            continue
#
        # Predictand (MHW property of interest)
        y = mhwBlock[key]
        valid = ~np.isnan(y) # non-NaN indices
#
        # Perform linear regression over valid indices
        if np.sum(~np.isnan(y)) > 0: # If at least one non-NaN value
            slope, y0, beta_lr, beta_up = stats.mstats.theilslopes(y[valid], X[valid], alpha=1-alpha)
            beta = np.array([y0, slope])
        else:
            beta_lr, beta_up = [np.nan, np.nan]
            beta = [np.nan, np.nan]
#
        # Insert regression coefficients into mean and trend dictionaries
        mean[key] = beta[0]
        trend[key] = beta[1]
#
        dtrend[key] = [beta_lr, beta_up]
#
    return mean, trend, dtrend

icnt= 0
for i in i_which:
	print (i, 'of', len(lon)-1)
	lon_map[icnt] = lon[i]
	jcnt = 0
	for j in j_which:
		lat_map[jcnt] = lat[j]
		sst = np.zeros(len(t))
		sst_ts = ds.sst.isel(lon=i, lat=j)
		sst[0]=sst_ts.values[0]
		for z in range(1,len(t)):
			sst[z] = sst_ts.values[z]
			continue
		if np.logical_not(np.isfinite(sst.sum())) + ((sst<-1).sum()>0): # check for land, ice
			jcnt += 1
			continue
		
		mhws, clim = mhw.detect(t, sst, pctile=pctile)
		mhwBlock = mhw.blockAverage(t, mhws, temp=sst)
        # Total count
		MHW_total[jcnt,icnt] = mhwBlock['count'].sum()
        # Mean and trend
		mean, trend, dtrend = mhw.meanTrend(mhwBlock)
        # Mean and trend
		MHW_cnt[jcnt,icnt], MHW_cnt_tr[jcnt,icnt], MHW_cnt_dtr[jcnt,icnt,:] = mean['count'], trend['count'], dtrend['count']
		MHW_dur[jcnt,icnt], MHW_dur_tr[jcnt,icnt], MHW_dur_dtr[jcnt,icnt,:] = mean['duration'], trend['duration'], dtrend['duration']
		MHW_max[jcnt,icnt], MHW_max_tr[jcnt,icnt], MHW_max_dtr[jcnt,icnt,:] = mean['intensity_max_max'], trend['intensity_max_max'], dtrend['intensity_max_max']
		MHW_mean[jcnt,icnt], MHW_mean_tr[jcnt,icnt], MHW_mean_dtr[jcnt,icnt,:] = mean['intensity_mean'], trend['intensity_mean'], dtrend['intensity_mean']
		MHW_cum[jcnt,icnt], MHW_cum_tr[jcnt,icnt], MHW_cum_dtr[jcnt,icnt,:] = mean['intensity_cumulative'], trend['intensity_cumulative'], dtrend['intensity_cumulative']
		MHW_var[jcnt,icnt], MHW_var_tr[jcnt,icnt], MHW_var_dtr[jcnt,icnt,:] = mean['intensity_var'], trend['intensity_var'], dtrend['intensity_var']
		MHW_td[jcnt,icnt], MHW_td_tr[jcnt,icnt], MHW_td_dtr[jcnt,icnt,:] = mean['total_days'], trend['total_days'], dtrend['total_days']
		MHW_tc[jcnt,icnt], MHW_tc_tr[jcnt,icnt], MHW_tc_dtr[jcnt,icnt,:] = mean['total_icum'], trend['total_icum'], dtrend['total_icum']
		SST_mean[jcnt,icnt], SST_tr[jcnt,icnt], SST_dtr[jcnt,icnt,:] = mean['temp_mean'], trend['temp_mean'], dtrend['temp_mean']
        # Time series
		MHW_cnt_ts[jcnt,icnt,:] += mhwBlock['count']
		MHW_dur_ts[jcnt,icnt,np.where(~np.isnan(mhwBlock['duration']))[0]] = mhwBlock['duration'][np.where(~np.isnan(mhwBlock['duration']))[0]]
		MHW_max_ts[jcnt,icnt,np.where(~np.isnan(mhwBlock['intensity_max_max']))[0]] = mhwBlock['intensity_max_max'][np.where(~np.isnan(mhwBlock['intensity_max_max']))[0]]
		MHW_mean_ts[jcnt,icnt,np.where(~np.isnan(mhwBlock['intensity_mean']))[0]] = mhwBlock['intensity_mean'][np.where(~np.isnan(mhwBlock['intensity_mean']))[0]]
		MHW_cum_ts[jcnt,icnt,np.where(~np.isnan(mhwBlock['intensity_cumulative']))[0]] = mhwBlock['intensity_cumulative'][np.where(~np.isnan(mhwBlock['intensity_cumulative']))[0]]
		MHW_var_ts[jcnt,icnt,np.where(~np.isnan(mhwBlock['intensity_var']))[0]] = mhwBlock['intensity_var'][np.where(~np.isnan(mhwBlock['intensity_var']))[0]]
		MHW_td_ts[jcnt,icnt,np.where(~np.isnan(mhwBlock['total_days']))[0]] = mhwBlock['total_days'][np.where(~np.isnan(mhwBlock['total_days']))[0]]
		MHW_tc_ts[jcnt,icnt,np.where(~np.isnan(mhwBlock['total_icum']))[0]] = mhwBlock['total_icum'][np.where(~np.isnan(mhwBlock['total_icum']))[0]]
		N_ts[jcnt,icnt,:] += (~np.isnan(mhwBlock['duration'])).astype(int)
		SST_ts[jcnt,icnt,:] = mhwBlock['temp_mean']

        # Up counts
		jcnt += 1
	icnt += 1
    # Save data so far
   
	
	outfile = '/home/ocean/tfg/mhw_stats/census_mod'
       
	if (i % 100) + (i == i_which[-1]):
		np.savez(outfile, lon_map=lon_map, lat_map=lat_map, SST_mean=SST_mean, MHW_total=MHW_total, MHW_cnt=MHW_cnt, MHW_dur=MHW_dur, MHW_max=MHW_max, MHW_mean=MHW_mean, MHW_cum=MHW_cum, MHW_var=MHW_var, MHW_td=MHW_td, MHW_tc=MHW_tc, SST_tr=SST_tr, MHW_cnt_tr=MHW_cnt_tr, MHW_dur_tr=MHW_dur_tr, MHW_max_tr=MHW_max_tr, MHW_mean_tr=MHW_mean_tr, MHW_cum_tr=MHW_cum_tr, MHW_var_tr=MHW_var_tr, MHW_td_tr=MHW_td_tr, MHW_tc_tr=MHW_tc_tr, SST_dtr=SST_dtr, MHW_cnt_dtr=MHW_cnt_dtr, MHW_dur_dtr=MHW_dur_dtr, MHW_max_dtr=MHW_max_dtr, MHW_mean_dtr=MHW_mean_dtr, MHW_cum_dtr=MHW_cum_dtr, MHW_var_dtr=MHW_var_dtr, MHW_td_dtr=MHW_td_dtr, MHW_tc_dtr=MHW_tc_dtr, SST_ts=SST_ts, MHW_cnt_ts=MHW_cnt_ts, MHW_dur_ts=MHW_dur_ts, MHW_max_ts=MHW_max_ts, MHW_mean_ts=MHW_mean_ts, MHW_cum_ts=MHW_cum_ts, MHW_var_ts=MHW_var_ts, MHW_td_ts=MHW_td_ts, MHW_tc_ts=MHW_tc_ts, N_ts=N_ts, years=years, alpha=alpha)
