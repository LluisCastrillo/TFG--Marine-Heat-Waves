
import numpy as np
import scipy.signal as sig
from scipy import linalg
from scipy import stats
from scipy import io
from scipy import interpolate as interp
from netCDF4 import Dataset

import ecoliver as ecj

from matplotlib import pyplot as plt
import mpl_toolkits.basemap as bm


#
# Load data and make plots
#
outfile = '/home/ocean/tfg/mhw_stats/census_mod'

data = np.load(outfile+'.npz')
lon_map = data['lon_map']
lat_map = data['lat_map']
SST_mean = data['SST_mean']
MHW_total = data['MHW_total']
MHW_cnt = data['MHW_cnt']
MHW_dur = data['MHW_dur']
MHW_max = data['MHW_max']
MHW_mean = data['MHW_mean']
MHW_cum = data['MHW_cum']
MHW_td = data['MHW_td']
MHW_tc = data['MHW_tc']
SST_tr = data['SST_tr']
MHW_cnt_tr = data['MHW_cnt_tr']
MHW_dur_tr = data['MHW_dur_tr']
MHW_max_tr = data['MHW_max_tr']
MHW_mean_tr = data['MHW_mean_tr']
MHW_cum_tr = data['MHW_cum_tr']
MHW_td_tr = data['MHW_td_tr']
MHW_tc_tr = data['MHW_tc_tr']
SST_dtr = data['SST_dtr']
MHW_cnt_dtr = data['MHW_cnt_dtr']
MHW_dur_dtr = data['MHW_dur_dtr']
MHW_max_dtr = data['MHW_max_dtr']
MHW_mean_dtr = data['MHW_mean_dtr']
MHW_cum_dtr = data['MHW_cum_dtr']
MHW_td_dtr = data['MHW_td_dtr']
MHW_tc_dtr = data['MHW_tc_dtr']
N_ts = data['N_ts']
years = data['years']
SST_ts = data['SST_ts']
MHW_cnt_ts = data['MHW_cnt_ts']
MHW_dur_ts = data['MHW_dur_ts']
MHW_max_ts = data['MHW_max_ts']
MHW_mean_ts = data['MHW_mean_ts']
MHW_cum_ts = data['MHW_cum_ts']
MHW_td_ts = data['MHW_td_ts']
MHW_tc_ts = data['MHW_tc_ts']




# Maps

domain = [27, -15, 30, -10]
domain_draw = [-60, 20, 60, 380]
domain_draw = [-60, 60, 60, 380]
dlat = 30
dlon = 60
llon, llat = np.meshgrid(lon_map, lat_map)
llon_lr, llat_lr = np.meshgrid(lon_map, lat_map)
bg_col = '0.6'
cont_col = '1.0'

plt.clf()
plt.subplot(2,1,1, facecolor=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True])
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, MHW_cnt, levels=np.arange(0.5,3.5+0.5,0.5), cmap=plt.cm.YlOrRd)
plt.colorbar()
plt.clim(0.75,3.75)
plt.title('Count')


plt.subplot(2,1,2, facecolor=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True])
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, MHW_cnt_tr*10, levels=[-4.5,-4,-3.5,-2,-1,-0.5,0.5,1,2,3.5,4,4.5,5,5.5], cmap=plt.cm.RdBu_r)
plt.colorbar()
plt.title('Count trend')

plt.savefig('mhw_stats/MHW_cnt.png', bbox_inches='tight', pad_inches=0.5)

plt.clf()
plt.subplot(2,1,1, facecolor=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True])
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, MHW_dur, levels=[5,10,15,20,30,60], cmap=plt.cm.gist_heat_r)
plt.colorbar()
plt.clim(8,80)
plt.title('Duration')
plt.subplot(2,1,2, facecolor=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True])
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, MHW_dur_tr*10, levels=[-50,-20,-10,-2.5,2.5,10,20,50], cmap=plt.cm.RdBu_r)
plt.colorbar()
plt.clim(-18,18)
plt.title('Duration trend')

plt.savefig('mhw_stats/MHW_dur.png', bbox_inches='tight', pad_inches=0.5)


plt.clf()
plt.subplot(2,1,1, facecolor=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True])
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, MHW_max, levels=np.arange(0,5+0.5,0.5), cmap=plt.cm.gist_heat_r)
plt.colorbar()
plt.clim(0.5,5.75)
plt.title('Intensity max')
plt.subplot(2,1,2, facecolor=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True])
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, MHW_max_tr*10, levels=[-0.6,-0.5,-0.25,-0.05,0.05,0.25,0.5,0.6], cmap=plt.cm.RdBu_r)
plt.colorbar()
plt.clim(-0.5,0.5)
plt.title('Intensity max trend')

plt.savefig('mhw_stats/MHW_max.png', bbox_inches='tight', pad_inches=0.5)


plt.clf()
plt.subplot(2,1,1, facecolor=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True])
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, MHW_mean, levels=np.arange(0,5+0.5,0.5), cmap=plt.cm.gist_heat_r)
plt.colorbar()
plt.clim(0.5,5.75)
plt.title('Intensity mean')
plt.subplot(2,1,2, facecolor=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True])
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, MHW_mean_tr*10, levels=[-0.6,-0.5,-0.25,-0.05,0.05,0.25,0.5,0.6], cmap=plt.cm.RdBu_r)
plt.colorbar()
plt.clim(-0.5,0.5)
plt.title('Intensity mean trend')

plt.savefig('mhw_stats/MHW_mean.png', bbox_inches='tight', pad_inches=0.5)

plt.clf()
plt.subplot(2,1,1, facecolor=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True])
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, MHW_cum, levels=[0,10,20,30,40,80,160], cmap=plt.cm.gist_heat_r)
plt.colorbar()
plt.clim(10,80)
plt.title('Intensity cumulative')
plt.subplot(2,1,2, facecolor=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True])
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, MHW_cum_tr*10, levels=[-100,-20,-10,-2.5,2.5,10,20,100], cmap=plt.cm.RdBu_r)
plt.colorbar()
plt.clim(-22,22)
plt.title('Intensity cumulative trend')

plt.savefig('mhw_stats/MHW_cum.png', bbox_inches='tight', pad_inches=0.5)

plt.clf()
plt.subplot(2,1,1, facecolor=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True])
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, MHW_td, levels=[10,15,20,25,30,35,40,45], cmap=plt.cm.gist_heat_r)
plt.colorbar()
plt.clim(15,70)
plt.title('Annual total MHW days [days]')
plt.subplot(2,1,2, facecolor=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True])
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, MHW_td_tr*10, levels=[-200,-30,-20,-10,-5,5,10,20,30,200], cmap=plt.cm.RdBu_r)
plt.colorbar()
plt.clim(-30,30)
plt.title('Trend [per decade]')

plt.savefig('mhw_stats/MHW_totDays.png', bbox_inches='tight', pad_inches=0.5)

plt.clf()
plt.subplot(2,1,1, facecolor=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True])
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, MHW_tc, levels=[0,10,20,30,40,60,80,100,200], cmap=plt.cm.gist_heat_r)
plt.colorbar()
plt.clim(10,225)
plt.title('Annual total cumulative intensity [deg.C-day]')
plt.subplot(2,1,2, facecolor=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True])
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, MHW_tc_tr*10, levels=[-200,-40,-20,-10,-5,5,10,20,40,200], cmap=plt.cm.RdBu_r)
plt.colorbar()
plt.clim(-45,45)
plt.title('Trend [per decade]')

plt.savefig('mhw_stats/MHW_totCum.png', bbox_inches='tight', pad_inches=0.5)