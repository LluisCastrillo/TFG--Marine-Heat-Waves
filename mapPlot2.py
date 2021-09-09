import numpy as np
import scipy.signal as sig
from scipy import linalg
from scipy import stats
from scipy import io
from scipy import interpolate as interp
from netCDF4 import Dataset

from mpl_toolkits.axes_grid1 import make_axes_locatable

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

domain = [25, -20, 35, -5]
domain_draw = [-60, 20, 60, 380]
domain_draw = [-60, 60, 60, 380]
dlat = 5
dlon = 10
llon, llat = np.meshgrid(lon_map, lat_map)
llon_lr, llat_lr = np.meshgrid(lon_map, lat_map)
bg_col = '0.6'
cont_col = '1.0'

plt.clf()
a=plt.subplot(1,2,1, facecolor=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True])
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, MHW_cnt, levels=np.arange(1,3+0.5,0.5), cmap=plt.cm.YlOrRd)

divider = make_axes_locatable(a)
cax = divider.append_axes("right", size="5%", pad=0.05)

plt.colorbar(cax=cax)
plt.clim(0.75,3.75)
plt.title('Número de MHW medio por año')


b=plt.subplot(1,2,2, facecolor=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[False,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,False])
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, MHW_cnt_tr*10, levels=[0,0.3,0.6,0.9,1.2], cmap=plt.cm.RdBu_r)

divider = make_axes_locatable(b)
cax = divider.append_axes("right", size="5%", pad=0.05)

plt.colorbar(cax=cax)


plt.savefig('mhw_stats/MHW_cnt.pdf', bbox_inches='tight', pad_inches=0.5)


##### CAMBIAR CALIDAD (pdf), DE VERTICAL A HOR., AJUSTAR BARRAS DE COLOR, AJUSTAR ESCALAS, TÍTULOS
 
plt.clf()
c=plt.subplot(1,2,1, facecolor=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True])
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, MHW_dur, levels=[5,7,9,11,13,15,17,19,21], cmap=plt.cm.gist_heat_r)
divider = make_axes_locatable(c)
cax = divider.append_axes("right", size="5%", pad=0.05)

plt.colorbar(cax=cax)
plt.clim(8,80)
plt.title('Duración media por año [días]')

d=plt.subplot(1,2,2, facecolor=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[False,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,False])
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, MHW_dur_tr*10, levels=[-0.5,0,2.5,5,7.5,10], cmap=plt.cm.RdBu_r)
divider = make_axes_locatable(d)
cax = divider.append_axes("right", size="5%", pad=0.05)

plt.colorbar(cax=cax)
plt.clim(-18,18)


plt.savefig('mhw_stats/MHW_dur.pdf', bbox_inches='tight', pad_inches=0.5)


plt.clf()
e=plt.subplot(1,2,1, facecolor=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True])
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, MHW_max, levels=np.arange(1,4+0.5,0.5), cmap=plt.cm.gist_heat_r)
divider = make_axes_locatable(e)
cax = divider.append_axes("right", size="5%", pad=0.05)

plt.colorbar(cax=cax)
plt.clim(0.5,5.75)
plt.title('Intensidad máxima media por año [ºC]')

f=plt.subplot(1,2,2, facecolor=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[False,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,False])
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, MHW_max_tr*10, levels=[-0.75,-0.5,-0.25,-0.05,0.05,0.25,0.5,0.75], cmap=plt.cm.RdBu_r)
divider = make_axes_locatable(f)
cax = divider.append_axes("right", size="5%", pad=0.05)

plt.colorbar(cax=cax)
plt.clim(-0.5,0.5)


plt.savefig('mhw_stats/MHW_max.pdf', bbox_inches='tight', pad_inches=0.5)


plt.clf()
g=plt.subplot(1,2,1, facecolor=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True])
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, MHW_mean, levels=np.arange(1,2.5+0.3,0.3), cmap=plt.cm.gist_heat_r)
divider = make_axes_locatable(g)
cax = divider.append_axes("right", size="5%", pad=0.05)

plt.colorbar(cax=cax)
plt.clim(0.5,5.75)
plt.title('Intensidad media por año [ºC-día]')
h=plt.subplot(1,2,2, facecolor=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[False,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,False])
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, MHW_mean_tr*10, levels=[-0.5,-0.25,-0.05,0.05,0.25,0.5], cmap=plt.cm.RdBu_r)
divider = make_axes_locatable(h)
cax = divider.append_axes("right", size="5%", pad=0.05)

plt.colorbar(cax=cax)
plt.clim(-0.5,0.5)


plt.savefig('mhw_stats/MHW_mean.pdf', bbox_inches='tight', pad_inches=0.5)

plt.clf()
i=plt.subplot(1,2,1, facecolor=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True])
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, MHW_cum, levels=[10,15,20,25,30,35], cmap=plt.cm.gist_heat_r)
divider = make_axes_locatable(i)
cax = divider.append_axes("right", size="5%", pad=0.05)

plt.colorbar(cax=cax)
plt.clim(10,80)
plt.title('Intensidad acumulada media por [ºC]')

j=plt.subplot(1,2,2, facecolor=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[False,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,False])
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, MHW_cum_tr*10, levels=[-5,0,5,10,15,20], cmap=plt.cm.RdBu_r)
divider = make_axes_locatable(j)
cax = divider.append_axes("right", size="5%", pad=0.05)

plt.colorbar(cax=cax)
plt.clim(-22,22)


plt.savefig('mhw_stats/MHW_cum.pdf', bbox_inches='tight', pad_inches=0.5)

plt.clf()
k=plt.subplot(1,2,1, facecolor=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True])
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, MHW_td, levels=[15,19,23,27,31], cmap=plt.cm.gist_heat_r)
divider = make_axes_locatable(k)
cax = divider.append_axes("right", size="5%", pad=0.05)

plt.colorbar(cax=cax)
plt.clim(15,70)
plt.title('Días totales de MHW por año medios[días]')

l=plt.subplot(1,2,2, facecolor=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[False,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,False])
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, MHW_td_tr*10, levels=[-5,0,5,10,15,20,25], cmap=plt.cm.RdBu_r)
divider = make_axes_locatable(l)
cax = divider.append_axes("right", size="5%", pad=0.05)

plt.colorbar(cax=cax)
plt.clim(-30,30)


plt.savefig('mhw_stats/MHW_totDays.pdf', bbox_inches='tight', pad_inches=0.5)

plt.clf()
m=plt.subplot(1,2,1, facecolor=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True])
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, MHW_tc, levels=[20,25,30,35,40,45,50,55,60,65], cmap=plt.cm.gist_heat_r)
divider = make_axes_locatable(m)
cax = divider.append_axes("right", size="5%", pad=0.05)

plt.colorbar(cax=cax)
plt.clim(10,225)
plt.title('Intensidad acumulada anual media[ºC-día]')

n=plt.subplot(1,2,2, facecolor=bg_col)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
proj.fillcontinents(color=(0,0,0), lake_color=(0,0,0), ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[False,False,False,False])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,False])
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, MHW_tc_tr*10, levels=[0,5,10,15,20,25,30], cmap=plt.cm.RdBu_r)
divider = make_axes_locatable(n)
cax = divider.append_axes("right", size="5%", pad=0.05)

plt.colorbar(cax=cax)
plt.clim(-45,45)


plt.savefig('mhw_stats/MHW_totCum.pdf', bbox_inches='tight', pad_inches=0.5)