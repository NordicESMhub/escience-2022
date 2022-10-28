#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 18:01:44 2022

@author: adag

xyz file naming convention
vgpm.yyyyddd.all.xyz.gz

vgpm = net primary production (units of mg C / m**2 / day) based on the standard vgpm algorithm
yyyy = year
ddd = day of year of the start of each monthly file
all = all pixels output, including those with no data (nodata = -9999)
xyz = text file, where x = longitude, y = latitude and z = npp value
gz = compressed with gzip (uncompress with gunzip)

"""
import matplotlib.pyplot as plt
import cmocean.cm as cmo
import numpy as np
import xarray as xr
import pandas as pd
import cartopy.crs as ccrs

if __name__ == "__main__":
    path = '/home/adag/Documents/eScience/MODIS/'
    file = 'vgpm.2002182.all.xyz'
    # loop through filenames and concate by month
    daysinmonth = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    firstdayinmonth = np.cumsum(np.append(1,daysinmonth[:-1]))
    year = 2003
    
    var = "value"
    monthnr = 1
    df = None
    for day in firstdayinmonth:
        date = '%i-%s-01'%(year, str.zfill(str(monthnr),2))
        file = 'vgpm.%i%s.all.xyz'%(year, str.zfill(str(day),3))
        tmp =  pd.read_table(path + file, skiprows=2, delim_whitespace=True,
                         names=['lon', 'lat', 'value'])
        tmp = tmp.replace(-9999.0,np.nan)
        tmp['time'] = pd.Timestamp(date)
        # extract Nordic Seas and Barents Sea, comment out if you want global data
        tmp = tmp[(tmp['lat'] > 60 ) & (tmp['lon'] > -25) & (tmp['lon'] < 50) ]
        
        if isinstance(tmp, pd.core.frame.DataFrame):
            df = pd.concat([df, tmp])
        else:
           df = tmp
        monthnr = monthnr+1
#%% convert to xarray
    ds = df.to_xarray()
    ds =ds.assign_coords(
     {"index": pd.MultiIndex.from_arrays([ds.time.values, ds.lat.values, ds.lon.values], 
                                                      names=["time","lat", "lon"])}).unstack("index").rename({'value':'npp'})
#%%
    ds = ds.npp.groupby('time.month').mean(dim='time')
    x,y = np.meshgrid(ds.lon.values, ds.lat.values)
    vmin = 0
    vmax = 4000
    dval = 250
    levels = np.arange(vmin, vmax + dval, dval)
    fig= plt.figure(figsize=(9,7))
    fig.suptitle('MODIS: net primary production (NPP) based on the standard vgpm algorithm')
    for monthnr in ds.month.values:
        da = ds.sel(month=monthnr)
        ax = plt.subplot(3, 4, monthnr, projection=ccrs.Orthographic(0, 90))
        cf = ax.contourf(x, y, da.values,
         colormap=cmo.haline, transform=ccrs.PlateCarree(), 
         vmin = vmin, vmax = vmax, levels = levels, extend="max")
        ax.coastlines(linewidth=0.5)
        ax.set_extent([-25, 50, 60, 90], ccrs.PlateCarree())
        ax.gridlines(linewidth=0.5, ylocs = [60, 70, 80], linestyle='--')
        ax.set_title('%i-%s-01'%(year, str.zfill(str(monthnr),2)), fontsize=11)
    cax = fig.add_axes([0.3, 0.08, 0.4, 0.02])
    cb = plt.colorbar(cf, cax=cax, ticks=levels[::4], orientation="horizontal")
    cb.set_label("net primary production (mg C / m² / day)", fontsize=10)
    plt.subplots_adjust(left=0.05, bottom=0.12, right=.95, top=0.92, hspace=0.15, wspace = 0.)
   #plt.savefig('figures/NPP_MODIS_%s.png'%year)
  