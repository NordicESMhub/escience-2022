#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 16:30:24 2021

@author: Ada Gjermundsen

General python functions for analyzing NorESM data 

"""
import xarray as xr
import numpy as np
import warnings

warnings.simplefilter("ignore")


def consistent_naming(ds):
    """
    The naming convention for coordinates and dimensions are not the same 
    for noresm raw output and cmorized variables. This function rewrites the 
    coords and dims names to be consistent and the functions thus work on all
    Choose the cmor naming convention.

    Parameters
    ----------
    ds : xarray.Dataset 

    Returns
    -------
    ds : xarray.Dataset

    """
    if "latitude" in ds.coords and "lat" not in ds.coords:
        ds = ds.rename({"latitude": "lat"})
    if "longitude" in ds.coords and "lon" not in ds.coords:
        ds = ds.rename({"longitude": "lon"})
    if "region" in ds.dims:
        ds = ds.rename(
            {"region": "basin"}
        )  # are we sure that it is the dimension and not the variable which is renamed? Probably both
        # note in BLOM raw, region is both a dimension and a variable. Not sure how xarray handles that
        # in cmorized variables sector is the char variable with the basin names, and basin is the dimension
        # don't do this -> ds = ds.rename({'region':'sector'})
    if "x" in ds.dims:
        ds = ds.rename({"x": "i"})
    if "y" in ds.dims:
        ds = ds.rename({"y": "j"})
    if "ni" in ds.dims:
        ds = ds.rename({"ni": "i"})
    if "nj" in ds.dims:
        ds = ds.rename({"nj": "j"})
    if "nlat" in ds.dims:
        ds = ds.rename({"nlat": "j"})
    if "nlon" in ds.dims:
        ds = ds.rename({"nlon": "i"})
    if "depth" in ds.dims:
        ds = ds.rename({"depth": "lev"})
    if "nbnd" in ds.dims:
        ds = ds.rename({"nbnd": "bnds"})
    if "nbounds" in ds.dims:
        ds = ds.rename({"nbounds": "bnds"})
    if "bounds" in ds.dims:
        ds = ds.rename({"bounds": "bnds"})
    if "type" in ds.coords:
        ds = ds.drop("type")
    if 'latitude_bnds' in ds.variables:
        ds = ds.rename({'latitude_bnds':'lat_bnds'})
    if 'longitude_bnds' in ds.variables:
        ds = ds.rename({'longitude_bnds':'lon_bnds'})
    if 'nav_lat' in ds.coords:
        ds = ds.rename({'nav_lon':'lon','nav_lat':'lat'})
    if 'bounds_nav_lat' in ds.variables:
        ds = ds.rename({'bounds_nav_lat':'vertices_latitude','bounds_nav_lon':'vertices_longitude'})
    return ds


def global_avg(ds):
    """Calculates globally averaged values

    Parameters
    ----------
    ds : xarray.DataArray i.e. ds[var]

    Returns
    -------
    ds_out :  xarray.DataArray with globally averaged values
    """
    # to include functionality for subsets or regional averages:
    if "time" in ds.dims:
        weights = xr.ufuncs.cos(xr.ufuncs.deg2rad(ds.lat)) * ds.notnull().mean(dim=("lon", "time"))
    elif "month" in ds.dims:
        weights = xr.ufuncs.cos(xr.ufuncs.deg2rad(ds.lat)) * ds.notnull().mean(dim=("lon", "month"))
    else:
        weights = xr.ufuncs.cos(xr.ufuncs.deg2rad(ds.lat)) * ds.notnull().mean(dim=("lon"))
    ds_out = (ds.mean(dim="lon") * weights).sum(dim="lat") / weights.sum()
    if "long_name" in ds.attrs:
        ds_out.attrs["long_name"] = "Globally averaged " + ds.long_name
    if "units" in ds.attrs:
        ds_out.attrs["units"] = ds.units
    if "standard_name" in ds.attrs:
        ds_out.attrs["standard_name"] = ds.standard_name
    return ds_out


def fix_cam_time(ds):
    """ NorESM raw CAM h0 files have incorrect time variable output,
    thus it is necessary to use time boundaries to get the correct time
    If the time variable is not corrected, none of the functions involving time
    e.g. yearly_avg, seasonal_avg etc. will provide correct information

    Parameters
    ----------
    ds : xarray.Dataset 

    Returns
    -------
    ds_weighted : xarray.Dataset with corrected time
    """
    from cftime import DatetimeNoLeap

    months = ds.time_bnds.isel(bnds=0).dt.month.values
    years = ds.time_bnds.isel(bnds=0).dt.year.values
    dates = [DatetimeNoLeap(year, month, 15) for year, month in zip(years, months)]
    ds = ds.assign_coords(time=dates)
    return ds


def fix_time(ds, yr0=1850):
    """
    If there are problems with the calender used in the cmorized files (e.g. GFDL-ESM4)
    This function will overwrite the time array such that (all) other functions can be used
    Not needed for NorESM analysis in general, but e.g. in feedback analysis both kernels and
    model dataset need to use the same time  
    
    """
    from itertools import product
    from cftime import DatetimeNoLeap

    yr = np.int(ds.time.shape[0] / 12)
    yr1 = yr + yr0
    dates = [DatetimeNoLeap(year, month, 16) for year, month in product(range(yr0, yr1), range(1, 13))]
    bounds_a = [DatetimeNoLeap(year, month, 1) for year, month in product(range(yr0, yr1), range(1, 13))]
    bounds_b = bounds_a[1:]
    bounds_b.append(DatetimeNoLeap(yr1, 1, 1))
    bounds = np.reshape(np.concatenate([bounds_a, bounds_b]), [ds.time.shape[0], 2])
    ds = ds.assign_coords(time=dates)
    # set attributes
    ds["time"].attrs["bounds"] = "time_bnds"
    ds["time"].attrs["axis"] = "T"
    ds["time"].attrs["long_name"] = "time"
    ds["time"].attrs["standard_name"] = "time"
    # ds['time'].attrs['cell_methods'] = 'time: mean'
    ds["time"].attrs["calendar"] = "noleap"
    ds["time"].attrs["units"] = "days since %04d-01-16 00:00" % yr0
    ds["time_bnds"] = xr.DataArray(bounds, dims=("time", "bnds"))
    ds["time_bnds"].attrs["axis"] = "T"
    ds["time_bnds"].attrs["long_name"] = "time bounds"
    ds["time_bnds"].attrs["standard_name"] = "time_bnds"
    return ds


def yearly_avg(ds):
    """ Calculates timeseries over yearly averages from timeseries of monthly means
    The weighted average considers that each month has a different number of days.

    Parameters
    ----------
    ds : xarray.DataArray i.e. ds[var]

    Returns
    -------
    ds_weighted : xarray.DataArray with yearly averaged values

    """
    # Note! NorESM raw CAM h0 files have wrong time variable, necessary to use time boundaries to
    # get the correct time
    month_length = ds.time.dt.days_in_month
    weights = month_length.groupby("time.year") / month_length.groupby("time.year").sum()
    # Test that the sum of the weights for each year is 1.0
    np.testing.assert_allclose(weights.groupby("time.year").sum().values, np.ones(len(np.unique(ds.time.dt.year))))
    # Calculate the weighted average:
    ds_weighted = (ds * weights).groupby("time.year").sum(dim="time")
    if "long_name" in ds.attrs:
        ds_weighted.attrs["long_name"] = "Annual mean " + ds.long_name
    if "units" in ds.attrs:
        ds_weighted.attrs["units"] = ds.units
    if "standard_name" in ds.attrs:
        ds_weighted.attrs["standard_name"] = ds.standard_name

    return ds_weighted


def seasonal_avg_timeseries(ds, var=""):
    """Calculates timeseries over seasonal averages from timeseries of monthly means
    The weighted average considers that each month has a different number of days.
    Using 'QS-DEC' frequency will split the data into consecutive three-month periods, 
    anchored at December 1st. 
    I.e. the first value will contain only the avg value over January and February 
    and the last value only the December monthly averaged value
    
    Parameters
    ----------
    ds : xarray.DataArray i.e.  ds[var]
        
    Returns
    -------
    ds_out: xarray.DataSet with 4 timeseries (one for each season DJF, MAM, JJA, SON)
            note that if you want to include the output in an other dataset, e.g. dr,
            you should use xr.merge(), e.g.
            dr = xr.merge([dr, seasonal_avg_timeseries(dr[var], var)])
    """
    month_length = ds.time.dt.days_in_month
    sesavg = (ds * month_length).resample(time="QS-DEC").sum() / month_length.where(ds.notnull()).resample(
        time="QS-DEC"
    ).sum()
    djf = sesavg[0::4].to_dataset(name=var + "_DJF").rename({"time": "time_DJF"})
    mam = sesavg[1::4].to_dataset(name=var + "_MAM").rename({"time": "time_MAM"})
    jja = sesavg[2::4].to_dataset(name=var + "_JJA").rename({"time": "time_JJA"})
    son = sesavg[3::4].to_dataset(name=var + "_SON").rename({"time": "time_SON"})
    ds_out = xr.merge([djf, mam, jja, son])
    if "long_name" in ds.attrs:
        ds_out.attrs["long_name"] = "Seasonal mean " + ds.long_name
    if "units" in ds.attrs:
        ds_out.attrs["units"] = ds.units
    if "standard_name" in ds.attrs:
        ds_out.attrs["standard_name"] = ds.standard_name
    return ds_out


def seasonal_avg(ds):
    """Calculates seasonal averages from timeseries of monthly means
    The time dimension is reduced to 4 seasons: 
        * season   (season) object 'DJF' 'JJA' 'MAM' 'SON'
    The weighted average considers that each month has a different number of days.
    
    Parameters
    ----------
    ds : xarray.DataArray i.e.  ds[var]
        
    Returns
    -------
    ds_weighted : xarray.DataArray 
    """
    month_length = ds.time.dt.days_in_month
    # Calculate the weights by grouping by 'time.season'.
    weights = month_length.groupby("time.season") / month_length.groupby("time.season").sum()
    # Test that the sum of the weights for each season is 1.0
    np.testing.assert_allclose(weights.groupby("time.season").sum().values, np.ones(4))
    # Calculate the weighted average
    ds_weighted = (ds * weights).groupby("time.season").sum(dim="time")
    if "long_name" in ds.attrs:
        ds_weighted.attrs["long_name"] = "Seasonal mean " + ds.long_name
    if "units" in ds.attrs:
        ds_weighted.attrs["units"] = ds.units
    if "standard_name" in ds.attrs:
        ds_weighted.attrs["standard_name"] = ds.standard_name
    return ds_weighted


def mask_region_latlon(ds, lat_low=-90, lat_high=90, lon_low=0, lon_high=360):
    """Subtract data from a confined region
    Note, for the atmosphere the longitude values go from 0 -> 360.
    Also after regridding
    This is not the case for ice and ocean variables for which some cmip6 models
    use -180 -> 180
    
    Parameters
    ----------
    ds : xarray.DataArray or xarray.DataSet
    lat_low : int or float, lower latitude boudary. The default is -90.
    lat_high : int or float, lower latitude boudary. The default is 90.
    lon_low :  int or float, East boudary. The default is 0.
    lon_high : int or float, West boudary. The default is 360.
    
    Returns
    -------
    ds_out : xarray.DataArray or xarray.DataSet with data only for the selected region
    
    Then it is still possible to use other functions e.g. global_mean(ds) to 
    get an averaged value for the confined region 
    """
    ds_out = ds.where((ds.lat >= lat_low) & (ds.lat <= lat_high))
    if lon_high > lon_low:
        ds_out = ds_out.where((ds_out.lon >= lon_low) & (ds_out.lon <= lon_high))
    else:
        boole = (ds_out.lon.values <= lon_high) | (ds_out.lon.values >= lon_low)
        ds_out = ds_out.sel(lon=ds_out.lon.values[boole])
    if "long_name" in ds.attrs:
        ds_out.attrs["long_name"] = (
            "Regional subset (%i,%i,%i,%i) of " % (lat_low, lat_high, lon_low, lon_high) + ds.long_name
        )
    if "units" in ds.attrs:
        ds_out.attrs["units"] = ds.units
    if "standard_name" in ds.attrs:
        ds_out.attrs["standard_name"] = (
            "Regional subset (%i,%i,%i,%i) of " % (lat_low, lat_high, lon_low, lon_high) + ds.standard_name
        )
    return ds_out


def get_areacello(cmor=True):
    """
    only if pweight / areacello is not provided. Only works for 1deg ocean (CMIP6) as it is now
    """
    if not cmor:
        try:
            # This path only works on FRAM and BETZY
            grid = xr.open_dataset("/cluster/shared/noresm/inputdata/ocn/blom/grid/grid_tnx1v4_20170622.nc")
            pweight = grid.parea * grid.pmask.where(grid.pmask > 0)
            # add latitude and longitude info good to have in case of e.g. regridding
            pweight = pweight.assign_coords(lat=grid.plat)
            pweight = pweight.assign_coords(lon=grid.plon)
        except:
            # on NIRD there is no common place for blom grid files, only to the cmorized files
            cmor = True
    if cmor:
        # This path only works on NIRD. NS9034 is not mounted in /trd-projects* so impossible to add a general path
        area = xr.open_dataset(
            "/projects/NS9034K/CMIP6/CMIP/NCC/NorESM2-MM/piControl/r1i1p1f1/Ofx/areacello/gn/latest/areacello_Ofx_NorESM2-MM_piControl_r1i1p1f1_gn.nc"
        )
        mask = xr.open_dataset(
            "/projects/NS9034K/CMIP6/CMIP/NCC/NorESM2-MM/piControl/r1i1p1f1/Ofx/sftof/gn/latest/sftof_Ofx_NorESM2-MM_piControl_r1i1p1f1_gn.nc"
        )
        pweight = area.areacello * mask.sftof
    pweight = consistent_naming(pweight.to_dataset(name='pweight'))
    return pweight.pweight


def select_month(ds, monthnr):
    """ Calulates timeseries for one given month 

    Parameters
    ----------
    ds : xarray.DataArray i.e. ds[var]
    monthnr: int , nr of month 1 = January, 2 = February etc.
    
    Returns
    -------
    ds : xarray.DataArray with single month values
    """
    ds = ds.sel(time=ds.time.dt.month == monthnr)
    return ds

def lat_lon_bnds(ds):
    """ This function calculates and add longitude and latitude boundaries to dataset

    Parameters
    ----------
    ds : xarray.Dataset with lat, lon dimensions
    
    Returns
    -------
    ds : xarray.Dataset same as input Dataset, but with lat_bnd and lon_bnd added as DataArray(s)
    """
    lon_b = np.concatenate(
        (
            np.array(
                [ds.lon[0].values - 0.5 * ds.lon.diff(dim="lon").values[0]]
            ),
            0.5 * ds.lon.diff(dim="lon").values + ds.lon.values[:-1],
            np.array(
                [ds.lon[-1].values + 0.5 * ds.lon.diff(dim="lon").values[-1]]
            ),
        )
    )
    lon_b = np.reshape(
        np.concatenate([lon_b[:-1], lon_b[1:]]), [2, len(ds.lon.values)]
    ).T
    lon_b = xr.DataArray(lon_b, dims=("lon", "bnds"), coords={"lon": ds.lon})
    lon_b.attrs["units"] = "degrees_east"
    lon_b.attrs["axis"] = "X"
    lon_b.attrs["bounds"] = "lon_bnds"
    lon_b.attrs["standard_name"] = "longitude_bounds"
    lon_b.attrs["long_name"] = "Longitude bounds"
    ds["lon_bnds"] = xr.DataArray(
        lon_b, dims=("lon", "bnds"), coords={"lon": ds.lon}
    )

    lat_b = np.concatenate(
        (
            np.array([-90], float),
            0.5 * ds.lat.diff(dim="lat").values + ds.lat.values[:-1],
            np.array([90], float),
        )
    )
    lat_b = np.reshape(
        np.concatenate([lat_b[:-1], lat_b[1:]]), [2, len(ds.lat.values)]
    ).T
    lat_b = xr.DataArray(lat_b, dims=("lat", "bnds"), coords={"lat": ds.lat})
    lat_b = lat_b.where(lat_b < 90.0, 90.0)
    lat_b = lat_b.where(lat_b > -90.0, -90.0)
    lat_b.attrs["long_name"] = "Latitude bounds"
    lat_b.attrs["units"] = "degrees_north"
    lat_b.attrs["axis"] = "Y"
    lat_b.attrs["bounds"] = "lat_bnds"
    lat_b.attrs["standard_name"] = "latitude_bounds"
    ds["lat_bnds"] = lat_b
    return ds



def sea_ice_ext(ds, pweight=None, cmor=True):
    """ 
    Calculates the sea ice extent from the sea ice concentration fice in BLOM raw output and siconc in cmorized files
    Sea ice concentration (fice or siconc) is the percent areal coverage of ice within the ocean grid cell. 
    Sea ice extent is the integral sum of the areas of all grid cells with at least 15% ice concentration.
    
    Please note that is consistent to use the area variable defined on the ocean grid in relation to sea-ice variables, 
    but you have to ignore the final j-row of e.g. area. So drop the last row with j=385 of area (and ocean output of sea ice) when dealing with the sea ice variables.
 
    Parameters
    ----------
    ds : xarray.DataArray i.e.  ds[var] (var = fice in BLOM)
    pweight : xarray.DataArray with area information
    
    Returns
    -------
    ds_out : xarray.Dataset with sea-extent for each hemisphere, in March and in September

    """
    ds_out = None
    if not isinstance(pweight, xr.DataArray):
        pweight = get_areacello(cmor=cmor)
    pweight = pweight.isel(j=slice(0, 384))
    if ds.j.shape[0] == 385:
        ds = ds.isel(j=slice(0, 384))
    for monthnr in [3, 9]:
        da = select_month(ds, monthnr)
        mask = xr.where(da >= 15, 1, 0)
        parea = mask * pweight
        parea = parea.assign_coords(lat=pweight.lat)
        parea = parea.where(parea > 0)
        # seems like there is a factor 1e2 off. Expected 1e-12, not 1e-14
        SHout = 1e-14 * (parea.where(parea.lat <= 0).sum(dim=("i", "j")))
        SHout.attrs["standard_name"] = "siext_SH_0%i" % monthnr
        SHout.attrs["units"] = "10^6 km^2"
        SHout.attrs["long_name"] = "southern_hemisphere_sea_ice_extent_month_0%i" % monthnr
        NHout = 1e-14 * (parea.where(parea.lat >= 0).sum(dim=("i", "j")))
        NHout.attrs["standard_name"] = "siext_NH_0%i" % monthnr
        NHout.attrs["units"] = "10^6 km^2"
        NHout.attrs["long_name"] = "northern_hemisphere_sea_ice_extent_month_0%i" % monthnr
        if isinstance(ds_out, xr.Dataset):
            ds_out = xr.merge(
                [
                    ds_out,
                    SHout.to_dataset(name="siext_SH_0%i" % monthnr),
                    NHout.to_dataset(name="siext_NH_0%i" % monthnr),
                ]
            )
        else:
            ds_out = xr.merge(
                [SHout.to_dataset(name="siext_SH_0%i" % monthnr), NHout.to_dataset(name="siext_NH_0%i" % monthnr)]
            )

    return ds_out


def sea_ice_area(ds, pweight=None, cmor=True):
    """ 
    Calculates the sea ice extent from the sea ice concentration fice in BLOM raw output and siconc in cmorized files
    Sea ice concentration (fice or siconc) is the percent areal coverage of ice within the ocean grid cell. 
    Sea ice area is the integral sum of the product of ice concentration and area of all grid cells with at least 15% ice concentration
    
    Parameters
    ----------
    ds : xarray.DataArray i.e.  ds[var] (var = fice in BLOM)
    pweight : xarray.DataArray with area information
    
    Returns
    -------
    ds_out : xarray.Dataset with sea-extent for each hemisphere, in March and in September

    """
    ds_out = None
    if not isinstance(pweight, xr.DataArray):
        pweight = get_areacello(cmor=cmor)
    pweight = pweight.isel(j=slice(0, 384))
    if ds.j.shape[0] == 385:
        ds = ds.isel(j=slice(0, 384))
    for monthnr in [3, 9]:
        da = select_month(ds, monthnr)
        mask = xr.where(da >= 15, 1, 0)
        # convert to fraction
        parea = mask * (1e-2 * da * pweight)
        parea = parea.assign_coords(lat=pweight.lat)
        parea = parea.where(parea > 0)
        # hmmm seems like there is a factor 1e2 off. Expected 1e-12, not 1e-14
        SHout = 1e-14 * (parea.where(parea.lat <= 0).sum(dim=("i", "j")))
        SHout.attrs["standard_name"] = "siarea_SH_0%i" % monthnr
        SHout.attrs["units"] = "10^6 km^2"
        SHout.attrs["long_name"] = "southern_hemisphere_sea_ice_area_month_0%i" % monthnr
        NHout = 1e-14 * (parea.where(parea.lat >= 0).sum(dim=("i", "j")))
        NHout.attrs["standard_name"] = "siarea_NH_0%i" % monthnr
        NHout.attrs["units"] = "10^6 km^2"
        NHout.attrs["long_name"] = "northern_hemisphere_sea_ice_area_month_0%i" % monthnr
        if isinstance(ds_out, xr.Dataset):
            ds_out = xr.merge(
                [
                    ds_out,
                    SHout.to_dataset(name="siarea_SH_0%i" % monthnr),
                    NHout.to_dataset(name="siarea_NH_0%i" % monthnr),
                ]
            )
        else:
            ds_out = xr.merge(
                [SHout.to_dataset(name="siarea_SH_0%i" % monthnr), NHout.to_dataset(name="siarea_NH_0%i" % monthnr)]
            )

    return ds_out


def select_atlantic_latbnds(ds):
    """
    Selects the Atlantic meridional overtuning streamfunction / heat transport 
    @2&N (rapid), @45N and the maximum between 20N and 60N

    Parameters
    ----------
    ds : xarray.DataArray i.e.  ds[var] var = mmflxd (raw) and var = msftmz (cmorized) 
                             or ds[var] var = mhflx (raw) and var = hfbasin (cmorized)
    
    Returns
    -------
    a zip list of the 3 sections xarray.DataArray and the latitudes for the corresponding sections 
 
    """
    # basin = 0 ->  sector = atlantic_arctic_ocean
    ds = ds.isel(basin=0)
    amoc26 = ds.sel(lat=26)
    amoc45 = ds.sel(lat=45)
    amoc20_60 = ds.sel(lat=slice(20, 60)).max(dim="lat")
    return zip([amoc26, amoc45, amoc20_60], ["26N", "45N", "max20N_60N"])


def amoc(ds):
    """ 
    Calculates the Atlantic meridional overturning circulation 
    @26N (rapid), @45N and the maximum between 20N and 60N  
    
    Parameters
    ----------
    ds : xarray.DataArray i.e.  ds[var] var = mmflxd (raw) and var = msftmz (cmorized)
    
    Returns
    -------
    ds_out : xarray.Dataset with AMOC @26N, 45N and max(20N,60N)
    """
    ds_out = None
    zipvals = select_atlantic_latbnds(ds)
    for da, lat_lim in zipvals:
        da = 1e-9 * da.max(dim="lev")
        da.attrs["long_name"] = "Max Atlantic Ocean Overturning Mass Streamfunction @%s" % lat_lim
        da.attrs["units"] = "kg s-1"
        da.attrs["standard_name"] = "max_atlantic_ocean_overturning_mass_streamfunction_%s" % lat_lim
        da.attrs["description"] = (
            "Max Atlantic Overturning mass streamfunction arising from all advective mass transport processes, resolved and parameterized @%s"
            % lat_lim
        )
        if "lat" in da.coords:
            da.attrs["lat"] = "%.1fN" % da.lat.values
            da = da.drop("lat")
        if "basin" in da.coords:
            da.attrs["basin"] = "%s" % da.basin.values
            da = da.drop("basin")
        da = da.to_dataset(name="amoc_%s" % lat_lim)
        if isinstance(ds_out, xr.Dataset):
            ds_out = xr.merge([ds_out, da])
        else:
            ds_out = da
    return ds_out


def atl_hfbasin(ds):
    """ 
    Calculates the Atlantic northward ocean heat transport 
    @26N (rapid), @45N and the maximum between 20N and 60N  
    
    Parameters
    ----------
    ds : xarray.DataArray i.e.  ds[var] var = mhflx (raw) and var = hfbasin (cmorized)
    
    Returns
    -------
    ds_out : xarray.Dataset with AOHT @26N, 45N and max(20N,60N)
    """
    ds_out = None
    zipvals = select_atlantic_latbnds(ds)
    for da, lat_lim in zipvals:
        da = 1e-15 * da
        da.attrs["long_name"] = "Atlantic Northward Ocean Heat Transport @%s" % lat_lim
        da.attrs["units"] = "PW"
        da.attrs["standard_name"] = "atlantic_northward_ocean_heat_transport_%s" % lat_lim
        da.attrs[
            "description"
        ] = "Contains contributions from all physical processes affecting the northward heat transport, including resolved advection, parameterized advection, lateral diffusion, etc."
        if "lat" in da.coords:
            da.attrs["lat"] = "%.1fN" % da.lat.values
            da = da.drop("lat")
        if "basin" in da.coords:
            da.attrs["basin"] = "%s" % da.basin.values
            da = da.drop("basin")
        da = da.to_dataset(name="aoht_%s" % lat_lim)
        if isinstance(ds_out, xr.Dataset):
            ds_out = xr.merge([ds_out, da])
        else:
            ds_out = da
    return ds_out


def areaavg_ocn(ds, pweight=None, cmor=True):
    """ 
    Calculates area averaged values   
    
    Parameters
    ----------
    ds : xarray.DataArray i.e.  ds[var] 
    pweight :   xarray.DataArray with area of ocean and land masks as nan. Default is None
    cmor :      boolean, True for cmorized variables and False for NorESM RAW output. Not so important for 1 deg ocean

    Returns
    -------
    ds_out : xarray.DataArray
    """
    if not isinstance(pweight, xr.DataArray):
        pweight = get_areacello(cmor=cmor)
    # sea-ice variables are on i = 360, j = 384 grids
    if ds.j.shape[0] == 384:
        pweight = pweight.isel(j=slice(0, 384))
    ds_out = ((ds * pweight).sum(dim=("j", "i"))) / pweight.sum()
    if "long_name" in ds.attrs:
        ds_out.attrs["long_name"] = "Globally averaged " + ds.long_name
    if "units" in ds.attrs:
        ds_out.attrs["units"] = ds.units
    if "standard_name" in ds.attrs:
        ds_out.attrs["standard_name"] = ds.standard_name
    return ds_out


def regionalavg_ocn(ds, lat_low=-90, lat_high=90, lon_low=0, lon_high=360, pweight=None, cmor=True):
    """ 
    Calculates area averaged values in a region constrained by lat_low, lat_high, lon_low, lon_high 
    lat_low must be less than lat_high, and both need to have values in the range (-90,90)
    lon_low must be less than lon_high, and both need to have values in the range (0,360)
    
    Parameters
    ----------
    ds :        xarray.DataArray i.e.  ds[var] 
    lat_low :   int, in range (-90,90). Default is -90
    lat_high :  int, in range (-90,90). Default is 90
    lon_low :   int, in range (0, 360). Default is 0
    lon_high :  int, in range (0, 360). Default is 360
    pweight :   xarray.DataArray with area of ocean and land masks as nan. Default is None
    cmor :      boolean, True for cmorized variables and False for NorESM RAW output. Not so important for 1 deg ocean

    Returns
    -------
    ds_out : xarray.DataArray
    """
    if not isinstance(pweight, xr.DataArray):
        pweight = get_areacello(cmor=cmor)
    pweight = mask_region_latlon(pweight, lat_low=lat_low, lat_high=lat_high, lon_low=lon_low, lon_high=lon_high)
    
    ds_out = ((ds * pweight).sum(dim=("j", "i"))) / pweight.sum()
    if "long_name" in ds.attrs:
        ds_out.attrs["long_name"] = (
            "Regional avg (latlims: %i,%i, lonlims: %i,%i) " % (lat_low, lat_high, lon_low, lon_high) + ds.long_name
        )
    if "units" in ds.attrs:
        ds_out.attrs["units"] = ds.units
    if "standard_name" in ds.attrs:
        ds_out.attrs["standard_name"] = "Regional_avg_(%i_%i_%i_%i) " + ds.standard_name
    return ds_out


def volumeavg_ocn(ds, dp, pweight=None, cmor=True):
    """ 
    Calculates volume averaged values   
    
    Parameters
    ----------
    ds : xarray.DataArray i.e. ds[var] e.g. var = thatao, so, tempn, saltn 
    dp : xarray.DataArray with pressure thinkness
    cmor :      boolean, True for cmorized variables and False for NorESM RAW output. Not so important for 1 deg ocean

    Returns
    -------
    ds_out : xarray.DataArray
    """
    if not isinstance(pweight, xr.DataArray):
        pweight = get_areacello(cmor=cmor)
    if "sigma" in ds.coords:
        ds = (ds * dp).sum(dim="sigma")
        dpweight = dp.sum(dim="sigma")
        ds_out = (pweight * ds).sum(dim=("j", "i")) / (pweight * dpweight).sum(dim=("i", "j"))
    if "long_name" in ds.attrs:
        ds_out.attrs["long_name"] = "Volume averaged " + ds.long_name
    if "units" in ds.attrs:
        ds_out.attrs["units"] = ds.units
    if "standard_name" in ds.attrs:
        ds_out.attrs["standard_name"] = ds.standard_name
    return ds_out
