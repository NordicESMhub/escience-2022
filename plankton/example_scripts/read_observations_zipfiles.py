from __future__ import annotations

from pathlib import Path
from typing import IO, Iterator
from zipfile import ZipFile, is_zipfile

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cmocean.cm as cmo
import cartopy.crs as ccrs

def read_zip_data(path: Path) -> pd.DataFrame:
    assert path.is_file(), f"missing {path.name}"
    assert is_zipfile(path), f"{path.name} is not a zip file"
    data = zip_reader(path)
    return pd.concat(data, ignore_index=True)


def zip_reader(path: Path) -> Iterator[pd.DataFrame]:
    print(f"processing {path.name}")
    with ZipFile(path) as zip:
        for name in zip.namelist():
            if name.endswith(".csv") and not name.endswith("_dens.csv"):
                print(f"reading {name}")
                with zip.open(name, mode="r") as csv:
                    yield read_csv(csv)


def read_csv(csv: Path | IO) -> pd.DataFrame:
    df = pd.read_csv(
        csv,
        parse_dates=["yyyy-mm-ddThh:mm"],
        dtype={
            "Cruise": "string",
            "Station": "string",
            "Type": "string",
            "Latitude [degrees_north]": "float16",
            "Longitude [degrees_east]": "float16",
            "Bot. Depth [m]": "float16",
            "PRES [db]": "float16",
            "TEMP [deg C]": "float16",
            "PSAL [psu]": "float16",
            "DOXY [ml/l]": "float16",
        },
    )
    return df.pipe(extract_date).pipe(filter_domain)


def extract_date(df: pd.DataFrame) -> pd.DataFrame:
    date = df["yyyy-mm-ddThh:mm"].dt.date.astype("datetime64")
    return df.assign(date=date).drop(columns=["yyyy-mm-ddThh:mm"])


def filter_domain(df: pd.DataFrame) -> pd.DataFrame:
    lat = df["Latitude [degrees_north]"].gt(60)
    lon = df["Longitude [degrees_east]"].between(-25, 50, inclusive="neither")
    return df[lat & lon]


if __name__ == "__main__":
    DATA = Path("/home/adag/Documents/eScience/BarentsSeaObs/ICES_CTD_BS.zip")
    df = read_zip_data(DATA)
    var = 'TEMP [deg C]'
    print(df[var].describe())

#%% E.g. only look at the upper 100 m
    depth = df['Bot. Depth [m]'].lt(100)
    df = df[depth]
    varlist = [var, 'date', 'Latitude [degrees_north]', 'Longitude [degrees_east]']
    df = df[varlist]
    ds = df.to_xarray()
    ds =ds.assign_coords({"index": pd.MultiIndex.from_arrays([ds.date.values, ds['Latitude [degrees_north]'].values, ds['Longitude [degrees_east]'].values], names=["time","lat","lon"])}).unstack("index")

#%%
    ds = ds[var].groupby('time.month').mean(dim='time')
    x,y = np.meshgrid(ds.lon.values, ds.lat.values)
    vmin = -2
    vmax = 14
    dval = .5
    levels = np.arange(vmin, vmax + dval, dval)
    fig= plt.figure(figsize=(9,7))
    fig.suptitle('Observatios of from ship cruises - August (1990 - 2019)')
    for monthnr in ds.month.values:
        da = ds.sel(month=monthnr)
        ax = plt.subplot(3, 4, monthnr, projection=ccrs.Orthographic(0, 90))
        cf = ax.contourf(x, y, da.values, colormap=cmo.haline, transform=ccrs.PlateCarree(), 
          vmin = vmin, vmax = vmax, levels = levels, extend="max")
        ax.coastlines(linewidth=0.5)
        ax.set_extent([-25, 50, 60, 90], ccrs.PlateCarree())
        ax.gridlines(linewidth=0.5, ylocs = [60, 70, 80], linestyle='--')
        ax.set_title('%s-01'%(str.zfill(str(monthnr),2)), fontsize=11)
    cax = fig.add_axes([0.3, 0.08, 0.4, 0.02])
    cb = plt.colorbar(cf, cax=cax, ticks=levels[::4], orientation="horizontal")
    cb.set_label("SST [degC]", fontsize=10)
    plt.subplots_adjust(left=0.05, bottom=0.12, right=.95, top=0.92, hspace=0.15, wspace = 0.)
    #plt.savefig('figures/NPP_MODIS_%s.png'%year)
      