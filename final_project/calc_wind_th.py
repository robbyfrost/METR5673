# --------------------------------------------------
# Name: calc_wind_th.py
# Author: Robby M. Frost and J.W. Thiesing
# Advanced Radar Research Center
# University of Oklahoma
# Created: 1 Dec 2025
# Purpose: Calculate delta vmax time-height series
# --------------------------------------------------
import numpy as np
import xarray as xr
import pyart
import datetime
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.ticker import MultipleLocator
import matplotlib.dates as mdates
from matplotlib import cm, colors
import glob
import sys
import os
from grf_functions import *
# --------------------------------------------------
# settings

# radius [m]
rad = 500
# velocity magnitude for clutter mask [m/s]
vmag = 3
# spectrum width cutoff for clutter mask [m/s]
swc = 3

# flag for grf or not
grf = True
# --------------------------------------------------
# read in data

# path to radar data
if grf:
    drad = "/data/arrcwx/wrt_jtrf/morton/grf/bf2/"
else:
    drad = "/data/arrcwx/wrt_jtrf/morton/no_grf/"
# path to output vortex positions
dvort = "/data/arrcwx/wrt_jtrf/morton/sa/"

# files
ffiles = sorted(glob.glob(f"{drad}DEALIASED*"))
files = [os.path.basename(f) for f in ffiles]
tstrings = [f[17:32] for f in files]
tdt = np.array([datetime.strptime(tstr, "%Y%m%d-%H%M%S") for tstr in tstrings])
nt = len(tstrings)


# read in radar data
rall = []
for jt in range(len(files)):
    radar = pyart.io.read(ffiles[jt])
    rall.append(radar)

# read vortex positions
vffiles = sorted(glob.glob(f"{dvort}vortex*"))
xv, yv = [np.empty(len(vffiles)) for _ in range(2)]
for jt in range(len(files)):
    vortex = np.loadtxt(vffiles[jt], delimiter=',', skiprows=1)
    xv[jt], yv[jt] = vortex[0], vortex[1]
# --------------------------------------------------
# calculate time-height series

# arrays to store
vmax, vmin, zmax, zmin = [np.empty(len(vffiles)) for _ in range(4)]
# loop over radar files
for jt, radar in enumerate(rall):
    # position information
    x, y, z = radar.get_gate_x_y_z(0)
    # distance from tornado
    dist = ((x-xv[jt]*1e3)**2 + (y-yv[jt]*1e3)**2)**(1/2)
    # mask within radius
    dist_mask = dist <= rad
    
    # extract variables
    vel = radar.fields['VEL']['data']
    veldel = radar.fields['VELDEL']['data']
    sw = radar.fields['WIDTH']['data']
    # rudimentary clutter mask
    cmask = (np.abs(vel) < vmag) & (sw < swc)
    # apply masks
    vel_masked0 = np.where(dist_mask, veldel, np.nan)
    if grf:
        vel_masked = vel_masked0
    else:
        vel_masked = np.where(cmask, np.nan, vel_masked0)
    
    # min/max
    vmax[jt] = np.nanmax(vel_masked)
    maxidx = np.argmax(vel_masked)
    vmin[jt] = np.nanmin(vel_masked)
    minidx = np.argmin(vel_masked)

    # get indices
    mask_indices = np.where(dist_mask)
    imax = (mask_indices[0][maxidx], mask_indices[1][maxidx])
    imin = (mask_indices[0][minidx], mask_indices[1][minidx])
    # altitudes of min/max
    zmax[jt] = z[imax]
    zmin[jt] = z[imin]
# delta vmax
deltav = vmax - vmin
# --------------------------------------------------
# store in xarray and output

# dataset
ts_ds = xr.Dataset(
    data_vars={
        'vmax': (('time'), vmax),
        'vmin': (('time'), vmin),
        'deltav': (('time'), deltav),
        'zmax': (('time'), zmax),
        'zmin': (('time'), zmin),
    },
    coords={'time': tdt}
)
# output
if grf:
    dout = f"{dvort}vmax_th_grf.nc"
else:
    dout = f"{dvort}vmax_th_no_grf.nc"
if os.path.exists(dout):
    os.remove(dout)
ts_ds.to_netcdf(dout)
print(f"Output time-height series to: {dout}")