# --------------------------------------------------
# Name: PyDealias.py
# Author: Robby M. Frost
# Advanced Radar Research Center
# University of Oklahoma
# Created: 17 Nov 2025
# Purpose: Dealias radial velocity data similar 
# to solo3, but with mpl_point_clicker
# --------------------------------------------------
# imports
import pyart
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.widgets import TextBox
from matplotlib.path import Path
from mpl_point_clicker import clicker
from matplotlib import rc
import glob
import os

# plotting set up
plt.rcParams['axes.labelweight'] = 'normal'
plt.rcParams['text.latex.preamble'] = r'\usepackage{bm}'
rc('font', family='sans-serif')
rc('font', weight='normal', size=6)
rc('figure', facecolor='white')
rc('axes', titlesize=8)
# --------------------------------------------------
# PPI plotting function
def plot_ppi(x, y, ref, vel, plot_strs):
    """
    Plot RaXPol PPI

    :param dict radar: pyart radar object
    :param str tstring: time string for plotting
    """
    # make fig
    fig, axs = plt.subplots(figsize=(6,3.5), 
                            ncols=2,
                            constrained_layout=True,
                            dpi=150,
                            sharey=True)
    fig.suptitle(f"{plot_strs['instr_name']} {plot_strs['tplot']} UTC (El={plot_strs['el']:.1f}$^{{\\circ}}$)", 
                fontsize=10, 
                fontweight='bold')
    # reflectivity
    ax = axs[0]
    vmin, vmax = -10, 70
    pcm = ax.pcolormesh(x, y, ref,
                        vmin=vmin, vmax=vmax,
                        cmap='Carbone42')
    
    cbar = fig.colorbar(pcm, ax=ax, orientation='horizontal', pad=0.015)
    cbar.set_label("$Z$ [dBZ]")
    cbar.set_ticks(np.arange(vmin, vmax + 0.001, 10))
    ax.set_title("Equivalent Reflectivity Factor")
    ax.set_ylabel("Meridional Distance [km]")
    # radial velocity
    ax = axs[1]
    vmin, vmax = -50, 50
    pcm = ax.pcolormesh(x, y, vel,
                        vmin=vmin, vmax=vmax,
                        cmap='Carbone42')
    cbar = fig.colorbar(pcm, ax=ax, orientation='horizontal', pad=0.015)
    cbar.set_label("$V_r$ [m s$^{-1}$]")
    cbar.set_ticks(np.arange(vmin, vmax + 0.001, 10))
    ax.set_title("Radial Velocity")
    # clean up plot
    for iax in axs.flatten():
        iax.set_aspect('equal')
        iax.set_xlim(-2,3)
        iax.set_xlabel("Zonal Distance [km]")
        iax.set_ylim(3,8)
        iax.grid(alpha=0.6)

    return fig, axs


# --------------------------------------------------
# masking function
def mask_inside(xx, yy, px, py):
    """
    Mask points inside of a polygon. Useful for extracting
    data from a specific region of a storm at different 
    grid resolutions

    :param array gx: Array of x positions from dataset [km]
    :param array gy: Array of y positions from dataset [km]
    :param array px: Array of x positions of polygon [km]
    :param array py: Array of y positions of polygon [km]
    """
    # array holding polygon points
    poly_coords = np.array([list(p) for p in zip(px, py)])
    # connect polygon points
    poly_path = Path(poly_coords)

    # list of all dataset positions
    ds_pts = np.vstack((xx.ravel(), yy.ravel())).T

    # mask dataset points inside of polygon
    inside_mask = poly_path.contains_points(ds_pts)
    inside_mask = inside_mask.reshape(xx.shape)

    return inside_mask


# --------------------------------------------------
# dealias inside mask
def dealias_in_mask(vel, 
                    n_folds, 
                    inside_mask,
                    nyquist):
    
    # points that are not folded
    if n_folds < 0:
        unf_mask = vel < 0
    if n_folds > 0:
        unf_mask = vel > 0
    # full mask for dealiasing
    full_mask = inside_mask & unf_mask
    # mask velocity field
    vel_masked = vel[full_mask]

    # unfold masked region
    veldel = vel_masked - 2*n_folds*nyquist
    # add back to velocity field
    velnew = vel.copy()
    velnew[full_mask] = veldel

    return velnew


# --------------------------------------------------
# read radar data

# drad = input("Full path to radar moment data: ")
# drad = [drad if drad[-1]=="/" else f"{drad}/"][0]
drad = "/data/arrcwx/wrt_jtrf/morton/grf/bf2/"

# desired file index
jt = int(input("Index of file to read: "))

# list of full file paths
ffiles = glob.glob(f"{drad}RAXPOL*")
# number of files
nfiles = len(ffiles)
# just file names
files = [os.path.basename(ffiles[jt]) for jt in range(nfiles)]


# read radar file
radar = pyart.io.read(ffiles[jt])

# adjust reflectivity
ref_adj = radar.fields['DBZ']['data'] + 20
radar.add_field_like('DBZ', 'DBZ_ADJ', ref_adj, replace_existing=True)
# extract velocity
vel = radar.fields['VEL']['data']

# fix metadata
radar.metadata['instrument_name'] = 'RaXPol'
# store params
el = float(radar.fixed_angle['data'])
prt = float(radar.instrument_parameters['prt']['data'][-1])
nyquist = 0.0308077 / (prt * 4)
# time information
tu = radar.time['units']
tplot = tu[14:-10] + " " + tu[-9:-1]
# position information
x, y, z = radar.get_gate_x_y_z(0)

# plotting stuff
plot_strs = {
    'el': el,
    'tplot': tplot,
    'instr_name': radar.metadata['instrument_name']
}


# --------------------------------------------------
# plot and mark points to dealias

# -----------------------
# space actions
first_press = True
velnew = None
def on_space(event):
    global first_press, velnew
    
    # space pressed
    if event.key != " ":
        return

    # polygon
    poly_coords = klicker.get_positions()['polygon']
    if not np.array_equal(poly_coords[0], poly_coords[-1]):
        poly_coords = np.vstack([poly_coords, poly_coords[0]])
    
    # mask selected points
    inside_mask = mask_inside(x/1e3, 
                              y/1e3,
                              poly_coords[:,0], 
                              poly_coords[:,1])
    # number of folds
    n_folds = int(input("Number of folds: "))

    # first unfold
    if first_press:
        first_press = False
        velnew = dealias_in_mask(vel, n_folds, inside_mask, nyquist)
    # subsequent folds
    else:
        velnew = dealias_in_mask(velnew, n_folds, inside_mask, nyquist)

    # reset positions
    klicker._positions['polygon'] = []
    # plot unfolded velocity
    axs[1].pcolormesh(x/1e3, y/1e3, velnew,
                      vmin=-50, vmax=50,
                      cmap='Carbone42')
    plt.show()


# -----------------------
# escape exit actions
def on_esc(event):
    """Called whenever ENTER is pressed: closes plot."""
    if event.key == "escape":
        plt.close(event.canvas.figure)


# -----------------------
# actual plotting

# make fig
fig, axs = plot_ppi(x/1e3, y/1e3, ref_adj, vel, plot_strs)
# draw points
klicker = clicker(axs[1], ["polygon"], markers=["+"], colors='black')
# Connect key events
fig.canvas.mpl_connect("key_press_event", on_space)
fig.canvas.mpl_connect("key_press_event", on_esc)
plt.show()


# -----------------------
# output
cflag = input("Look good? (y/n) ")
if (cflag == 'y') or (cflag == 'Y'):
    # add dealiased velocity to radar object
    velfield = {
        'data': velnew,
        'units': 'm/s',
        'long_name': 'Dealiased Radial Velocity',
        'standard_name': 'VELDEL',
        'coordinates': radar.fields['VEL']['coordinates']
    }
    radar.add_field('VELDEL', velfield, replace_existing=True)

    # output path
    dout = f"{drad}DEALIASED-{files[jt]}"
    if os.path.exists(dout):
        os.remove(dout)
    # output to cfradial
    pyart.io.write_cfradial(dout, radar)
    print(f"Output to: {dout}")

if (cflag == 'n') or (cflag == 'N'):
    print("Changes removed, restart script")



# --------------------------------------------------
# old space actions

# first_press = True
# def on_space(event):
#     """Called whenever SPACE is pressed."""
#     global first_press
#     # trigger when space pressed
#     if event.key == " ":

#         # first unfold
#         if first_press:
#             first_press = False
#             # extract points
#             poly_coords = klicker.get_positions()['polygon']

#             # close polygon
#             if not np.array_equal(poly_coords[0], poly_coords[-1]):
#                 poly_coords = np.vstack([poly_coords, poly_coords[0]])
#             # mask inside polygon
#             inside_mask = mask_inside(x/1e3, # radar x positions
#                                       y/1e3, # radar y positions
#                                       poly_coords[:,0], # polygon x positions
#                                       poly_coords[:,1] # polygon y positions
#                                       )
#             n_folds = int(input("Number of folds: "))
#             # dealias
#             velnew = dealias_in_mask(vel,
#                                     n_folds,
#                                     inside_mask,
#                                     nyquist)
            
#             # remove old points
#             klicker._positions['polygon'] = []
#             # for artist in klicker.artists['polygon']:
#             #     artist.remove()
#             # klicker.artists['polygon'] = []
#             # replot with dealiased velocity
#             axs[1].pcolormesh(x/1e3, y/1e3, velnew,
#                               vmin=-50, vmax=50,
#                               cmap='pyart_Carbone42')
#             plt.show()
#             return velnew
        
#         # subsequent unfolds
#         if not first_press:
#             # extract points
#             poly_coords = klicker.get_positions()['polygon']

#             # close polygon
#             if not np.array_equal(poly_coords[0], poly_coords[-1]):
#                 poly_coords = np.vstack([poly_coords, poly_coords[0]])
#             # mask inside polygon
#             inside_mask = mask_inside(x/1e3, # radar x positions
#                                     y/1e3, # radar y positions
#                                     poly_coords[:,0], # polygon x positions
#                                     poly_coords[:,1] # polygon y positions
#                                     )
#             n_folds = int(input("Number of folds: "))
#             # dealias
#             velnew = dealias_in_mask(velnew,
#                                     n_folds,
#                                     inside_mask,
#                                     nyquist)
            
#             # remove old points
#             klicker._positions['polygon'] = []
#             # for artist in klicker.artists['polygon']:
#             #     artist.remove()
#             # klicker.artists['polygon'] = []
#             # replot with dealiased velocity
#             axs[1].pcolormesh(x/1e3, y/1e3, velnew,
#                             vmin=-50, vmax=50,
#                             cmap='pyart_Carbone42')
#             plt.show()