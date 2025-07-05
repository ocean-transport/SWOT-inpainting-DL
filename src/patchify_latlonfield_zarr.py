import xarray as xr
import zarr
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

import sys
sys.path.append('/home/tm3076/projects/NYU_SWOT_project/SWOT-data-analysis/src')
import interp_utils
import swot_utils

from importlib import reload
import os
import traceback
import glob

#turn off warnings
import warnings
warnings.filterwarnings("ignore")


# For sbatch script
import argparse
import time

import logging




def subset_raw_SST(idx, path_to_dataset, output_dir, varname, n=128, L_x=512e3, L_y=512e3):
    patch_coords = np.load('/home/tm3076/scratch/pytorch_learning_tiles/np_SST_masks/x_y_coordinates.npy')
    lon0 = patch_coords[idx,0]
    lat0 = patch_coords[idx,1]
    patch_ID = patch_coords[idx,2]
    
    LOG_FILENAME = f'./sbatch_output_logs/{patch_ID}_error.out'
    logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)
    logging.debug(f'Errors with patchify for patch {patch_ID} listed below')

    output_dir = f"{output_dir}/"
    os.makedirs(output_dir,exist_ok=True)
    ds_gridded = []   

    # First open a dask view of the llc4320 surface fields
    worker_ds = xr.open_zarr(path_to_dataset)
    if "__xarray_dataarray_variable__" in worker_ds:
        worker_ds = worker_ds.rename({"__xarray_dataarray_variable__":f"{varname}"})
    ds_tiles = []
    if not os.path.exists(f"{output_dir}/{str(int(patch_ID)).zfill(3)}.zarr"):
        for t in range(len(worker_ds.time)):
            ds_llc = swot_utils.xr_subset(worker_ds.isel(time=t).compute(), [lat0-4,lat0+4], lon_bounds=[lon0-4,lon0+4])
            ds_gridded = interp_utils.grid_everything(ds_llc, lat0, lon0,  n=n, L_x=L_x, L_y=L_y)
            ds_tiles.append(ds_gridded)
            ds_llc.close()
        try:
            xr.concat(ds_tiles,dim="time").to_zarr(f"{output_dir}/{str(int(patch_ID)).zfill(3)}.zarr")
        except:
            logging.exception(f'Got exception on main handler for FIELD patch{str(int(patch_ID)).zfill(3)}')
            logging.exception(f'tiles look like {ds_tiles}')            
    else:
        print(f"{output_dir}/{str(int(patch_ID)).zfill(3)}.zarr already exists! Skipping for now..")
        pass

    
    return 
    

# ─────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patchify a dataset with latitude/longitude coordinates \
                                                  onto a collection of gridded patches.")
    parser.add_argument("--i", nargs="?", type=float, required=True, help="patch index interval start (i.e. starts patching from patch i*interval)")
    parser.add_argument("--interval", nargs="?", type=float, required=True, help="patch index interval length")
    parser.add_argument("--path_to_dataset", nargs="?", type=str, required=True, help="Path to dataset we want to patchify")
    parser.add_argument("--output_dir", nargs="?", type=str, required=True, help="Path to dataset we want to patchify")
    parser.add_argument("--varname", nargs="?", type=str, required=True, help="Name of patched variable")

    parser.add_argument("--n", nargs="?", type=int, default=128, help="Number of x and y grid points per patch")
    parser.add_argument("--L_x", nargs="?", type=int, default=512e3, help="Grid spacing in the x-direction")
    parser.add_argument("--L_y", nargs="?", type=int, default=512e3, help="Grid spacing in the y-direction")
    
    args = parser.parse_args()

# Parse
args = parser.parse_args()

patch_coords = np.load('/home/tm3076/scratch/pytorch_learning_tiles/np_SST_masks/x_y_coordinates.npy')
patch_ID = patch_coords[1,2]
print(str(int(patch_ID)).zfill(3))

# Run over 50-patch intervals
for patch in range(int(args.i)*int(args.interval),min(len(patch_coords),int(args.i)*int(args.interval)+int(args.interval))):
    print(patch)
    subset_raw_SST(patch, args.path_to_dataset, args.output_dir, args.varname, n=args.n, L_x=args.L_x, L_y=args.L_y)

