import numpy as np
import xarray as xr
import zarr
from datetime import date, timedelta
import warnings
warnings.filterwarnings("ignore")
import os
from glob import glob
import dask
dask.config.set(scheduler='synchronous')

import sys
sys.path.append('/home.ufs/tm3076/swot_SUM03/SWOT_project/SWOT-data-analysis/src')
import interp_utils

import torch
from torch.utils.data import Dataset, DataLoader

import traceback
import threading

PRINT_LOCK = threading.Lock()

from functools import partial

def standardize(x, mean=None, std=1.0):
    if mean is not None:
        x = x - mean
    return x / std

def standardize_samplewise(x, std=1.0):
    return (x - np.mean(x)) / std

def no_transform(x):
    return x



class llc4320_dataset(Dataset):
    def __init__(self, data_dir, mid_timestep, N_t, patch_coords, 
                 infields, outfields, in_mask_list, out_mask_list, 
                 in_transform_list, out_transform_list,
                 SST_quality_level=1, sst_only=False, sst_cloud_mask=False,
                 N=128, L_x=512e3, L_y=512e3, flatten=False, return_meta_data=True,
                 standards=None, multiprocessing=False, device=None, cloud_rho=.7):

        self.device = device
        self.data_dir = data_dir
        self.mid_timestep = mid_timestep
        self.N_t = N_t
        self.patch_coords = patch_coords
        self.infields = infields
        self.outfields = outfields
        self.in_mask_list = in_mask_list
        self.out_mask_list = out_mask_list
        self.in_transform_list = in_transform_list
        self.out_transform_list = out_transform_list
        self.SST_quality_level = SST_quality_level
        self.N = N
        self.L_x = L_x
        self.L_y = L_y
        self.flatten = flatten
        self.return_meta_data = return_meta_data
        self.cloud_rho = cloud_rho

        # Standards: tuple or dict
        if standards is None:
            standards = {
                "mean_ssh": 0.0, "std_ssh": 1.0,
                "mean_sst": 0.0, "std_sst": 1.0
            }

        mean_ssh = standards["mean_ssh"]
        std_ssh = standards["std_ssh"]
        mean_sst = standards["mean_sst"]
        std_sst = standards["std_sst"]

        # Define reusable transforms
        self.transforms = {
            "std_ssh_norm": partial(standardize, std=std_ssh),
            "std_sst_norm": partial(standardize, std=std_sst),
            "std_mean_ssh_norm": partial(standardize_samplewise, std=std_ssh),
            "std_mean_sst_norm": partial(standardize_samplewise, std=std_sst),
            "std_global_mean_ssh_norm": partial(standardize, mean=mean_ssh, std=std_ssh),
            "std_global_mean_sst_norm": partial(standardize, mean=mean_sst, std=std_sst),
            "no_transform": no_transform,
        }

        # Preload SWOT swaths
        self.worker_generic_swath0 = xr.open_zarr(f"{self.data_dir}/SWOT_swaths_488/hawaii_c488_p015.zarr")
        self.worker_generic_swath1 = xr.open_zarr(f"{self.data_dir}/SWOT_swaths_488/hawaii_c488_p028.zarr")

        # Preload cloud masks
        self.cloud_catalog = xr.open_zarr(f"{self.data_dir}/HRS_SST_tiles/agg_cloud_percentages/catalog.zarr").compute()

    def __len__(self):
        return self.patch_coords.shape[0]

    def __getitem__(self, idx):
        try:
            return self._load_patch(idx)
        except Exception as e:
            print(f"[Warning] Failed to load patch {idx}: {e} — falling back to patch 065")
            return self._load_patch(patch_id="065")

    def _load_patch(self, idx=None, patch_id=None):
        if patch_id is None:
            patch_id = str(int(self.patch_coords[idx, 2])).zfill(3)
            coords = self.patch_coords[idx]
        else:
            coords = None
        invars = self._load_patch_fields(patch_id, self.infields, self.in_transform_list, self.in_mask_list)
        outvars = self._load_patch_fields(patch_id, self.outfields, self.out_transform_list, self.out_mask_list)
        invar = torch.stack(invars, dim=1)
        outvar = torch.stack(outvars, dim=1)
        if self.flatten:
            invar = invar.flatten(0, 1)
            outvar = outvar.flatten(0, 1)
        if self.return_meta_data:
            metadata = {
                "patch_ID": patch_id,
                "mid_timestep": self.mid_timestep,
                "patch_coords": coords,
                "latitude": self.latitude,
                "longitude": self.longitude
            }
            return invar, outvar, metadata
        return invar, outvar

    def _load_patch_fields(self, patch_id, fields, transform_keys, mask_keys):
        variables = []
        for i, field in enumerate(fields):
            ds = xr.open_zarr(f"{self.data_dir}/{field}/{patch_id}.zarr").isel(
                time=slice(int(self.mid_timestep - self.N_t / 2), int(self.mid_timestep + self.N_t / 2))
            )
            self.latitude = ds.latitude.values
            self.longitude = ds.longitude.values
            var = ds[list(ds.data_vars.keys())[0]]
            var = self.transforms[transform_keys[i]](var)
            mask = self.get_mask(mask_keys[i], patch_id)
            variables.append(torch.tensor(var.values) * mask)
        return variables

    def get_mask(self, mask_key, patch_ID):
        if (mask_key is None) or ("None" in mask_key):
            return 1
        elif "swot" in str(mask_key).lower():
            return self.get_random_swot_mask()
        elif "cloud_tseries" in str(mask_key).lower():
            return self.get_cloud_mask_timeseries(patch_ID)
        elif "cloud_rho" in str(mask_key).lower():
            return self.get_cloud_mask_rho()
        else:
            raise ValueError(f"Unknown mask type: {mask_key}")

    def get_random_swot_mask(self, version="random"):
        # Helper function to generate a SWOT-like mask
        sw_corner = [-152.5, 30.0]
        ne_corner = [-149.5, 42.0]
        lon = np.random.randint(sw_corner[0], ne_corner[0])
        lat = np.random.randint(sw_corner[1], ne_corner[1])
        if version == "random":
            nrand = np.random.randint(2)
            if nrand == 0:
                m0 = interp_utils.grid_everything(self.worker_generic_swath0, lat, lon, n=self.N, L_x=self.L_x, L_y=self.L_y)
            else:
                m0 = interp_utils.grid_everything(self.worker_generic_swath1, lat, lon, n=self.N, L_x=self.L_x, L_y=self.L_y)
            mask = (m0.ssha.fillna(0)).values > 0
        elif version == "both":
            m0 = interp_utils.grid_everything(self.worker_generic_swath0, lat, lon, n=self.N, L_x=self.L_x, L_y=self.L_y)
            m1 = interp_utils.grid_everything(self.worker_generic_swath1, lat, lon, n=self.N, L_x=self.L_x, L_y=self.L_y)
            mask = (m0.ssha.fillna(0) + m1.ssha.fillna(0)).values > 0
        return torch.tensor(mask.astype(np.float32))

    def get_cloud_mask_timeseries(self, patch_ID):
        path = f"{self.data_dir}/HRS_SST_tiles/agg_cloud_masks/{patch_ID}.nc"
        cm = xr.open_dataset(path).sst_filtered_q5
        mid = np.random.randint(int(self.N_t / 2), len(cm.time) - int(self.N_t / 2))
        cm = cm.isel(time=slice(mid - self.N_t // 2, mid + self.N_t // 2))
        cm = (cm * 0 + 1).where(cm > 0, other=0)
        return torch.tensor(cm.values)

    def get_cloud_mask_rho(self):
        cloud_catalog_rho = self.cloud_catalog.where(self.cloud_catalog.rho>=self.cloud_rho,drop=True)
        sample_N = cloud_catalog_rho.isel(i_time = np.random.randint(len(cloud_catalog_rho.i_time)))
        sample_N_tstep = int(sample_N.patch_timestep)
        sample_N_patch_ID = str(int(sample_N.patch_id)).zfill(3)
        path = f"{self.data_dir}/HRS_SST_tiles/agg_cloud_masks/{sample_N_patch_ID}.nc"
        cm = ~np.isnan(xr.open_dataset(path).isel(time=sample_N_tstep).sst_filtered_q5)
        return torch.tensor(cm.values)


