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
import time



import sys
if os.path.exists('/home.ufs/tm3076/swot_SUM03/SWOT_project/SWOT-data-analysis/src'):
    sys.path.append('/home.ufs/tm3076/swot_SUM03/SWOT_project/SWOT-data-analysis/src')
else: 
    sys.path.append('/home/tm3076/projects/NYU_SWOT_project/SWOT-data-analysis/src')
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
                 N=128, L_x=512e3, L_y=512e3, return_metadata=False,
                 standards=None, squeeze=False, multiprocessing=False, device=None, cloud_rho=.7,
                 return_masks=False, time_loading=False):

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
        self.squeeze = squeeze
        self.return_meta_data = return_metadata
        self.cloud_rho = cloud_rho
        self.return_masks = return_masks
        self.time_loading = time_loading

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
            "std_global_dailyclimatology_mean_sst_norm": partial(standardize, mean=mean_sst, std=std_sst),
            "no_transform": no_transform,
        }

        # Preload SWOT swaths
        self.worker_generic_swath0 = xr.open_zarr(f"{self.data_dir}/SWOT_swaths_488/hawaii_c488_p015.zarr")
        self.worker_generic_swath1 = xr.open_zarr(f"{self.data_dir}/SWOT_swaths_488/hawaii_c488_p028.zarr")
        # Preload cloud mask catalog
        self.cloud_catalog = xr.open_zarr(f"{self.data_dir}/catalog.zarr").compute()
        self.cloud_catalog_rho = self.cloud_catalog.where(self.cloud_catalog.rho>=self.cloud_rho,drop=True)
    def __len__(self):
        return self.patch_coords.shape[0]

    def __getitem__(self, idx):
        ########################
        self.t0 = time.perf_counter()
        try:
            result = self._load_patch(idx)
        except Exception as e:
            print(f"[Warning] Failed to load patch {idx}: {e} — falling back to patch 065")
            result = self._load_patch(patch_id="065")
        ########################
        if self.time_loading:
            print(f"[Timer] Total __getitem__ duration: {time.perf_counter() - self.t0:.3f} sec")
        return result

    def _load_patch(self, idx=None, patch_id=None):
        if patch_id is None:
            patch_id = str(int(self.patch_coords[idx, 2])).zfill(3)
            coords = self.patch_coords[idx]
        else:
            coords = None
        self.t1 = time.perf_counter()
        invars, in_masks = self._load_patch_fields(patch_id, self.infields, self.in_transform_list, self.in_mask_list)
        invar = torch.nan_to_num(torch.stack(invars, dim=1), nan=0)
        in_masks = torch.nan_to_num(torch.stack(in_masks, dim=1),nan=0)
        ########################
        if self.time_loading:
            print(f"[Timer] Input vars + masks loaded in {time.perf_counter() - self.t1:.3f} sec")
        outfields_specified = (self.outfields not in (None, [], "none"))
        if outfields_specified:
            self.t2 = time.perf_counter()
            out_vars, out_masks = self._load_patch_fields(patch_id, self.outfields,self.out_transform_list, self.out_mask_list)
            outvar = torch.nan_to_num(torch.stack(out_vars, dim=1), nan=0)
            out_masks = torch.nan_to_num(torch.stack(out_masks, dim=1),nan=0)
            ########################
            if self.time_loading:
                print(f"[Timer] Output vars + masks loaded in {time.perf_counter() - self.t2:.3f} sec")
        else:
            outvar, out_masks = torch.tensor([[0]]).float(), torch.tensor([[0]]).float()
        if self.squeeze:
            invar = invar.squeeze()
            outvar = outvar.squeeze()
            in_masks = in_masks.squeeze()
            out_masks = out_masks.squeeze()
            ########################
            if self.time_loading:
                print(f"[Timer] Total _load_patch duration: {time.perf_counter() - self.t0:.3f} sec")
        if self.return_meta_data:
            metadata = {
                "patch_ID": patch_id,
                "patch_coords": coords,
                "mid_timestep_idx": self.mid_timestep,
                "time": self.time,
                "latitude": self.latitude,
                "longitude": self.longitude
            }
            return torch.nan_to_num(invar,nan=0).float(), torch.nan_to_num(outvar,nan=0).float(), metadata
        elif self.return_masks:
            return invar.float(), outvar.float(), in_masks.float(), out_masks.float()
        else:
            return invar.float(), outvar.float()

    def _load_patch_fields(self, patch_id, fields, transform_keys, mask_keys):
        variables = []
        masks = []
        for i, field in enumerate(fields):
            ds = xr.open_zarr(f"{self.data_dir}/{field}/{patch_id}.zarr").isel(
                time=slice(int(self.mid_timestep - self.N_t / 2), int(self.mid_timestep + self.N_t / 2))
            )
            # Save some metadata
            self.time = ds.time.values
            self.latitude = ds.latitude.values
            self.longitude = ds.longitude.values
            var = ds[list(ds.data_vars.keys())[0]]
            var = self.transforms[transform_keys[i]](var)
            mask = self.get_mask(mask_keys[i], patch_id)
            if self.time_loading:
                print(f"[Timer] Field '{field}' loaded and mask '{mask_keys[i]}' applied in {time.perf_counter() - self.t0:.3f} sec")
            variables.append(torch.tensor(var.values).float() * mask)
            masks.append(torch.broadcast_to(mask, var.shape))
        return variables, masks

    def get_mask(self, mask_key, patch_ID):
        if (mask_key is None) or ("None" in mask_key):
            return torch.tensor([1]).float()
        elif "swot" in str(mask_key).lower():
            sampling="all"
            version="random"
            if "calval" in str(mask_key).lower():
                version="calval"
            if "central" in str(mask_key).lower():
                sampling="central"
            if "random" in str(mask_key).lower():
                sampling="random"
            if "nadir" in str(mask_key).lower():
                result = (self.get_random_swot_mask(sampling=sampling,version=version) + self.get_nadir_altimeter_mask(patch_ID)) > 0
            else:
                result = self.get_random_swot_mask(sampling=sampling,version=version)
        elif "nadir" in str(mask_key).lower():
            result = self.get_nadir_altimeter_mask(patch_ID)
        elif "cloud_tseries" in str(mask_key).lower():
            result = self.get_cloud_mask_timeseries(patch_ID)
        elif "cloud_rho" in str(mask_key).lower():
            result = self.get_cloud_mask_rho()
        else:
            raise ValueError(f"Unknown mask type: {mask_key}")
        if self.time_loading:
            print(f"[Timer] Mask '{mask_key}' generated in {time.perf_counter() - self.t0:.3f} sec")
        return result

    def get_random_swot_mask(self, version="random", sampling="all"):
        # Helper function to generate a SWOT-like mask
        sw_corner = [-154.5, 35.3]
        ne_corner = [-147.5, 42.3]
        lat_max, lat_min, l_step, lon_i = 9000, 2000, 4, np.random.randint(5)
        lon = np.random.uniform(sw_corner[0], ne_corner[0])
        lat = np.random.uniform(sw_corner[1], ne_corner[1])
        if version == "random":
            nrand = np.random.randint(2)
            if nrand == 0:
                m0 = interp_utils.grid_everything(self.worker_generic_swath0.ssha[lat_min:lat_max:l_step,lon_i::l_step], lat, lon, n=self.N, L_x=self.L_x, L_y=self.L_y)
            else:
                m0 = interp_utils.grid_everything(self.worker_generic_swath1.ssha[lat_min:lat_max:l_step,lon_i::l_step], lat, lon, n=self.N, L_x=self.L_x, L_y=self.L_y)
            mask = (m0.fillna(0)).values > 0
        elif version == "calval":
            ms = [interp_utils.grid_everything(self.worker_generic_swath0.ssha[lat_min:lat_max:l_step,lon_i::l_step], lat, lon, n=self.N, L_x=self.L_x, L_y=self.L_y),
                  interp_utils.grid_everything(self.worker_generic_swath1.ssha[lat_min:lat_max:l_step,lon_i::l_step], lat, lon, n=self.N, L_x=self.L_x, L_y=self.L_y)
                 ]
            a0 = np.random.randint(2)
            a1 = int((a0-1)**2)
            calval_mask = torch.stack([torch.tensor(ms[a0].fillna(0).values > 0).float(), torch.tensor(ms[a1].fillna(0).values > 0).float()])
            if self.N_t > 1:
                mask_broadcast = torch.broadcast_to(calval_mask,(self.N_t//2+self.N_t%2,2,128,128))
                mask = mask_broadcast.reshape(self.N_t+self.N_t%2,128,128)[:self.N_t]
            else:
                mask = calval_mask[0]
        mask = torch.tensor(mask).float()
        if sampling=="all":
            return torch.tensor(mask).float()
        elif sampling=="central":
            mask_N_t = torch.zeros([self.N_t]+list(mask.size()))
            mask_N_t[int(self.N_t/2),:,:] = mask
            return torch.tensor(mask_N_t).float()
        elif sampling=="random":
            mask_N_t = torch.zeros([self.N_t]+list(mask.size()))
            mask_N_t[np.random.randint(self.N_t),:,:] = mask
            return torch.tensor(mask_N_t).float()

    def get_nadir_altimeter_mask(self, patch_ID, version="random", sample_time="1D"):
        if version == "random":
            try:
                random_tile = xr.open_zarr(f"{self.data_dir}/{np.random.randint(422):03}.zarr").sla_filtered
            except:
                 random_tile = xr.open_zarr(f"{self.data_dir}/copernicus_nadir_SSH/002.zarr").sla_filtered
            random_tile = random_tile.resample(time=sample_time).mean()
            mid = np.random.randint(int(self.N_t / 2), len(random_tile.time) - int(self.N_t / 2))
            nadir_mask = random_tile.isel(time=slice(mid - self.N_t//2, mid + self.N_t//2 + self.N_t%2))
            nadir_mask = (nadir_mask * 0 + 1).where(nadir_mask > 0, other=0)
        return torch.tensor(nadir_mask.values).float()
    
    def get_cloud_mask_timeseries(self, patch_ID):
        path = f"{self.data_dir}/HRS_SST_tiles/agg_cloud_masks/{patch_ID}.nc"
        cmask = xr.open_dataset(path).sst_filtered_q5
        mid = np.random.randint(int(self.N_t / 2), len(cmask.time) - int(self.N_t / 2))
        cmask = cmask.isel(time=slice(mid - self.N_t // 2, mid + self.N_t // 2))
        cmask = (cmask * 0 + 1).where(cmask > 0, other=0)
        return torch.tensor(cmask.values).float()

    def get_cloud_mask_rho(self):
        masks = []
        for t in range(self.N_t):
            sample_N = self.cloud_catalog_rho.isel(i_time = np.random.randint(len(self.cloud_catalog_rho.i_time)))
            sample_N_tstep = int(sample_N.patch_timestep)
            sample_N_patch_ID = str(int(sample_N.patch_id)).zfill(3)
            path = f"{self.data_dir}/HRS_SST_tiles/agg_cloud_masks/{sample_N_patch_ID}.nc"
            masks.append(~np.isnan(xr.open_dataset(path).isel(time=sample_N_tstep).sst_filtered_q5))
        return torch.tensor(np.stack(masks,axis=0)).float()


