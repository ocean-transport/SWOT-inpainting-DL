import numpy as np
import xarray as xr
import zarr
from datetime import date, timedelta
import warnings
warnings.filterwarnings("ignore")
import os
from glob import glob


import sys
sys.path.append('/home/tm3076/projects/NYU_SWOT_project/SWOT-data-analysis/src')
import swot_utils
import data_loaders
import interp_utils

import torch
from torch.utils.data import Dataset, DataLoader


import traceback
import threading

PRINT_LOCK = threading.Lock()

class llc4320_dataset(Dataset):
    """
    A PyTorch Dataset for loading and preparing SSH and SST data from llc4320 for pytorch inference.

    Attributes:
        data_dir (str): The directory containing all input data files.
        mid_date (datetime.date): The middle date of the time period to extract data, i.e. the desired reconstruction date.
        N_t (int): The number of time steps (days) to include in the dataset.
        mean_ssh (float): Mean SSH value for standardization.
        std_ssh (float): Standard deviation of SSH values for standardization.
        mean_sst (float): Mean SST value for standardization.
        std_sst (float): Standard deviation of SST values for standardization.
        coord_grids (np.ndarray): Coordinates for the output subdomain grids.
        n (int, optional): Number of bins in the output grid (default: 128).
        L_x (float, optional): Longitudinal range of the output grid in meters (default: 960e3).
        L_y (float, optional): Latitudinal range of the output grid in meters (default: 960e3).
        multiprocessing (bool, optional): Whether to use multiprocessing for data loading (default: True).
        
    Methods:
        __len__(): Returns the number of samples in the dataset.
        worker_init_fn(worker_id): Initializes worker processes for multiprocessing.
        __getitem__(idx): Returns a tuple containing input and output data for the given index.

            Input Data (invar): A tensor of shape (N_t, 2, n, n) containing gridded SST and SSH data.
            Output Data (outvar): A tensor containing SSH data.
    """
    
    def __init__(self, data_dir, mid_timestep, N_t, patch_coords, 
                 infields, outfields, in_mask_list, out_mask_list, 
                 in_transform_list, out_transform_list,
                 SST_quality_level=1, sst_only=False, sst_cloud_mask=False,
                 N=128,L_x=512e3,L_y=512e3,
                 multiprocessing=False, device=None):
        """
        Initialize the dataset with paths, normalization parameters, and processing flags.
        
        Args:
            data_dir (str): Path to directory containing data files
            mid_timestep (int): Central timestep for data extraction
            N_t (int): Number of timesteps to load (centered around mid_timestep)
            mean_ssh/std_ssh (float): SSH normalization parameters  
            mean_sst/std_sst (float): SST normalization parameters
            patch_coords (np.ndarray): Array of patch coordinates
            SST_quality_level (int): Minimum quality level for SST cloud masking
            sst_only (bool): If True, only load SST data
            sst_cloud_mask (bool): If True, apply cloud masking to SST
            multiprocessing (bool): Enable multiprocessing support
            device (str): Device to load data onto (e.g., 'cuda')
        """
        # Set device to interface with GPU
        # Device configuration (GPU/CPU)
        self.device = device
        
        # File system and data parameters
        self.data_dir = data_dir
        self.mid_timestep = mid_timestep
        self.N_t = N_t
        self.patch_coords = patch_coords
        self.max_outvar_length = 400  # Maximum length of output variables

        # Add functionality to feed in different masks and fields
        self.infields = infields # [SST, SSH, etc]
        self.outfields = outfields # [SST, SSH, etc]
        self.in_mask_list = in_mask_list # ["cloud mask", "SWOT mask", None,]
        self.out_mask_list = out_mask_list # ["cloud mask", "SWOT mask", None,]
        self.in_transform_list = in_transform_list # [lamda x: (x - mean_ssh)/std_ssh, ...]
        self.out_transform_list = out_transform_list # [lamda x: (x - mean_ssh)/std_ssh, ...]

        # Cloud masking flags
        self.SST_quality_level = SST_quality_level

        # SWOT masking flags
        self.N = N
        self.L_x = L_x
        self.L_y = L_y

        # Preload SWOT swath templates for masking
        self.worker_generic_swath0 = xr.open_zarr(f"{self.data_dir}/SWOT_swaths_488/hawaii_c488_p015.zarr").load()
        self.worker_generic_swath1 = xr.open_zarr(f"{self.data_dir}/SWOT_swaths_488/hawaii_c488_p028.zarr").load()
        
    
    def __len__(self):
        # The length of a sample. For now just run through all patches
        return self.patch_coords.shape[0]
    

    def get_random_swot_mask(self):
        # I'm sampling over the cross-over region north of Hawaii, 
        # defined by the lat-lon coordinates below:
        sw_corner = [-153.0, 30.0]
        ne_corner = [-149.0, 42.0]
        random_center_lon = np.random.randint(sw_corner[0],ne_corner[0])
        random_center_lat = np.random.randint(sw_corner[1],ne_corner[1])
        swot_mask0 = interp_utils.grid_everything(self.worker_generic_swath0, random_center_lat, random_center_lon,  n=self.N, L_x=self.L_x, L_y=self.L_y)*0+1
        swot_mask1 = interp_utils.grid_everything(self.worker_generic_swath1, random_center_lat, random_center_lon,  n=self.N, L_x=self.L_x, L_y=self.L_y)*0+1
        swot_mask = (swot_mask0.ssha.fillna(0) + swot_mask1.ssha.fillna(0)).values>0

        return torch.tensor(swot_mask*1)

    
    def get_cloud_mask_timeseries(self, patch_ID):
        # Load aggregated cloud masks for the patch
        cloud_mask_timeseries = xr.open_zarr(f"{self.data_dir}/Cloud_masks_aggregated/{patch_ID}_aggregate.zarr")
        # Handle edge cases where requested timesteps exceed available data
        if self.mid_timestep + int(self.N_t/2) > len(cloud_mask_timeseries.time):
            mid_timestep_rand = np.random.randint(int(self.N_t/2), 
                                                len(cloud_mask_timeseries.time)-int(self.N_t/2))
            cloud_mask_timeseries = cloud_mask_timeseries.isel(
                time=range(int(mid_timestep_rand-self.N_t/2), 
                          int(mid_timestep_rand+int(self.N_t/2))))
        else:
            cloud_mask_timeseries = cloud_mask_timeseries.isel(
                time=range(int(self.mid_timestep-self.N_t/2), 
                          int(self.mid_timestep+int(self.N_t/2))))
        # Apply quality level threshold
        cloud_mask_timeseries_ql = cloud_mask_timeseries.quality_level >= self.SST_quality_level
        
        return torch.tensor(cloud_mask_timeseries_ql.values*1)


    def get_mask(self, mask_key, patch_ID):
        if "swot" in str(mask_key):
            return self.get_random_swot_mask()
        if "cloud" in str(mask_key):
            return self.get_cloud_mask_timeseries(patch_ID)
        else:
            return 1
            
    
    def __getitem__(self, idx):
        patch_ID = str(int(self.patch_coords[idx,2])).zfill(3)
        worker_info = torch.utils.data.get_worker_info()
        try:
            # Loop through the fields in the "try" block to make sure
            # to catch cases where a patches may be absent in one field 
            # but not the other..
            invars_loaded = []
            outvars_loaded = []
            for i, field in enumerate(self.infields):
                invar = xr.open_zarr(f"{self.data_dir}/{field}/{patch_ID}.zarr").isel(time=slice(int(self.mid_timestep-self.N_t/2), int(self.mid_timestep+self.N_t/2)))
                # Pull the variable associated with the first key, assuming there's only one per .zarr file .
                # In the future each patch file should contain all of the fields I want
                invar = invar[list(invar.data_vars.keys())[0]]
                invar_transformed = self.in_transform_list[i](invar)
                mask = self.get_mask(self.in_mask_list[i], patch_ID)
                invars_loaded.append(torch.tensor(invar_transformed.values)*mask)
        except Exception as e:
            # If we get an exception, automatically use a known stable patch
            patch_ID = "065"
            invars_loaded = []
            outvars_loaded = []
            for i, field in enumerate(self.infields):
                invar = xr.open_zarr(f"{self.data_dir}/{field}/{patch_ID}.zarr").isel(time=slice(int(self.mid_timestep-self.N_t/2), int(self.mid_timestep+self.N_t/2)))
                # Pull the variable associated with the first key, assuming there's only one per .zarr file .
                # In the future each patch file should contain all of the fields I want
                invar = invar[list(invar.data_vars.keys())[0]]
                invar_transformed = self.in_transform_list[i](invar)
                mask = self.get_mask(self.in_mask_list[i], patch_ID)
                invars_loaded.append(torch.tensor(invar_transformed.values)*mask)
        # By the time you get here the patch_ID should be ok
        for i, field in enumerate(self.outfields):
            outvar = xr.open_zarr(f"{self.data_dir}/{field}/{patch_ID}.zarr").isel(time=slice(int(self.mid_timestep-self.N_t/2), int(self.mid_timestep+self.N_t/2)))
            # Pull the variable associated with the first key, assuming there's only one per .zarr file .
            # In the future each patch file should contain all of the fields I want
            outvar = outvar[list(outvar.data_vars.keys())[0]]
            outvar_transformed = self.out_transform_list[i](outvar)
            mask = self.get_mask(self.out_mask_list[i], patch_ID)
            outvars_loaded.append(torch.tensor(outvar_transformed.values)*mask)   

        invar = torch.stack(invars_loaded, dim = 1)
        outvar = torch.stack(outvars_loaded, dim = 1)

        metadata = {"patch_ID":patch_ID, "mid_timestep":self.mid_timestep, "patch_coords":self.patch_coords[idx]}
        
        return invar, outvar, metadata

