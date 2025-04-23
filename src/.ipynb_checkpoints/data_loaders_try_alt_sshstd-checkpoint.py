import numpy as np
import xarray as xr
import zarr
from datetime import date, timedelta
import warnings
warnings.filterwarnings("ignore")
import os
from glob import glob


import sys
sys.path.append('../SWOT-data-analysis/src/')
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
    
    def __init__(self, data_dir, mid_timestep, N_t, mean_ssh, std_ssh, mean_sst, std_sst, patch_coords, multiprocessing = False, device=None):
        # Set device to interface with GPU
        self.device = device
        
        # Interfacing with file system
        self.data_dir = data_dir
        self.mid_timestep = mid_timestep
        self.N_t = N_t
        self.patch_coords = patch_coords
        self.max_outvar_length = 400
        
        # SSH and SST normalization constants
        self.mean_ssh = mean_ssh
        self.std_ssh = std_ssh
        self.mean_sst = mean_sst
        self.std_sst = std_sst

        self.worker_generic_swath0 = xr.open_zarr(f"{self.data_dir}/SWOT_swaths_488/hawaii_c488_p015.zarr").load()
        self.worker_generic_swath1 = xr.open_zarr(f"{self.data_dir}/SWOT_swaths_488/hawaii_c488_p028.zarr").load()
        
    
    def __len__(self):
        # The length of a sample. For now just run through all patches
        return self.patch_coords.shape[0]
    

    def get_random_swot_mask(self):
        #np.random.seed(torch.initial_seed() % 2**32)
        # I'm sampling over the cross-over region north of Hawaii, 
        # defined by the lat-lon coordinates below:
        sw_corner = [-153.0, 30.0]
        ne_corner = [-149.0, 42.0]
        #print("line 82")
        random_center_lon = np.random.randint(sw_corner[0],ne_corner[0])
        random_center_lat = np.random.randint(sw_corner[1],ne_corner[1])
        #print("self.worker_generic_swath0",self.worker_generic_swath0)
        #print("self.worker_generic_swath1",self.worker_generic_swath1)
        swot_mask0 = interp_utils.grid_everything(self.worker_generic_swath0, random_center_lat, random_center_lon,  n=128, L_x=512e3, L_y=512e3)*0+1
        swot_mask1 = interp_utils.grid_everything(self.worker_generic_swath1, random_center_lat, random_center_lon,  n=128, L_x=512e3, L_y=512e3)*0+1
        swot_mask = (swot_mask0.ssha.fillna(0) + swot_mask1.ssha.fillna(0)).values>0
        #print("line 88")
        return torch.tensor(swot_mask)

    
    def open_timeseries(self, sst_files, ssh_files):
        sst_gridded = []
        ssh_gridded = []
        
        #cloud_mask_gridded = []        
        # These timesteps span the number of 12-hourly snapshots I'll be using from llc4320
        for t_step in range(int(self.mid_timestep-self.N_t/2), int(self.mid_timestep+int(self.N_t/2))):   
            sst_dataset = xr.open_dataset(sst_files[t_step]).SST.load().fillna(0)
            ssh_dataset = xr.open_dataset(ssh_files[t_step]).Eta.load().fillna(0)
            # Mean subtraction step, removing for now...
            #sst_dataset.values[sst_dataset.values!=0] = (sst_dataset.values[sst_dataset.values!=0] - np.mean(sst_dataset.values[sst_dataset.values!=0]))/self.std_sst
            ssh_dataset.values[ssh_dataset.values!=0] = (ssh_dataset.values[ssh_dataset.values!=0] - np.mean(ssh_dataset.values[ssh_dataset.values!=0]))/self.std_ssh
            sst_gridded.append(sst_dataset.values)
            ssh_gridded.append(ssh_dataset.values)
            #cloud_mask_gridded.append(np.load(cloud_files[t_step]))
            sst_dataset.close()
            ssh_dataset.close()

        # Consolidate gridded observational fields
        sst_gridded = np.asarray(sst_gridded)
        ssh_gridded = torch.tensor(ssh_gridded)

        # Try doing normalization on the whole sst time series..?
        sst_gridded[sst_gridded!=0] = (sst_gridded[sst_gridded!=0] - np.mean(sst_gridded[sst_gridded!=0]))/self.std_sst

        return torch.tensor(sst_gridded), ssh_gridded

    
    def __getitem__(self, idx):
        patch_ID = str(int(self.patch_coords[idx,2])).zfill(3)
        
        worker_info = torch.utils.data.get_worker_info()

        # Watch out these cloud masks may not include data for all dates and thus 
        # may not be taken on exactly the same date as the model SST/SSH tiles.
        #cloud_files = sorted(glob(f"{self.data_dir}/np_SST_masks/{patch_ID}/*"))
        #print("len(sst_files)",len(sst_files),end=" ")
        #print("len(ssh_files)",len(ssh_files))
        
        try:
            sst_files = sorted(glob(f"{self.data_dir}/np_llc4320_SST_tiles/{patch_ID}/*"))
            ssh_files = sorted(glob(f"{self.data_dir}/np_llc4320_SSH_tiles/{patch_ID}/*"))
            sst_gridded, ssh_gridded = self.open_timeseries(sst_files, ssh_files)
            swot_mask_gridded = self.get_random_swot_mask()
            #cloud_mask_gridded = np.asarray(cloud_mask_gridded)
        except Exception as e:
            #with PRINT_LOCK:
            #    traceback.print_exc()
            #with open("training_logs.txt", "a") as f:
                #print(f"Error reading file {self.data_dir} for patch {patch_ID}: {e}")
            #    f.write(f"Failed to load file for patch {patch_ID}, replacing it with patch 065\n")
            sst_files = sorted(glob(f"{self.data_dir}/np_llc4320_SST_tiles/065/*"))
            ssh_files =sorted(glob(f"{self.data_dir}/np_llc4320_SSH_tiles/065/*"))
            sst_gridded, ssh_gridded = self.open_timeseries(sst_files, ssh_files)
            swot_mask_gridded = self.get_random_swot_mask()
            #cloud_mask_gridded = np.asarray(cloud_mask_gridded)

        invar = torch.stack((sst_gridded, ssh_gridded*swot_mask_gridded), dim = 1)
        outvar = ssh_gridded
        
        return invar, outvar


def custom_collate_fn(batch):
  # Filter out None values from the batch
  batch = list(filter(lambda x: x is not None, batch))
  # If the batch is empty, return an empty tensor or handle as needed
  if not batch:
      return torch.empty(0)
  return torch.stack(batch)

