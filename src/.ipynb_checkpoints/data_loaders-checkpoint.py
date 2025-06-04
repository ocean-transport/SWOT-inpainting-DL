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
                 standards=None, multiprocessing=False, device=None):

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
        if mask_key is None:
            return 1
        elif "swot" in str(mask_key).lower():
            return self.get_random_swot_mask()
        elif "cloud" in str(mask_key).lower():
            return self.get_cloud_mask_timeseries(patch_ID)
        else:
            raise ValueError(f"Unknown mask type: {mask_key}")

    def get_random_swot_mask(self):
        sw_corner = [-153.0, 30.0]
        ne_corner = [-149.0, 42.0]
        lon = np.random.randint(sw_corner[0], ne_corner[0])
        lat = np.random.randint(sw_corner[1], ne_corner[1])

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

'''
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
<<<<<<< HEAD
                 N=128,L_x=512e3,L_y=512e3,flatten=False,return_meta_data=True,
                 standards = (0, 1, 0, 1),
                 multiprocessing=False, device=None):
=======
                 N=128,L_x=512e3,L_y=512e3,
                 multiprocessing=False, device=None,return_masks=False):
>>>>>>> cc1915dc9d78b31cb95f6cfb42049fbdd10e3c15
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
            return_masks (bool) : Return cloud / SWOT masks, default=False
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
        self.worker_generic_swath0 = xr.open_zarr(f"{self.data_dir}/SWOT_swaths_488/hawaii_c488_p015.zarr")
        self.worker_generic_swath1 = xr.open_zarr(f"{self.data_dir}/SWOT_swaths_488/hawaii_c488_p028.zarr")

<<<<<<< HEAD
        self.flatten = flatten
        self.return_meta_data = return_meta_data
            
        # You must define these values before using them
        mean_ssh = standards["mean_ssh"]   # Replace with actual global mean
        std_ssh = standards["std_ssh"]   # Replace with actual global std
        mean_sst = standards["mean_sst"]   # Replace with actual global mean
        std_sst = standards["std_sst"]   # Replace with actual global std

        # Define the transform dictionary using top-level functions
        self.transforms = {
                "std_ssh_norm": partial(standardize, std=std_ssh),
                "std_sst_norm": partial(standardize, std=std_sst),
                "std_mean_ssh_norm": partial(standardize_samplewise, std=std_ssh),
                "std_mean_sst_norm": partial(standardize_samplewise, std=std_sst),
                "std_global_mean_ssh_norm": partial(standardize, mean=mean_ssh, std=std_ssh),
                "std_global_mean_sst_norm": partial(standardize, mean=mean_sst, std=std_sst),
                "no_transform": no_transform,
                }
=======
        self.return_masks = return_masks
>>>>>>> cc1915dc9d78b31cb95f6cfb42049fbdd10e3c15
    
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
        cloud_mask_timeseries = xr.open_dataset(f"{self.data_dir}/HRS_SST_tiles/agg_cloud_masks/{patch_ID}.nc").sst_filtered_q5
        """
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
        """
        mid_timestep_rand = np.random.randint(int(self.N_t/2), 
                                            len(cloud_mask_timeseries.time)-int(self.N_t/2))
        cloud_mask_timeseries = cloud_mask_timeseries.isel(
            time=range(int(mid_timestep_rand-self.N_t/2), 
                      int(mid_timestep_rand+int(self.N_t/2))))
        cloud_mask_timeseries = (cloud_mask_timeseries*0+1).where(cloud_mask_timeseries>0,other=0)
        return torch.tensor(cloud_mask_timeseries.values)


    def get_mask(self, mask_key, patch_ID):
        if mask_key is None:
            return 1
        elif "swot" in str(mask_key).lower():
            return self.get_random_swot_mask()
        elif "cloud" in str(mask_key).lower():
            return self.get_cloud_mask_timeseries(patch_ID)
        else:
            raise ValueError(f"Unknown mask type: {mask_key}")
    
    
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
    
            tensor = torch.tensor(var.values) * mask
            variables.append(tensor)
        return variables


    def __getitem__(self, idx):
        try:
            return self._load_patch(idx)
        except Exception as e:
            print(f"Warning: failed to load patch {idx} with error {e}, falling back to patch 065")
            return self._load_patch(patch_id="065")

    """
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
                # Pull the latitude and longitude, put in output metadata
                latitude, longitude = invar.latitude.values, invar.longitude.values
                # Pull the variable associated with the first key, assuming there's only one per .zarr file .
                # In the future each patch file should contain all of the fields I want
                invar = invar[list(invar.data_vars.keys())[0]]
                #invar_transformed = self.in_transform_list[i](invar)
                invar_transformed = self.transforms[self.in_transform_list[i]](invar)
                mask = self.get_mask(self.in_mask_list[i], patch_ID)
                invars_loaded.append(torch.tensor(invar_transformed.values)*mask)
        except Exception as e:
            # If we get an exception, automatically use a known stable patch
            patch_ID = "065"
            invars_loaded = []
            outvars_loaded = []
            for i, field in enumerate(self.infields):
                invar = xr.open_zarr(f"{self.data_dir}/{field}/{patch_ID}.zarr").isel(time=slice(int(self.mid_timestep-self.N_t/2), int(self.mid_timestep+self.N_t/2)))
                # Pull the latitude and longitude, put in output metadata
                latitude, longitude = invar.latitude.values, invar.longitude.values
                # Pull the variable associated with the first key, assuming there's only one per .zarr file .
                # In the future each patch file should contain all of the fields I want
                invar = invar[list(invar.data_vars.keys())[0]]
                #invar_transformed = self.in_transform_list[i](invar)
                invar_transformed = self.transforms[self.in_transform_list[i]](invar)
                mask = self.get_mask(self.in_mask_list[i], patch_ID)
                invars_loaded.append(torch.tensor(invar_transformed.values)*mask)
        # By the time you get here the patch_ID should be set to "065" in the event of an error
        for i, field in enumerate(self.outfields):
            outvar = xr.open_zarr(f"{self.data_dir}/{field}/{patch_ID}.zarr").isel(time=slice(int(self.mid_timestep-self.N_t/2), int(self.mid_timestep+self.N_t/2)))
            # Pull the variable associated with the first key, assuming there's only one per .zarr file .
            # In the future each patch file should contain all of the fields I want
            outvar = outvar[list(outvar.data_vars.keys())[0]]
            #outvar_transformed = self.out_transform_list[i](outvar)
            outvar_transformed = self.transforms[self.out_transform_list[i]](outvar)
            mask = self.get_mask(self.out_mask_list[i], patch_ID)
            outvars_loaded.append(torch.tensor(outvar_transformed.values)*mask)
        invar = torch.nan_to_num(torch.stack(invars_loaded, dim = 1))
        outvar = torch.nan_to_num(torch.stack(outvars_loaded, dim = 1))

        if self.flatten:
            invar = invar.flatten(0,1)
            outvar = outvar.flatten(0,1)

        metadata = {"patch_ID":patch_ID, 
                    "mid_timestep":self.mid_timestep, 
                    "patch_coords":self.patch_coords[idx],
                    "latitude":latitude,
                    "longitude":longitude}
<<<<<<< HEAD
        if self.return_meta_data:
            return invar, outvar, metadata
        else:
            return invar, outvar
        """
=======
        if self.return_masks:
            metadata["out_masks"] = [self.get_mask(self.out_mask_list[i], patch_ID) for i in range(len(self.out_mask_list))]
            metadata["in_masks"] = [self.get_mask(self.in_mask_list[i], patch_ID) for i in range(len(self.in_mask_list))]
        
        return invar, outvar, metadata


>>>>>>> cc1915dc9d78b31cb95f6cfb42049fbdd10e3c15


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def get_data_loaders(**data_loader_hparams):
    """
    Creates and returns data loaders for training, validation, and testing.
    
    Args:
        **model_hparams: Dictionary containing model hyperparameters including:
            - batch_size (int): Number of samples per batch
            - Number_timesteps (int): N_t parameter for datasets
            - mean_ssh/std_ssh (float): SSH normalization parameters
            - mean_sst/std_sst (float): SST normalization parameters
            - multiprocessing (bool): Enable multiprocessing
            
    Returns:
        tuple: (train_data_loader, val_data_loader, test_data_loader)
    """
    # Detect available device
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    
    # Define data paths
    DATASET_PATH = data_loader_hparams["DATASET_PATH"]
    CHECKPOINT_PATH = data_loader_hparams["CHECKPOINT_PATH"]
    DRIVE_PATH = data_loader_hparams["DRIVE_PATH"]

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    if device == "cpu":
        n_cpus = 0
    else:
        # Get available CPUs for parallel loading
        n_cpus = torch.get_num_threads()
        
    # Load patch coordinates (excluding land)
    patch_coords = zarr.load(f'{DATASET_PATH}/np_SST_masks/x_y_coordinates_noland.zarr')    

    # Create full dataset by concatenating datasets for different timesteps
    full_dataset = torch.utils.data.ConcatDataset([
        llc4320_dataset(
            DATASET_PATH,
            i_mid_timestep,
            model_hparams["Number_timesteps"],
            model_hparams["mean_ssh"],
            model_hparams["std_ssh"],
            model_hparams["mean_sst"],
            model_hparams["std_sst"],
            patch_coords,
            SST_quality_level=model_hparams["SST_quality_level"], 
            sst_only=model_hparams["sst_only"], 
            sst_cloud_mask=model_hparams["sst_cloud_mask"],
            multiprocessing=model_hparams["multiprocessing"]
        ) 
        for i_mid_timestep in range(30, 360, 5)  # Every 5 timesteps from 30 to 360
    ])
    print(f"size full_dataset: {len(full_dataset)}")
    
    # Split dataset into train/validation/test (70%/20%/10%)
    train_length = int(0.7 * len(full_dataset))
    validation_length = int(0.2 * len(full_dataset))
    test_length = len(full_dataset) - train_length - validation_length
    train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, 
        (train_length, validation_length, test_length))
    
    # Print dataset sizes
    print(f"size train_dataset: {len(train_dataset)}")
    print(f"size validation_dataset: {len(validation_dataset)}")
    print(f"size test_dataset: {len(test_dataset)}")

    # Worker initialization function (currently does nothing)
    def worker_init_fn(worker_id):
        worker_info = torch.utils.data.get_worker_info()
    
    # Create data loaders with parallel loading support
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=model_hparams["batch_size"],
        shuffle=True,
        num_workers=n_cpus,
        worker_init_fn=worker_init_fn,
        persistent_workers=model_hparams["multiprocessing"])
    
    val_data_loader = DataLoader(
        validation_dataset,
        batch_size=model_hparams["batch_size"],
        shuffle=True,
        num_workers=n_cpus,
        worker_init_fn=worker_init_fn,
        persistent_workers=model_hparams["multiprocessing"])
    
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=model_hparams["batch_size"],
        shuffle=True,
        num_workers=n_cpus,
        worker_init_fn=worker_init_fn,
        persistent_workers=model_hparams["multiprocessing"])
    
    return train_data_loader, val_data_loader, test_data_loader
'''
