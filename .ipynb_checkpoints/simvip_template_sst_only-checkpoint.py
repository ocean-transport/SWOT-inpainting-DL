# Load standard modules
import os
import numpy as np
import json
import math
import matplotlib.pyplot as plt

from datetime import date, timedelta
import xarray as xr
import zarr

# Pytorch modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# cuda setup, set seed for reproducability 
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
set_seed(41)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
print(f"Using device: {device}")


# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

# Set torch dtype to float64
torch.set_default_dtype(torch.float64)

# Set model path
import sys
sys.path.append('/home/tm3076/projects/NYU_SWOT_project/Inpainting_Pytorch_gen/SWOT-inpainting-DL/src')
import simvip_model
import data_loaders
import trainers



import torch
for i in range(torch.cuda.device_count()):
   print(torch.cuda.get_device_properties(i))

from importlib import reload
simvip_model = reload(simvip_model)
data_loaders = reload(data_loaders)
trainers = reload(trainers)


print(f"Using {torch.get_num_threads()} threads!")

from lightning_fabric.utilities import optimizer



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

class lightning_model_template(pl.LightningModule):

    def __init__(self, base_model, **model_kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=["base_model"])
        self.forward_model = base_model
    
    def forward(self, x):
        """
        x: patches
        Forward function is the SimVip encoder
        """
        y_hat = self.forward_model(x)
        return y_hat
    
    def _get_reconstruction_gradient_loss(self, batch, mode="train"):
        """
        Given a batch of images, this function returns the reconstruction loss (using MSE here)
        """
        x, y, meta_data = batch
        x, y = x.double(), y.double()
        y = y[:,:,np.newaxis,:,:]
        y_hat = self.forward(x)

        y_inr_grad_i = torch.gradient(y,dim=3)
        y_inr_grad_j = torch.gradient(y,dim=4)
        y_out_grad_i = torch.gradient(y_hat,dim=3)
        y_out_grad_j = torch.gradient(y_hat,dim=4)

        y_inr_grad_ii = torch.gradient(y_inr_grad_i[0],dim=3)
        y_inr_grad_jj = torch.gradient(y_inr_grad_j[0],dim=4)
        y_out_grad_ii = torch.gradient(y_out_grad_i[0],dim=3)
        y_out_grad_jj = torch.gradient(y_out_grad_j[0],dim=4)
    
        loss = F.mse_loss(y, y_hat, reduction="none")
        loss_grad_i = F.mse_loss(y_inr_grad_i[0], y_out_grad_i[0], reduction="none")
        loss_grad_j = F.mse_loss(y_inr_grad_j[0], y_out_grad_j[0], reduction="none")
        loss_grad_ii = F.mse_loss(y_inr_grad_ii[0], y_out_grad_ii[0], reduction="none")
        loss_grad_jj = F.mse_loss(y_inr_grad_jj[0], y_out_grad_jj[0], reduction="none")
    
        # Normalize the loss by the num of pixels
        loss = loss.sum(dim=[1,2,3,4]).mean(dim=[0])/(self.hparams.Number_timesteps*128*128)
        loss_grad_i = loss_grad_i.sum(dim=[1,2,3,4]).mean(dim=[0])/(self.hparams.Number_timesteps*128*128)
        loss_grad_j = loss_grad_j.sum(dim=[1,2,3,4]).mean(dim=[0])/(self.hparams.Number_timesteps*128*128)
        loss_grad_ii = loss_grad_ii.sum(dim=[1,2,3,4]).mean(dim=[0])/(self.hparams.Number_timesteps*128*128)
        loss_grad_jj = loss_grad_jj.sum(dim=[1,2,3,4]).mean(dim=[0])/(self.hparams.Number_timesteps*128*128)

        # The alphas adjust weights of the gradient loss.
        loss = self.hparams.alpha0*loss + self.hparams.alpha1*(loss_grad_i + loss_grad_j) + self.hparams.alpha2*(loss_grad_ii + loss_grad_jj)
        self.log(f'{mode}_loss', loss)
        
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,500], gamma=0.1)
        return {"optimizer":optimizer, "lr_scheduler":lr_scheduler}

    def training_step(self, batch, batch_idx):
        loss = self._get_reconstruction_gradient_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._get_reconstruction_gradient_loss(batch, mode="val")
    
    def test_step(self, batch, batch_idx):
        self._get_reconstruction_gradient_loss(batch, mode="test")


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Modules for loading data, set data pathimport urllib
DATASET_PATH = '/home/tm3076/scratch/pytorch_learning_tiles'
CHECKPOINT_PATH = '/home/tm3076/scratch/tm3076/pytorch_training'
DRIVE_PATH = "."

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
if device == "cpu":
    n_cpus = 0
    multiprocessing = False
else:
    # Get available CPUs for parallel loading
    n_cpus = torch.get_num_threads()
    multiprocessing = True

mid_timestep = 100
N_t = 20
#pre-computed global normalisation stats
mean_ssh = 0.074
std_ssh = 0.0986
mean_sst = 293.307
std_sst = 8.726 
SST_quality_level = 2

# Lambda transforms to apply to input fields
std_ssh_norm = lambda x: x/std_ssh
std_sst_norm = lambda x: x/std_sst
std_sst_norm = lambda x: (x-mean_sst)/std_sst
no_transform = lambda x: x

infields = ["zarr_llc4320_SSH_tiles_4km_filtered","zarr_llc4320_SST_tiles_4km"]
outfields = ["zarr_llc4320_SSH_tiles_4km_filtered"]
in_mask_list = ["swot","cloud"]
out_mask_list = [None]
in_transform_list = [std_ssh_norm,std_sst_norm]
out_transform_list = [std_ssh_norm]

#patch_coords = np.load(f'{DATASET_PATH}/np_SST_masks/x_y_coordinates.npy')
patch_coords = zarr.load(f'{DATASET_PATH}/x_y_coordinates_noland.zarr')
batch_size = 5

# Data loader 
full_dataset = torch.utils.data.ConcatDataset([data_loaders.llc4320_dataset(DATASET_PATH, i_mid_timestep, N_t, patch_coords, 
                                                                            infields, outfields, in_mask_list, out_mask_list, 
                                                                            in_transform_list, out_transform_list, 
                                                                            SST_quality_level = SST_quality_level, 
                                                                            multiprocessing = multiprocessing,
                                                                           ) for i_mid_timestep in range(30,360,5)])

# Split into train, validation, and test data
train_length = int(0.7*len(full_dataset))
validation_length = int(0.2*len(full_dataset))
test_length = len(full_dataset) - train_length - validation_length
train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(full_dataset, (train_length, validation_length, test_length))

# Verify size of datasets
print(f"size train_dataset: {len(train_dataset)}")
print(f"size validation_dataset: {len(validation_dataset)}")
print(f"size test_dataset: {len(test_dataset)}")

def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()

train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_cpus, worker_init_fn = worker_init_fn, persistent_workers=multiprocessing)
val_data_loader   = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, num_workers=n_cpus, worker_init_fn = worker_init_fn, persistent_workers=multiprocessing)
test_data_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=n_cpus, worker_init_fn = worker_init_fn, persistent_workers=multiprocessing)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# Friday Mar 28 9:45AM
model_hparams = {"model_name":"SimVip_sst_only",
                  "Number_timesteps":N_t,
                  "alpha0":1,
                  "alpha1":50,
                  "alpha2":50,
                  "lr":3e-5,
                  "drop":.2,
                  "drop_path":.15,
                  "multiprocessing":multiprocessing,
                 "infields":infields,
                 "outfields":outfields,
                 "in_mask_list":in_mask_list,
                 "out_mask_list":out_mask_list,
                }

base_model = simvip_model.SimVP_Model_no_skip_sst(in_shape=(N_t,len(infields),128,128),**model_hparams)

model, results = trainers.train_model(lightning_model_template(base_model,**model_hparams),
                                      device,
                                     "SimVip_sst_only", 
                                     train_data_loader, 
                                     val_data_loader, 
                                     test_data_loader,
                                     **model_hparams)


print("SimVip results", results)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
