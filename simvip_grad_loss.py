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
sys.path.append('./src/')
import simvip_model
import data_loaders



import torch
for i in range(torch.cuda.device_count()):
   print(torch.cuda.get_device_properties(i))

from importlib import reload
simvip_model = reload(simvip_model)
data_loaders = reload(data_loaders)


print(f"Using {torch.get_num_threads()} threads!")






from lightning_fabric.utilities import optimizer

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

class SimVip_Autoencoder(pl.LightningModule):

    def __init__(self, Number_timesteps=30, alpha0=0, alpha1=0, alpha2=0, lr=0.0001, **model_kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.forward_model = simvip_model.SimVP_Model_no_skip_sst(in_shape=(self.hparams.Number_timesteps,2,128,128),**model_kwargs)

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
        x, y = batch
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
 



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def train_model(training_checkpoint_path, model_name, train_data_loader, val_data_loader, test_data_loader, **model_hparams):
    
    trainer = pl.Trainer(default_root_dir=os.path.join("./", model_name),
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=1,
                         max_epochs=180,
                         callbacks=[ModelCheckpoint(save_weights_only=True, verbose=True, save_last=True, mode="min", monitor="val_loss", every_n_train_steps=20),
                                    LearningRateMonitor("epoch")])
    
    trainer.logger._log_graph = True         # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    print(f"training_checkpoint_path: {training_checkpoint_path}")
    pretrained_filename = os.path.join(training_checkpoint_path, "last.ckpt")
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        model = SimVip_Autoencoder.load_from_checkpoint(pretrained_filename) # Automatically loads the model with the saved hyperparameters
    
    elif os.path.isfile(training_checkpoint_path):
        print(f"Found checkpoint at {training_checkpoint_path}")
        model = SimVip_Autoencoder.load_from_checkpoint(training_checkpoint_path) # Load best checkpoint after training
        trainer.fit(model, train_loader, val_loader)
        model = SimVip_Autoencoder.load_from_checkpoint(trainer.checkpoint_callback.best_model_path) # Load best checkpoint after training

    else:
        print("Running from scratch")
        pl.seed_everything(42) # To be reproducable
        model = SimVip_Autoencoder(**model_hparams)
        trainer.fit(model, train_data_loader, val_data_loader)
        model = SimVip_Autoencoder.load_from_checkpoint(trainer.checkpoint_callback.best_model_path) # Load best checkpoint after training

    # Test best model on validation and test set
    val_result = trainer.test(model, val_data_loader, verbose=False)
    test_result = trainer.test(model, test_data_loader, verbose=False)
    result = {"test": test_result, "val": val_result}

    return model, result



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
# Modules for loading data, set data pathimport urllib
DATASET_PATH = '/home/tm3076/scratch/pytorch_learning_tiles'
CHECKPOINT_PATH = '/home/tm3076/scratch/tm3076/pytorch_training'
sst_dir = "zarr_llc4320_SST_tiles_4km_filtered"
ssh_dir = "zarr_llc4320_SSH_tiles_4km_filtered"
DRIVE_PATH = "."

n_cpus = torch.get_num_threads()#os.cpu_count()
mid_timestep = 100
Number_timesteps = 20
#pre-computed global normalisation stats
mean_ssh = 0.074
std_ssh = 0.0986
mean_sst = 293.307
std_sst = 8.726 
#patch_coords = np.load(f'{DATASET_PATH}/np_SST_masks/x_y_coordinates.npy')
patch_coords = zarr.load(f'{DATASET_PATH}/np_SST_masks/x_y_coordinates_noland.zarr')
batch_size = 5


multiprocessing = True
# Data loader 
full_dataset = torch.utils.data.ConcatDataset([data_loaders.llc4320_dataset(DATASET_PATH, sst_dir, ssh_dir, i_mid_timestep, Number_timesteps, mean_ssh, std_ssh, mean_sst, std_sst, patch_coords, multiprocessing = multiprocessing) for i_mid_timestep in range(30,360,5)])

print(f"size full_dataset: {len(full_dataset)}")
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

train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_cpus, worker_init_fn = worker_init_fn, persistent_workers=True)
val_data_loader   = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, num_workers=n_cpus, worker_init_fn = worker_init_fn, persistent_workers=True)
test_data_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=n_cpus, worker_init_fn = worker_init_fn, persistent_workers=True)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


traing_checkpoint_path = "None"#f'/home/tm3076/scratch/pytorch_training/{model_name}/lightning_logs/version_1/checkpoints'
#traing_checkpoint_path = '/home/tm3076/scratch/pytorch_training/SimVip_gradloss_t20_4km/lightning_logs/version_3/checkpoints'


# Friday Mar 28 9:45AM
model_hparams = {"Number_timesteps":Number_timesteps,
                  "alpha0":1,
                  "alpha1":50,
                  "alpha2":50,
                  "lr":3e-5,
                  "std_ssh": std_ssh,
                  "std_sst": std_sst,
                  "drop":.2,
                  "drop_path":.15,}

model_name = f"SimVip_gradloss_filtered_fields_v0"

model, results = train_model(traing_checkpoint_path, 
                             model_name, 
                             train_data_loader, 
                             val_data_loader, 
                             test_data_loader,
                             **model_hparams)


print("SimVip results", results)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
