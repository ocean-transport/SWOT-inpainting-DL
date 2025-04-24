import os

# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def train_model(base_model, device, training_checkpoint_path, train_data_loader, val_data_loader, test_data_loader, **model_hparams):
    
    training_checkpoint_path = os.path.join("./", model_hparams["model_name"])
    
    # Define trainer
    trainer = pl.Trainer(default_root_dir=training_checkpoint_path,
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
        model = base_model.load_from_checkpoint(pretrained_filename) # Automatically loads the model with the saved hyperparameters
    elif os.path.isfile(training_checkpoint_path):
        print(f"Found checkpoint at {training_checkpoint_path}")
        model = base_model.load_from_checkpoint(training_checkpoint_path) # Load best checkpoint after training
        trainer.fit(model, train_loader, val_loader)
        model = base_model.load_from_checkpoint(trainer.checkpoint_callback.best_model_path) # Load best checkpoint after training
    else:
        print("Running from scratch")
        pl.seed_everything(42) # To be reproducable
        model = base_model#(**model_hparams)
        trainer.fit(model, train_data_loader, val_data_loader)
        model = base_model.load_from_checkpoint(trainer.checkpoint_callback.best_model_path) # Load best checkpoint after training

    # Test best model on validation and test set
    val_result = trainer.test(model, val_data_loader, verbose=False)
    test_result = trainer.test(model, test_data_loader, verbose=False)
    result = {"test": test_result, "val": val_result}

    return model, result
