#%%
import os
import time
import numpy as np
import torch
from models.TransMorph import dict_model_variants
import models.TransMorph as TransMorph

from torchio.transforms import RandomElasticDeformation, RandomMotion, RandomGhosting

import monai
from monai.transforms import Compose, LoadImaged, RandGibbsNoised, ScaleIntensityRanged, RandAffined, RandBiasFieldd, EnsureChannelFirstd, CenterSpatialCropd, RandGaussianSmoothd, RandGaussianNoised
from monai.transforms import SqueezeDimd, RepeatChanneld, RemoveRepeatedChanneld, AsDiscreted
from monai.data import DataLoader, Dataset, CacheDataset, PersistentDataset, CacheNTransDataset
from monai.optimizers import WarmupCosineSchedule

import config
from auxiliary import utils, train_val_test, plotting

if config.model_name != 'TransMorph2D':
    raise ValueError('Attention: running TransMorph2D script but another model_name was specified in config_generic.py !')

supervised_time_string = '_supervised_' + time.strftime("%Y-%m-%d-%H:%M:%S")  

import wandb
if config.wandb_usage:
    # login to weights and biases
    wandb.login() 
    # start a new wandb run to track this script
    wandb.init(
        dir=config.path_project_results,
        
        # set the wandb project where this run will be logged
        project="target_segmentation",
        # project="target_segmentation_debug",

        name=config.start_time_string + supervised_time_string,
        
        # track hyperparameters and run metadata
        config={
        "dataset": config.dataset,
        "moving_id": config.moving_id,    
        "model_name": config.model_name,
        "model_variant": dict_model_variants[config.model_variant].items(),
        "batch_size": config.batch_size,
        "lr": config.lr,
        "lr_scheduler": config.lr_scheduler,
        "epoch_nr": config.epoch_nr,
        "load_state": config.load_state,
        "loss_name": config.loss_name,
        "loss_weights": config.loss_weights,
        "prob_randaffine": config.prob_randaffine,
        "prob_randbiasfield": config.prob_randbiasfield,
        "prob_randgibbsnoise": config.prob_randgibbsnoise,
        "prob_randgaussiansmooth": config.prob_randgaussiansmooth,
        }
    )

#%%
# GET DATA

# settings paths
path_saving = os.path.join(config.path_project_results, 'training', config.model_name, config.start_time_string + supervised_time_string)
os.makedirs(path_saving, exist_ok=True)
# path_saving = None
path_loading = os.path.join(config.path_project_results, 'training', config.model_name, config.start_time_string)

# get a list with dictionaries with paths to fixed and moving images and labels
train_files = []
# pool data from all training patients here
for patient in config.patients_training:
    path_dataset_supervised = os.path.join(config.path_project_data, 'testing', 'images', 'final_with_contours', config.observer_training, patient)
    train_files.extend(utils.get_paths_dict(path_dataset=path_dataset_supervised, moving_id=config.moving_id, seg=True))
print(f'Number of supervised training image pairs: {len(train_files)}')

val_files = []
# pool data from all validation patients here
for patient in config.patients_validation:
    path_dataset_supervised = os.path.join(config.path_project_data, 'testing', 'images', 'final_with_contours', config.observer_validation, patient)
    val_files.extend(utils.get_paths_dict(path_dataset=path_dataset_supervised, moving_id=config.moving_id, seg=True))
print(f'Number of supervised validation image pairs: {len(val_files)}')


train_transforms = Compose(
    [
        LoadImaged(keys=["fixed_image", "moving_image", "fixed_seg", "moving_seg"]),
        EnsureChannelFirstd(keys=("fixed_image", "moving_image", "fixed_seg", "moving_seg"), channel_dim=-1),  # images have shape (270,270,1)
        ScaleIntensityRanged(
            keys=["fixed_image", "moving_image"],
            a_min=0,
            a_max=1000,
            b_min=0.0,
            b_max=1.0,
            clip=False,
        ),
        RandAffined(
            # keys=["fixed_image", "moving_image", "fixed_seg", "moving_seg"],
            keys=["moving_image", "moving_seg"],
            # mode=("bilinear", "bilinear", "nearest", "nearest"),
            mode=("bilinear", "nearest"),
            prob=config.prob_randaffine,
            spatial_size=(224, 224),
            rotate_range=(0, np.pi / 15),
            shear_range=(0.2, 0.2), 
            translate_range=(20,20),
            scale_range=(0.2, 0.2),
        ),
        RepeatChanneld(keys=("moving_image", "moving_seg"), repeats=4),  # duplicate 3rd dimension --> (4,h,w)
        EnsureChannelFirstd(keys=("moving_image", "moving_seg"), channel_dim="no_channel"),  # add fake 4th dim for torchio --> (1,4,h,w)
        RandomElasticDeformation(
            include=["moving_image", "moving_seg"], 
            num_control_points=(5,8,8),
            max_displacement=(1,18,18),    
            image_interpolation = "linear",
            label_interpolation = "nearest",  # somehow this does not work --> binarize manually
        ),
        AsDiscreted(keys=("moving_seg"), threshold=0.5),
        RandomMotion(
            include=["moving_image"], 
            degrees=45, 
            translation=50,
            p=config.prob_randmotion,
        ),
        SqueezeDimd(keys=("moving_image", "moving_seg"), dim=0),  # remove fake 4th dim --> (4,h,w)
        RemoveRepeatedChanneld(keys=("moving_image", "moving_seg"), repeats=4), # remove repetions--> (1,h,w)
        RandBiasFieldd(
            keys=["fixed_image", "moving_image"],
            # keys=["fixed_image"],
            coeff_range=(0,0.4),
            prob=config.prob_randbiasfield,                
        ),
        RandGibbsNoised(
            keys=["fixed_image", "moving_image"],
            prob=config.prob_randgibbsnoise, alpha=(0.1, 0.9)
        ),
        RandGaussianSmoothd(
            keys=["fixed_image", "moving_image"],
            prob=config.prob_randgaussiansmooth, approx='erf',
            sigma_x=(0.25, 1.5), sigma_y=(0.25, 1.5), 
        ),
        CenterSpatialCropd(
            keys=["fixed_image", "moving_image", "fixed_seg", "moving_seg"],
            roi_size=[224,224]
        )
    ]
)    


val_transforms = Compose(
    [
        LoadImaged(keys=["fixed_image", "moving_image", "fixed_seg", "moving_seg"]),
        EnsureChannelFirstd(keys=("fixed_image", "moving_image", "fixed_seg", "moving_seg"), channel_dim=-1),  # images/segmentations have shape (h,w,1)
        ScaleIntensityRanged(
            keys=["fixed_image", "moving_image"],
            a_min=0,
            a_max=1000,
            b_min=0.0,
            b_max=1.0,
            clip=False,
        ),
        CenterSpatialCropd(
            keys=["fixed_image", "moving_image", "fixed_seg", "moving_seg"],
            roi_size=[224,224]
        )
    ]
)    

# Dataset (vanilla) 
# train_ds = Dataset(data=train_files, transform=train_transforms)
# val_ds = Dataset(data=val_files, transform=val_transforms)
    
# CacheDataset precompute all non-random transforms of original data and store in memory, might run out if dataset large
train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=4)
val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=2)

# path_persistent_cache = os.path.join(config_generic.path_project_data, 'training', 'persistent_cache', config_generic.start_time_string)
# os.makedirs(path_persistent_cache, exist_ok=True)     
# PersisteneDataset uses persistent storage of non-random transformed training and validation data computed once and stored in persistently across runs
# train_ds = PersistentDataset(data=train_files, transform=train_transforms, cache_dir=path_persistent_cache)
# val_ds = PersistentDataset(data=val_files, transform=val_transforms, cache_dir=path_persistent_cache)

# CacheNTrans is an extension opf PersistenDataset which caches the result of first N transforms, no matter it's random or not.
# train_ds = CacheNTransDataset(data=train_files, transform=train_transforms, cache_dir=path_persistent_cache, cache_n_trans=10)
# val_ds = CacheNTransDataset(data=val_files, transform=val_transforms, cache_dir=path_persistent_cache, cache_n_trans=10)

# get data into batches using Dataloaders
train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=False, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=200, shuffle=False, num_workers=2)
    
check_data = monai.utils.first(train_loader)
fixed_image = check_data["fixed_image"][0][0]  # eg (32,1,224,224)
moving_image = check_data["moving_image"][0][0]
fixed_seg = check_data["fixed_seg"][0][0] 
moving_seg = check_data["moving_seg"][0][0]     

# print(f"moving_image shape: {moving_image.shape}")  # (h,w)
# print(f"fixed_image shape: {fixed_image.shape}")
plotting.plot_example_augmentations(moving_image, fixed_image, os.path.join(path_saving, 'augmentations_example_img.png'))
    
# print(f"moving_seg shape: {moving_seg.shape}")  # (h,w)
# print(f"fixed_seg shape: {fixed_seg.shape}")
plotting.plot_example_augmentations(moving_seg, fixed_seg, os.path.join(path_saving, 'augmentations_example_seg.png'))

#%%
# GET MODEL
model_config = dict_model_variants[config.model_variant]
model = TransMorph.TransMorph(model_config)
model.to(config.device)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total number of trainable parameters: {pytorch_total_params}\n')

# SET OPTIMIZER
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

# CONTINUING FROM PREVIOUS TRAINING
if config.load_state:
    print(f'Continuing previous training...')
    if 'LNCC' in config.loss_name:
        neg_loss=True
    else:
        neg_loss=False
    path_best_model = utils.get_path_to_best_model(path_loading, neg_loss=neg_loss)
    print(f'Path to best model: {path_best_model}')  
    model.load_state_dict(torch.load(path_best_model, map_location=config.device))
    path_best_optimizer = utils.get_path_to_best_optimizer(path_loading, neg_loss=neg_loss)
    print(f'Path to best optimizer: {path_best_optimizer}\n')
    optimizer.load_state_dict(torch.load(path_best_optimizer, map_location=config.device))
    epoch_start = int(path_best_optimizer[-25:-22]) + 1
    # update learning rate
    for g in optimizer.param_groups:
        g['lr'] = config.lr
else:
    epoch_start = 0


if config.lr_scheduler == 'WarmupCosine':
    if config.load_state is False:
        lr_scheduler = WarmupCosineSchedule(optimizer, warmup_steps=len(train_loader)*2, 
                                            t_total=len(train_loader)*config.epoch_nr*2,
                                            warmup_multiplier=0.5, last_epoch=-1)
    else:
       lr_scheduler = None 
else:     
    lr_scheduler = None

        
#%%
# TRANING LOOP
train_val_test.train_val_model_supervised(model, train_loader, optimizer, config.loss_name, config.loss_weights,
                                        epoch_nr=config.epoch_nr, device=config.device, 
                                        lr_scheduler=lr_scheduler, early_stopping_patience=config.epoch_nr//5, epoch_start=epoch_start,
                                        val_loader=val_loader, path_saving=path_saving, wandb_usage=config.wandb_usage, plot=config.plot)

# remove persistent cache dir
# print('Removing persistent_cache directory...')
# shutil.rmtree(path_persistent_cache)
