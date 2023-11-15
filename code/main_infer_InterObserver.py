#%%
import torch
import os
import numpy as np
import monai
from monai.transforms import Compose, LoadImaged, ScaleIntensityRanged, EnsureChannelFirstd, CenterSpatialCropd
from monai.data import DataLoader, Dataset
from monai.metrics import DiceMetric, HausdorffDistanceMetric, RMSEMetric, MSEMetric
from monai.metrics.regression import SSIMMetric

import config
from auxiliary import utils, plotting

if config.model_name != 'InterObserver':
    raise ValueError('Attention: running InterObserver script but another model_name was specified in config_generic.py !')

observer_VG = 'contoured_ground_truth_VG'
observer_LV = 'contoured_ground_truth_LV'
if config.inference == 'validation':
    patients = config.patients_validation
elif config.inference == 'testing':
    patients = config.patients_testing
    # patients = ['abdomen_patient0006']
else:
    raise ValueError('Attention: unknown inference type!')
#%%
# GET DATA AND RUN INFERENCE

infer_transforms = Compose(
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

path_dataset_LV = os.path.join(config.path_project_data, 'testing', 'images', 'final_with_contours', observer_LV)
path_dataset_VG = os.path.join(config.path_project_data, 'testing', 'images', 'final_with_contours', observer_VG)
results_dict = {}
# loop over each patient and perform evaluation
for patient in patients:
    path_saving = os.path.join(config.path_project_results, 'inference', config.inference, 'LMU_InterObserver', config.start_time_string, patient)
    os.makedirs(path_saving, exist_ok=True)
    # path_saving  = None

    path_patient_specific = os.path.join(config.path_project_data, 'testing', 'images', 'contoured_AI_assisted', '2023_09_11_val_and_test')
    # exclude frames which were used for patient specific training from all evaluations
    path_exclusion = os.path.join(utils.subdir_paths(os.path.join(os.path.join(path_patient_specific, patient), 'raw_cine'))[0]  , 'trained')
    excluded_frames = [file for file in os.listdir(path_exclusion) if os.path.isfile(os.path.join(path_exclusion, file))]  
    # print(f'The following frames are being excluded: {excluded_frames}')
    
    # get a list with dictionaries with paths to fixed and moving images for every patient separately
    infer_files_LV = utils.get_paths_dict(path_dataset=os.path.join(path_dataset_LV, patient),
                                        moving_id=config.moving_id, seg=True, excluded_frames=excluded_frames)    
    infer_files_VG = utils.get_paths_dict(path_dataset=os.path.join(path_dataset_VG, patient),
                                        moving_id=config.moving_id, seg=True, excluded_frames=excluded_frames)
    print(f'Number of inference pairs for {patient}: {len(infer_files_VG)}')

    # Dataset (vanilla) 
    infer_ds_LV = Dataset(data=infer_files_LV, transform=infer_transforms)
    infer_ds_VG = Dataset(data=infer_files_VG, transform=infer_transforms)

    # get data into batches using Dataloaders
    infer_loader_LV = DataLoader(infer_ds_LV, batch_size=200, shuffle=False, num_workers=2)
    infer_loader_VG = DataLoader(infer_ds_VG, batch_size=200, shuffle=False, num_workers=2)

    check_data = monai.utils.first(infer_loader_LV)
    fixed_image = check_data["fixed_image"][0][0]  # eg (32,1,224,224)
    moving_image = check_data["moving_image"][0][0]
    fixed_seg = check_data["fixed_seg"][0][0] 
    moving_seg = check_data["moving_seg"][0][0]     

    # print(f"moving_image shape: {moving_image.shape}")  # (h,w)
    # print(f"fixed_image shape: {fixed_image.shape}")
    plotting.plot_example_augmentations(moving_image, fixed_image, os.path.join(path_saving, 'augmentations_example_img_LV.png'))
        
    check_data = monai.utils.first(infer_loader_VG)
    fixed_image = check_data["fixed_image"][0][0]  # eg (32,1,224,224)
    moving_image = check_data["moving_image"][0][0]
    fixed_seg = check_data["fixed_seg"][0][0] 
    moving_seg = check_data["moving_seg"][0][0]     

    # print(f"moving_image shape: {moving_image.shape}")  # (h,w)
    # print(f"fixed_image shape: {fixed_image.shape}")
    plotting.plot_example_augmentations(moving_image, fixed_image, os.path.join(path_saving, 'augmentations_example_img_VG.png'))
    
    
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False, ignore_empty=True)
    hd95_metric = HausdorffDistanceMetric(include_background=False, distance_metric='euclidean', percentile=95, 
                                                    directed=False, reduction="mean", get_not_nans=False)
    hd50_metric = HausdorffDistanceMetric(include_background=False, distance_metric='euclidean', percentile=50, 
                                                    directed=False, reduction="mean", get_not_nans=False)
    rmseSI_metric = RMSEMetric(reduction="mean", get_not_nans=False)
    rmseAP_metric = RMSEMetric(reduction="mean", get_not_nans=False)
    
    # perform evaluation and get metrics for this patient
    print('Running inference...')
    with torch.no_grad():
        for batch_data_LV, batch_data_VG in zip(infer_loader_LV, infer_loader_VG):
            inputs_LV, targets_LV, \
                inputs_seg_LV, targets_seg_LV = batch_data_LV["moving_image"].to(config.device), batch_data_LV["fixed_image"].to(config.device), \
                                                    batch_data_LV["moving_seg"].to(config.device), batch_data_LV["fixed_seg"].to(config.device)

            inputs_VG, targets_VG, \
                inputs_seg_VG, targets_seg_VG = batch_data_VG["moving_image"].to(config.device), batch_data_VG["fixed_image"].to(config.device), \
                                                    batch_data_VG["moving_seg"].to(config.device), batch_data_VG["fixed_seg"].to(config.device)
                                                                            
            # print(f'Shape of inference inputs: {inputs.shape}')  # eg (224,224)
            # print(f'Shape of inference outputs: {outputs.shape}')  # eg (224,224)         
  
            # get center of mass of segmentations for RMSE
            com_LV, com_VG = utils.get_segmentation_com(targets_seg_LV, targets_seg_VG)
                      
            # compute metrics
            dice_metric(targets_seg_LV, targets_seg_VG)
            hd95_metric(targets_seg_LV, targets_seg_VG)
            hd50_metric(targets_seg_LV, targets_seg_VG)
            rmseSI_metric(com_LV[:,0,None], com_VG[:,0,None])
            rmseAP_metric(com_LV[:,1,None], com_VG[:,1,None])
            
        # compute metrics by averaging over batches and store results
        mean_dice_metric = round(np.nanmean(dice_metric.get_buffer()), 2)
        std_dice_metric = round(np.nanstd(dice_metric.get_buffer()), 2)
        mean_hd95_metric = round(np.nanmean(hd95_metric.get_buffer()), 2)
        std_hd95_metric = round(np.nanstd(hd95_metric.get_buffer()), 2)    
        mean_hd50_metric = round(np.nanmean(hd50_metric.get_buffer()), 2)
        std_hd50_metric = round(np.nanstd(hd50_metric.get_buffer()), 2)  
        mean_rmseSI_metric = round(np.nanmean(rmseSI_metric.get_buffer()), 2)
        std_rmseSI_metric = round(np.nanstd(rmseSI_metric.get_buffer()), 2)
        mean_rmseAP_metric = round(np.nanmean(rmseAP_metric.get_buffer()), 2)
        std_rmseAP_metric = round(np.nanstd(rmseAP_metric.get_buffer()), 2)    
        
        # print(torch.min(dice_metric.get_buffer()))
        # print(torch.argwhere(dice_metric.get_buffer()[:,0] < 0.60))
        # if len(torch.argwhere(dice_metric.get_buffer()[:,0] < 0.60)) > 0:
        #     for i in range(len(torch.argwhere(dice_metric.get_buffer()[:,0] < 0.60))):
        #         print(infer_files_VG[torch.argwhere(dice_metric.get_buffer()[:,0] < 0.60)[i]]['fixed_seg'])

        # print(torch.max(hd95_metric.get_buffer()))
        # print(torch.argwhere(hd95_metric.get_buffer()[:,0] > 15))
        # if len(torch.argwhere(hd95_metric.get_buffer()[:,0] > 15)) > 0:
        #     for i in range(len(torch.argwhere(hd95_metric.get_buffer()[:,0] > 15))):
        #         print(infer_files_VG[torch.argwhere(hd95_metric.get_buffer()[:,0] > 15)[i]]['fixed_seg'])
                
                      
        if torch.isnan(dice_metric.get_buffer()).any():
            nans = True
        else:
            nans = False

    # store mean and std per patient in dict
    eval_metrics_patient = {"MSE (mean)": '-',
                            "MSE (std)": '-',
                            "SSIM (mean)": '-',
                            "SSIM (std)": '-',
                            "Dice (mean)": mean_dice_metric,
                            "Dice (std)": std_dice_metric,
                            "HD95 (mean)": mean_hd95_metric,
                            "HD95 (std)": std_hd95_metric,
                            "HD50 (mean)": mean_hd50_metric,
                            "HD50 (std)": std_hd50_metric,
                            "RMSESI (mean)": mean_rmseSI_metric,
                            "RMSESI (std)": std_rmseSI_metric,
                            "RMSEAP (mean)": mean_rmseAP_metric,
                            "RMSEAP (std)": std_rmseAP_metric,
                            "NegJ (mean)": '-',
                            "NegJ (std)": '-',
                            "NaNs excluded": nans}
 
    # update dict with patient name and eval metrics
    results_dict[patient] = eval_metrics_patient
    
    # save metrics for all frames for each patient in txt for later analysis
    np.savetxt(os.path.join(path_saving, 'dice_metric.txt'), dice_metric.get_buffer().squeeze().detach().cpu().numpy())
    np.savetxt(os.path.join(path_saving, 'hd95_metric.txt'), hd95_metric.get_buffer().squeeze().detach().cpu().numpy())
    np.savetxt(os.path.join(path_saving, 'hd50_metric.txt'), hd50_metric.get_buffer().squeeze().detach().cpu().numpy())
    np.savetxt(os.path.join(path_saving, 'rmseSI_metric.txt'), rmseSI_metric.get_buffer().squeeze().detach().cpu().numpy())
    np.savetxt(os.path.join(path_saving, 'rmseAP_metric.txt'), rmseAP_metric.get_buffer().squeeze().detach().cpu().numpy())

#%%
# SAVE OVERALL RESULTS
path_saving_overall = os.path.dirname(path_saving)   # strip patient name folder from path
utils.results_dict_to_csv(results_dict, os.path.join(path_saving_overall, 'results.csv'))  # save results for each patient tgo csv
utils.add_summary_row(csv_file=os.path.join(path_saving_overall, 'results.csv'))  # calculcate mean over all patients and add it to csv
