#%%
import torch
import os
import monai
import numpy as np
from monai.transforms import Compose, LoadImaged, ScaleIntensityRanged, EnsureChannelFirstd, CenterSpatialCropd
from monai.data import DataLoader, Dataset
from monai.metrics import DiceMetric, HausdorffDistanceMetric, RMSEMetric, MSEMetric
from monai.metrics.regression import SSIMMetric
import nibabel as nb


import config
import models.plastimatch_bspline as plastimatch_bspline
from auxiliary import utils, plotting, metrics

if config.model_name != 'BSpline':
    raise ValueError('Attention: running BSpline script but another model_name was specified in config_generic.py !')

if config.inference == 'validation':
    observer = config.observer_validation
    patients = config.patients_validation
elif config.inference == 'testing':
    observer = config.observer_testing
    patients = config.patients_testing
else:
    raise ValueError('Attention: unknown inference type!')
#%%
# GET DATA AND RUN INFERENCE

infer_transforms = Compose(
    [
        LoadImaged(keys=["fixed_image", "moving_image", "fixed_seg", "moving_seg"]),
        EnsureChannelFirstd(keys=("fixed_image", "moving_image", "fixed_seg", "moving_seg"), channel_dim=-1),  # images/segmentations have shape (h,w,1)
        # ScaleIntensityRanged(
        #     keys=["fixed_image", "moving_image"],
        #     a_min=0,
        #     a_max=1000,
        #     b_min=0.0,
        #     b_max=1.0,
        #     clip=False,
        # ),
        CenterSpatialCropd(
            keys=["fixed_image", "moving_image", "fixed_seg", "moving_seg"],
            roi_size=[224,224]
        )
    ]
)    

path_dataset = os.path.join(config.path_project_data, 'testing', 'images', 'final_with_contours', observer)
path_patient_specific = os.path.join(config.path_project_data, 'testing', 'images', 'contoured_AI_assisted', '2023_09_11_val_and_test')
results_dict = {}
# loop over each patient and perform evaluation
for patient in patients:
    path_saving = os.path.join(config.path_project_results, 'inference', config.inference, 'LMU_observer_' + observer[-2:], config.model_name, config.start_time_string, patient)
    os.makedirs(path_saving, exist_ok=True)
    # path_saving  = None

    # exclude frames which were used for patient specific training from all evaluations
    path_exclusion = os.path.join(utils.subdir_paths(os.path.join(os.path.join(path_patient_specific, patient), 'raw_cine'))[0]  , 'trained')
    excluded_frames = [file for file in os.listdir(path_exclusion) if os.path.isfile(os.path.join(path_exclusion, file))]  
    # print(f'The following frames are being excluded: {excluded_frames}')
       
    # get a list with dictionaries with paths to fixed and moving images for every patient separately
    infer_files = utils.get_paths_dict(path_dataset=os.path.join(path_dataset, patient),
                                        moving_id=config.moving_id, seg=True, excluded_frames=excluded_frames)
    print(f'Number of inference pairs for {patient}: {len(infer_files)}')

    # Dataset (vanilla) 
    infer_ds = Dataset(data=infer_files, transform=infer_transforms)

    # get data into batches using Dataloaders
    infer_loader = DataLoader(infer_ds, batch_size=1, num_workers=1, shuffle=False)

    check_data = monai.utils.first(infer_loader)
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
    
    
    # set data range for structural similiraty
    data_range = torch.tensor([1.0], device=config.device)
       
    mse_metric = MSEMetric(reduction="mean", get_not_nans=False)
    ssim_metric = SSIMMetric(data_range=data_range, win_size=7, k1=0.01, k2=0.03, spatial_dims=2, reduction="mean", get_not_nans=False)
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False, ignore_empty=True)
    hd95_metric = HausdorffDistanceMetric(include_background=False, distance_metric='euclidean', percentile=95, 
                                                    directed=False, reduction="mean", get_not_nans=False)
    hd50_metric = HausdorffDistanceMetric(include_background=False, distance_metric='euclidean', percentile=50, 
                                                    directed=False, reduction="mean", get_not_nans=False)
    negj_metric = metrics.NegativeJacobianMetric(reduction="mean", get_not_nans=False)
    rmseSI_metric = RMSEMetric(reduction="mean", get_not_nans=False)
    rmseAP_metric = RMSEMetric(reduction="mean", get_not_nans=False)
    
    
    # as plastimatch runs in the terminal, the data needs unfortunately to be saved on disk...
    # initialize parameters for (temporary) data saving before
    affine=np.eye(4) # define the affine matrix to correctly orient image in vv, no effect in python
    affine[0,0]=0
    affine[0,1]=-1
    affine[1,1]=0
    affine[1,0]=-1
    affine[2,2]=1
    affine[3,3]=1
    os.makedirs(os.path.join(path_saving, 'tmp_data'), exist_ok=True)
    par_file = os.path.join(os.path.dirname(path_saving), 'bspline_parameters.txt')
    log_file = os.path.join(os.path.dirname(path_saving), 'log_file.txt')
    
    # perform evaluation and get metrics for this patient
    file_nr = 0
    print('Running inference...')
    with torch.no_grad():
        for batch_data in infer_loader:
            file_nr += 1
            print(f'----- File {file_nr}/{len(infer_files)} for current patient -----')
            
            inputs, targets, \
                inputs_seg, targets_seg = batch_data["moving_image"].squeeze().to(config.device), batch_data["fixed_image"].squeeze().to(config.device), \
                                                    batch_data["moving_seg"].squeeze().to(config.device), batch_data["fixed_seg"].squeeze().to(config.device)
            # print(f'Shape of inference inputs: {inputs.shape}')  # eg (224,224)
            # print(f'Shape of inference outputs: {outputs.shape}')  # eg (224,224) 
            
            # define paths and save current images to disk with shape (h,w,1) for plastimatch
            fixed = os.path.join(path_saving, 'tmp_data', 'fixed.nii.gz')
            targets_nifti = nb.Nifti1Image(targets[:,:,None].detach().cpu().numpy(), affine=affine)
            nb.save(targets_nifti, fixed)
            moving = os.path.join(path_saving, 'tmp_data', 'moving.nii.gz')
            inputs_nifti = nb.Nifti1Image(inputs[:,:,None].detach().cpu().numpy(), affine=affine)
            nb.save(inputs_nifti, moving)
            moving_seg = os.path.join(path_saving, 'tmp_data', 'moving_seg.nii.gz')
            inputs_seg_nifti = nb.Nifti1Image(inputs_seg[:,:,None].detach().cpu().numpy(), affine=affine)
            nb.save(inputs_seg_nifti, moving_seg)           
            output = os.path.join(path_saving, 'tmp_data', 'output.nii.gz')
            output_seg = os.path.join(path_saving, 'tmp_data', 'output_seg.nii.gz')
            xform_out = os.path.join(path_saving, 'tmp_data', 'xf.txt')
            vf = os.path.join(path_saving, 'tmp_data', 'vf.nii.gz')
            output_grid = os.path.join(path_saving, 'tmp_data', 'grid_output.nii.gz')

            # perform registration betwwen fixed and moving frames
            outputs = plastimatch_bspline.run_plastimatch_DIR(config.lambda_value, config.init_grid_spac, 
                                                    par_file, log_file, fixed, moving, output, xform_out, vf, 
                                                    metric=config.metric, max_its=config.max_its, 
                                                    impl=config.impl, load_output=True)
            
            # warp input segmentation
            outputs_seg, ddf = plastimatch_bspline.apply_existing_vf(log_file, vf, 
                                                            output=output_seg,  moving=moving_seg, warp_grid=False, load_output=True)
            
            # warp regular grid to see effect of deformation
            plastimatch_bspline.apply_existing_vf(log_file, xform_out, 
                                                output=output_grid, moving='', warp_grid=True)
            
            
            # get dim back to b,c,h,w and to tensor for metrics and plotting functions
            inputs = inputs.unsqueeze(0).unsqueeze(0)
            outputs = torch.tensor(outputs, dtype=torch.float32).squeeze().unsqueeze(0).unsqueeze(0)
            targets = targets.unsqueeze(0).unsqueeze(0)
            inputs_seg = inputs_seg.unsqueeze(0).unsqueeze(0)
            outputs_seg = torch.tensor(outputs_seg, dtype=torch.float32).squeeze().unsqueeze(0).unsqueeze(0)
            targets_seg = targets_seg.unsqueeze(0).unsqueeze(0)
            ddf = torch.moveaxis(torch.tensor(ddf, dtype=torch.float32).squeeze().unsqueeze(0), -1, 1)[:, 0:2, ...]  # -> (1,2,224,224)
                       
            # get center of mass of segmentations for RMSE
            outputs_com, targets_com = utils.get_segmentation_com(outputs_seg, targets_seg)
            
            # compute metrics
            mse_metric(outputs, targets)
            ssim_metric(outputs, targets)
            dice_metric(outputs_seg, targets_seg)
            hd95_metric(outputs_seg, targets_seg)
            hd50_metric(outputs_seg, targets_seg)
            negj_metric(ddf)
            rmseSI_metric(outputs_com[:,0,None], targets_com[:,0,None])
            rmseAP_metric(outputs_com[:,1,None], targets_com[:,1,None])
            
        # compute metrics by averaging over batches and store results
        mean_mse_metric = round(np.nanmean(mse_metric.get_buffer()), 4)
        std_mse_metric = round(np.nanstd(mse_metric.get_buffer()), 4)
        mean_ssim_metric = round(np.nanmean(ssim_metric.get_buffer()), 4)
        std_ssim_metric = round(np.nanstd(ssim_metric.get_buffer()), 4)
        mean_dice_metric = round(np.nanmean(dice_metric.get_buffer()), 2)
        std_dice_metric = round(np.nanstd(dice_metric.get_buffer()), 2)
        mean_hd95_metric = round(np.nanmean(hd95_metric.get_buffer()), 2)
        std_hd95_metric = round(np.nanstd(hd95_metric.get_buffer()), 2)    
        mean_hd50_metric = round(np.nanmean(hd50_metric.get_buffer()), 2)
        std_hd50_metric = round(np.nanstd(hd50_metric.get_buffer()), 2)       
        mean_negj_metric = round(np.nanmean(negj_metric.get_buffer()), 2)
        std_negj_metric = round(np.nanstd(negj_metric.get_buffer()), 2)    
        mean_rmseSI_metric = round(np.nanmean(rmseSI_metric.get_buffer()), 2)
        std_rmseSI_metric = round(np.nanstd(rmseSI_metric.get_buffer()), 2)
        mean_rmseAP_metric = round(np.nanmean(rmseAP_metric.get_buffer()), 2)
        std_rmseAP_metric = round(np.nanstd(rmseAP_metric.get_buffer()), 2)
        
    # store mean and std per patient in dict
    eval_metrics_patient = {"MSE (mean)": mean_mse_metric,
                            "MSE (std)": std_mse_metric,
                            "SSIM (mean)": mean_ssim_metric,
                            "SSIM (std)": std_ssim_metric,
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
                            "NegJ (mean)": mean_negj_metric,
                            "NegJ (std)": std_negj_metric}
 
    # update dict with patient name and eval metrics
    results_dict[patient] = eval_metrics_patient
    
       
    # save metrics for all frames for each patient in txt for later analysis
    np.savetxt(os.path.join(path_saving, 'mse_metric.txt'), mse_metric.get_buffer().squeeze().detach().cpu().numpy())
    np.savetxt(os.path.join(path_saving, 'ssim_metric.txt'), ssim_metric.get_buffer().squeeze().detach().cpu().numpy())
    np.savetxt(os.path.join(path_saving, 'dice_metric.txt'), dice_metric.get_buffer().squeeze().detach().cpu().numpy())
    np.savetxt(os.path.join(path_saving, 'hd95_metric.txt'), hd95_metric.get_buffer().squeeze().detach().cpu().numpy())
    np.savetxt(os.path.join(path_saving, 'hd50_metric.txt'), hd50_metric.get_buffer().squeeze().detach().cpu().numpy())
    np.savetxt(os.path.join(path_saving, 'negj_metric.txt'), negj_metric.get_buffer().squeeze().detach().cpu().numpy())
    np.savetxt(os.path.join(path_saving, 'rmseSI_metric.txt'), rmseSI_metric.get_buffer().squeeze().detach().cpu().numpy())
    np.savetxt(os.path.join(path_saving, 'rmseAP_metric.txt'), rmseAP_metric.get_buffer().squeeze().detach().cpu().numpy())
    
    if config.plot:
        print('Plotting some results...')
        plotting.plot_moving_fixed_and_outputs(moving=inputs, fixed=targets, 
                                                output=outputs, ddf=ddf, 
                                                sample_nr=-1, path_saving=path_saving)
        
        plotting.plot_moving_fixed_and_outputs_seg(moving=inputs_seg, fixed=targets_seg, fixed_img=targets,
                                                    output=outputs_seg, 
                                                    sample_nr=-1, path_saving=path_saving)

#%%
# SAVE RESULTS
path_saving_overall = os.path.dirname(path_saving)   # strip patient name folder from path
utils.results_dict_to_csv(results_dict, os.path.join(path_saving_overall, 'results.csv'))  # save results for each patient tgo csv
utils.add_summary_row(csv_file=os.path.join(path_saving_overall, 'results.csv'))  # calculcate mean over all patients and add it to csv