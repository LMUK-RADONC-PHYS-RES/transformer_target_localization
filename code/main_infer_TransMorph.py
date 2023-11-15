#%%
import torch
import os
import monai
from monai.transforms import Compose, LoadImaged, ScaleIntensityRanged, EnsureChannelFirstd, CenterSpatialCropd
from monai.data import DataLoader, Dataset

import config
from models.TransMorph import dict_model_variants
import models.TransMorph as TransMorph
from auxiliary import utils, train_val_test, plotting

if config.model_name != 'TransMorph2D':
    raise ValueError('Attention: running TransMorph2D script but another model_name was specified in config_generic.py !')
if config.load_state is False:
    raise Exception('Attention: performing inference but load_state is False !')

if config.inference == 'validation':
    observer = config.observer_validation
    patients = config.patients_validation
elif config.inference == 'testing':
    observer = config.observer_testing
    patients = config.patients_testing
else:
    raise ValueError('Attention: unknown inference type!')
        
#%%
# GET MODEL
model_config = dict_model_variants[config.model_variant]
model = TransMorph.TransMorph(model_config)
model.to(config.device)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total number of trainable parameters: {pytorch_total_params}\n')

# load baseline model
if config.patient_specific_inference is False:
    if 'LNCC' in config.loss_name:
        neg_loss=True
    else:
        neg_loss=False
    path_loading = os.path.join(config.path_project_results, 'training', config.model_name, config.start_time_string)
    path_best_model = utils.get_path_to_best_model(path_loading, neg_loss=neg_loss)
    print(f'Path to best model: {path_best_model}')  
    model.load_state_dict(torch.load(path_best_model))
    
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

path_dataset = os.path.join(config.path_project_data, 'testing', 'images', 'final_with_contours', observer)
path_patient_specific = os.path.join(config.path_project_data, 'testing', 'images', 'contoured_AI_assisted', '2023_09_11_val_and_test')
results_dict = {}
# loop over each patient and perform evaluation
for patient in patients:
    # path_saving  = None
    path_saving = os.path.join(config.path_project_results, 'inference', config.inference, 'LMU_observer_' + observer[-2:], config.model_name, config.start_time_string, patient)
    os.makedirs(path_saving, exist_ok=True)
    
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
    infer_loader = DataLoader(infer_ds, batch_size=1, shuffle=False, num_workers=2)

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


    # load patient specific model
    if config.patient_specific_inference:
        if 'LNCC' in config.loss_name:
            neg_loss=True
        else:
            neg_loss=False
        path_loading = os.path.join(config.path_project_results, 'training', config.model_name, config.start_time_string, patient)
        path_best_model = utils.get_path_to_best_model(path_loading, neg_loss=neg_loss)
        print(f'Path to best model: {path_best_model}')  
        model.load_state_dict(torch.load(path_best_model))
    
    
    # perform evaluation and get metrics for this patient
    print('Running inference...')
    eval_metrics_patient = train_val_test.evaluate_model(model, device=config.device, data_loader=infer_loader,
                                                            path_saving=path_saving, plot=config.plot)
    
    # update dict with patient name and eval metrics
    results_dict[patient] = eval_metrics_patient

#%%
# SAVE RESULTS
path_saving_overall = os.path.dirname(path_saving)   # strip patient name folder from path
utils.results_dict_to_csv(results_dict, os.path.join(path_saving_overall, 'results.csv'))  # save results for each patient tgo csv
utils.add_summary_row(csv_file=os.path.join(path_saving_overall, 'results.csv'))  # calculcate mean over all patients and add it to csv
