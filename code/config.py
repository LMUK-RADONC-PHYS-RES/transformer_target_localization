"""
Configuration file for main scripts
"""

#%%
import os
import matplotlib
import torch
import time
import sys

# set path to project folder 
# (to be adapted to your path inside Docker container where project folder is mounted)
path_project = '/home/target_localization'

# path to data folder
path_project_data = os.path.join(path_project, 'data')

# path to model buiding code folder
path_project_code = os.path.join(path_project, 'code')

# path to results folder
path_project_results = os.path.join(path_project, 'results')

# add project auxiliary and models folder to Python path to be able to import self written modules from anywhere
sys.path.append(os.path.join(path_project_code, 'auxiliary'))
sys.path.append(os.path.join(path_project_code, 'models'))
# import utils

#%%
# SET GENERAL SETTINGS 

print('-------------------------------------------------')
# GPU settings
gpu_iden = 0  # None=CPU, 0,1,...=GPU (explictly set GPU ID)
if gpu_iden is not None:
    if torch.cuda.is_available():  
        print('GPUs are available!')
        GPU_num = torch.cuda.device_count()
        for GPU_idx in range(GPU_num):
            GPU_name = torch.cuda.get_device_name(GPU_idx)
            print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
        
        device = torch.device(f'cuda:{gpu_iden}') 
        # set device nr to standard GPU
        torch.cuda.set_device(gpu_iden)  
        print('Currently using: ' + torch.cuda.get_device_name(gpu_iden)) 
    else:  
        print('No GPU available! Running on CPU...')
        device = torch.device('cpu') 
else:
    print('Running on CPU!')
    device = torch.device('cpu')
print('--------------------------------------------------')

# plot settings
matplotlib.rcParams.update({'font.size': 22})  # increase fontsize of all plots
plot = True # whether to plot results when running scripts
     
#%%
# SET PARAMETERS FOR SCRIPTS

# time at which run is started, used to save results
start_time_string = time.strftime("%Y-%m-%d-%H:%M:%S")  

# moving image (either or explicit frame name or None to pick the first one)
moving_id = None

# choose dataset
# dataset = '2023_05_10_test'
dataset = '2023_05_05_all_sites'

# other data params
wandb_usage = False
os.environ['WANDB_API_KEY'] = 'xxx'    # wandb API key for your user
train_val_split = 0.90     # percentage of unsupervised training samples, rest is validation
unsupervised_validation = False     # whether to use validation set without segmentations
supervised_validation = True     # whether to use validation set with segmentations
patient_specific_inference = True       # whether to use model trained on first frames of each patient in main_infer script
inference = 'testing'   #  validation, testing 

# choose patients for labeled fine-tuning, validation and testing set 
observer_training = 'contoured_ground_truth_LV'
patients_training = ['liver_patient0010', 'liver_patient0012', 'liver_patient0015', 'abdomen_patient0011', \
                    'prostate_patient0021', 'lung_patient0017', 'lung_patient0042', 'lung_patient0027', \
                    'lung_patient0019', 'lung_patient0091', 'pancreas_patient0001', 'pancreas_patient0007']
observer_validation = 'contoured_ground_truth_LV'
patients_validation = ['liver_patient0005', 'liver_patient0056', 'liver_patient0061', 'abdomen_patient0006', \
                    'prostate_patient0001', 'lung_patient0004', 'lung_patient0033', 'lung_patient0002', \
                    'pancreas_patient0022', 'pancreas_patient0005']
observer_testing = 'contoured_ground_truth_VG'  #  contoured_ground_truth_LV, contoured_ground_truth_VG
patients_testing = ['liver_patient0006', 'liver_patient0027', 'abdomen_patient0008', 'abdomen_patient0017', \
                    'prostate_patient0006', 'prostate_patient0013', 'lung_patient0041', 'lung_patient0046', \
                    'pancreas_patient0002', 'pancreas_patient0011', \
                    'heart_patient0001', 'heart_patient0002', 'mediastinum_patient0002']


# choose model for main script
model_name = 'TransMorph2D'  # TransMorph2D, BSpline, NoReg, InterObserver

if model_name == 'TransMorph2D':
    model_variant = 'TransMorph'    # see dict_model_variants in models.TranMorph
    load_state = False      # continue training with current best model/optimizer for epoch_nr epochs or load model for inferece
    if load_state: 
        # specify model 
        start_time_string = '2023-09-14-07:20:12'   # unsup --> this one has uploaded weights!
        # start_time_string = '2023-09-14-07:20:12_supervised_2023-10-02-15:15:24'   # unsup+sup
        # start_time_string = '2023-09-14-07:20:12_ps_2023-10-15-14:59:20'    # unsup+ps (val)
        # start_time_string = '2023-09-14-07:20:12_ps_2023-10-18-09:26:11'    # unsup+ps (test)
    batch_size = 2     # 2, 16, 64, 128, 192
    lr = 0.00001    # learning rate
    lr_scheduler = None   # None, WarmupCosine
    epoch_nr = 300    # nr of training epochs   30, 50, 100, 150, 300, 500
    loss_name = 'MSE-Diffusion'   # MSE, SSIM, LNCC, Dice, MSE-Diffusion, MSE-Bending, LNCC-Diffusion, LNCC-Bending, Dice-Diffusion, Dice-Bending, MSE-Dice-Diffusion, MSE-Dice-Bending
    loss_weights = [1, 0.01]   # weights for combined loss,  0th element is for image and 1st for displacmement part
    # loss_weights = [0.05, 1, 0.005]   # weights for combined loss,  0th element is for image and 1st for segmentation and 2nd for displacmement part
    prob_randaffine = 0.75
    prob_randelasticdef = 1.0  # must be 1.0
    prob_randmotion = 0.75
    prob_randbiasfield = 0.75
    prob_randgibbsnoise = 0.75
    prob_randgaussiansmooth = 0.75
    len_ps_train = 8         # nr of frames used for patient specific training
    
elif model_name == 'BSpline':
    device = torch.device('cpu')  # always run on CPU
    print('Running on CPU!')
    lambda_value = '5.0'  # regularization parameter
    init_grid_spac = '90'  # grid spacing in mm of first B-Spline stage
    metric='mse'   # mse, mi, gm
    max_its='500'
    impl = 'plastimatch'  # plastimatch, itk
            
elif model_name == 'NoReg':
    device = torch.device('cpu')  # always run on CPU
    print('Running on CPU!')

elif model_name == 'InterObserver':
    device = torch.device('cpu')  # always run on CPU
    print('Running on CPU!')    
    inference = 'testing' 
    
else:
    raise ValueError('Unknown model_name specified!')  

#%%
