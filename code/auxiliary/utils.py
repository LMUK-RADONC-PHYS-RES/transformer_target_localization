# %%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import SimpleITK as sitk  
import os, csv
import scipy
import pandas as pd
import matplotlib.pyplot as plt
from subprocess import run, PIPE
import random, time
from functools import wraps
import monai
from monai.transforms import Compose, LoadImaged, RandGibbsNoised, ScaleIntensityRanged, RandAffined, EnsureChannelFirstd, CenterSpatialCropd, RandGaussianSmoothd
from monai.data import DataLoader, Dataset, CacheDataset, PersistentDataset, CacheNTransDataset
import pystrum.pynd.ndutils as nd
# import word2number

#%%      
def subdir_paths(path):
    " Given a path the function returns only primary subdirectories in a sorted list. "
    return list(filter(os.path.isdir, [os.path.join(path, f) for f in sorted(os.listdir(path))]))

def read_mha(path):
    """
    ​Given a path the function reads the image and returns header as well as the intensity values into a 3d array.
    :param path: String value that represents the path to the .mha file
    :return: image executable and 3D array filled with intensity values
    """
    reader = sitk.ImageFileReader()
    reader.SetFileName(path)
    image = reader.Execute()
    
    # get array with values from sitk image
    image_array = sitk.GetArrayFromImage(image)
    
    return image, image_array
 
    
def write_mha(path, image_array, image=None):
    """
    Given a 3d array and header the function saves a .mha file at the requested path.
    :param image: Executable of image (contains header information, etc.). Highly recommended to provide it
    :param image_array: 3d array filled with intensity values corresponding to a .mha file
    :param path: String value that represents the path to the created .mha file
    :return: Image file corresponding to the input is saved to path
    """
    # get sitk image from array of values
    new_image = sitk.GetImageFromArray(image_array)
#    print('Size of image: ' + str(new_image.GetSize())) # (256, 256, 256)

    # write header info into new image
    if image is not None:
        new_image.SetOrigin(image.GetOrigin())
        new_image.SetSpacing(image.GetSpacing())
        
    # save image
    writer = sitk.ImageFileWriter()
    writer.SetFileName(path)
    writer.Execute(new_image)
    
    
def resample_mha(path, new_spacing, interpolator=sitk.sitkLinear, default_value=-1000):
    """
    Given a path to an sitk image (e.g. .mha file) it resamples the image to a given spacing
    :param path: path to image
    :param new_spacing: spacing to which image is resampled, e.g. [1,1,1] for isotropic resampling
    :param interpolator: which interpolator to use, e.g. sitk.sitkLinear, sitk.sitkNearestNeighbor, sitk.sitkBSpline
    :param default_value: pixel value when a transformed pixel is outside of the image. The default default pixel value is 0.
    :return: resampled image executable
    """
    # load sitk image and create instance of ResampleImageFilter class 
    image = sitk.ReadImage(path)
    resample = sitk.ResampleImageFilter()
    
    # set parameters for resample instance
    resample.SetInterpolator(interpolator)
    resample.SetDefaultPixelValue(default_value)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
#    print('Original origin: ' + str(image.GetOrigin()))
#    print('New origin: ' + str(resample.GetOutputOrigin()))
    resample.SetOutputSpacing(new_spacing)
    
    # compute new size and set parameters for resample instance
    orig_size = np.array(image.GetSize(), dtype=np.int)
    orig_spacing = np.array(image.GetSpacing(), dtype=np.float)
    new_spacing = np.array(new_spacing, dtype=np.float)
    new_size = orig_size*(orig_spacing/new_spacing)
    new_size = np.ceil(new_size).astype(np.int) #  image dimensions are in integers
    new_size = [int(s) for s in new_size]
    resample.SetSize(new_size) 
#    print('Original spacing: ' + str(orig_spacing)) # [1.14453125 1.14453125 3.  ]
#    print('New spacing: ' + str(new_spacing)) # [1. 1. 1.]
#    print('Original size: ' + str(orig_size)) # [512 512 135]
#    print('New size: ' + str(new_size)) # [586, 586, 405]

   # perform the resampling with the params set above
    new_image = resample.Execute(image)
    
    return new_image

def resample_image(in_array, out_spacing=(1.0, 1.0, 1.0), 
                   interpolator=sitk.sitkNearestNeighbor, default_value=0):
    """Resample numpy array to specified spacing using SITK.

    Args:
        in_array (numpy array): Image in mupy array format with dimensions (d,h,w).
        out_spacing (tuple, optional): Spacing of resampled image with dimensions (d,h,w) 
            compared to spacing of original image which is automatically set to (1.0, 1.0, 1.0). 
            For example  when setting out_spacing=(1.0, 1.0, 0.5) the image will  
            be upsampled by a factor of 2 in height.
        interpolator (SITK interpolator, optional): For example sitk.sitkLinear, sitk.sitkNearestNeighbor.
        default_value (int, optional): Pixel value used when a transformed pixel is outside of the image.

    Returns:
        out_array: Resampled image as numpy array with dimensions (d,h,w).
    """

    # get sitk image from numpy array
    in_image = sitk.GetImageFromArray(in_array)

    # get input and ouput spacing and size
    in_spacing = in_image.GetSpacing()  # (1.0, 1.0, 1.0)
    out_spacing = out_spacing[::-1] # change to sitk indexing (w,h,d)
    in_size = in_image.GetSize()  # (w,h,d) as SITK indexing is inverted compared to numpy
    out_size = (int(in_size[0] * (in_spacing[0] / out_spacing[0])),
                int(in_size[1] * (in_spacing[1] / out_spacing[1])),
                int(in_size[2] * (in_spacing[2] / out_spacing[2])))

    # set resampling parameters
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(out_size)
    resampler.SetOutputSpacing(out_spacing)
    resampler.SetOutputDirection(in_image.GetDirection())
    resampler.SetOutputOrigin(in_image.GetOrigin())
    resampler.SetDefaultPixelValue(default_value)
    resampler.SetInterpolator(interpolator)
    
    # perform interpolation with parameters set above
    out_image = resampler.Execute(in_image) 

    out_array = sitk.GetArrayFromImage(out_image)
    # print(out_array.shape) # (d,h,w) as GetArrayFromImage restores numpy indexing

    return out_array     
    
    
def runcmd(cmd, path_cwd, fn_out=None):
    """
    Runs a command as can be run in shell and prints the output if needed.
    Parameters:
        cmd: str, command as would be written in shell.
        path_cwd: str, current working directory
        fn_out: str, where to write output file
    """
#    print ('\n##########################################################')
#    print ('### SUBPROCESS START\n')
 
    # output of command line
    result = run(cmd, cwd=path_cwd, stdout=PIPE, stderr=PIPE, shell=True)
  
    if result.returncode == 0:
        #print(result.stdout)
        pass        
    else:
        if result.stderr:
            print('Error in runcmd.py occured: ' + str(result.stderr)) # does not work due do byte to string conversion?

    # write output to file if desired
    if fn_out is not None:
        print ('Writing output to file ' + fn_out + '\n')
        with open (fn_out, 'w') as file_out:
            file_out.write(str(result.stdout))
            
#    print ('### SUBPROCESS END')
#    print ('##########################################################\n')


def normalize(values, bounds, single_value=False, to_tensor=False):
    """ Normalize values in range define by bounds.

    Args:
        values (list or array or tensor): data to be normalized, shape=(nr_data_points)
        bounds (dict): current and desired bounds, for example
        {'actual':{'lower':5,'upper':15},'desired':{'lower':-1,'upper':1}}
        single_value: to give a single value as input (and output), i.e. nr_data_points=1
        to_tensor: convert to tensor

    Returns:
        array: array with normalized values
    """

    # convert tensor to numpy arrays on cpu 
    if torch.is_tensor(values):
        values = values.detach().cpu().numpy()
    else:
        pass  
    
    if single_value:
        if to_tensor:
            return torch.tensor(bounds['desired']['lower'] + (values - bounds['actual']['lower']) * \
                    (bounds['desired']['upper'] - bounds['desired']['lower']) / \
                    (bounds['actual']['upper'] - bounds['actual']['lower']))             
        else:
            return bounds['desired']['lower'] + (values - bounds['actual']['lower']) * \
                    (bounds['desired']['upper'] - bounds['desired']['lower']) / \
                    (bounds['actual']['upper'] - bounds['actual']['lower'])      
    else:  
        if to_tensor: 
            return torch.tensor(np.array([bounds['desired']['lower'] + (x - bounds['actual']['lower']) *
                    (bounds['desired']['upper'] - bounds['desired']['lower']) / 
                    (bounds['actual']['upper'] - bounds['actual']['lower']) for x in values]))
        else:
            return np.array([bounds['desired']['lower'] + (x - bounds['actual']['lower']) *
                    (bounds['desired']['upper'] - bounds['desired']['lower']) / 
                    (bounds['actual']['upper'] - bounds['actual']['lower']) for x in values])



def get_path_to_best_model(path_model_files, neg_loss=False):
    # load trained model
    model_files = [] 
    losses = [] 
    # loop over all subfolders and files of one pre training
    for _, _, file_list in os.walk(path_model_files):
        for file in file_list:
            if file[-4:] == '.pth':
                model_files.append(file)
                # append all the loss values (attention: nr of digits is hard-coded)
                # losses.append(float(file[-12:-4]))
                losses.append(float(file[30:38]))
                    
    losses = np.array(losses)
    if neg_loss:
        losses = -losses
    model_files = np.array(model_files)

    # find best model by looking at the smallest loss 
    best_model = model_files[np.argmin(losses)]
    path_best_model = os.path.join(path_model_files, best_model) 
    
    return path_best_model
    
def get_path_to_best_optimizer(path_model_files, neg_loss=False):
    # load trained model
    optimizer_files = [] 
    losses = [] 
    # loop over all subfolders and files of one pre training
    for _, _, file_list in os.walk(path_model_files):
        for file in file_list:
            if file[-3:] == '.pt':
                optimizer_files.append(file)
                # append all the loss values (attention: nr of digits is hard-coded)
                losses.append(float(file[-11:-3]))
                    
    losses = np.array(losses)
    if neg_loss:
        losses = -losses
    optimizer_files = np.array(optimizer_files)

    # find best by looking at the smallest loss 
    best_optimizer = optimizer_files[np.argmin(losses)]
    path_best_optimizer = os.path.join(path_model_files, best_optimizer)   
    
    return path_best_optimizer


def get_paths_dict(path_dataset, moving_id='frame_six.nii.gz', seg=False, excluded_frames=[]):
    """Get paths to all pairs of fixed and moving images in a specified directory. Needs to be adapted to own folder structure!

    Args:
        path_dataset (str): path to dataset, note that folder strucutre is hardcoded
        moving_id (str, optional): file name of moving image. Defaults to 'frame_six.nii.gz'. If None, the first frame is selected.
        seg (bool, optional): whether to include paths to segmentations
        excluded_frames (list, optional): list with frames to exclude (used for patient specific training)

    Returns:
        list: list with dictionaries with paths to fixed and moving images (and seg)
    """
    paths_fixed_image = []    # current frame
    paths_moving_image = []   # key frame
    paths_fixed_seg = []    # current segmentation
    paths_moving_seg = []   # key segmentation
    path_data = subdir_paths(path_dataset)
    
    if seg is False:
        # getting all paths to fixed and moving images
        for path_case in path_data:
            # print(f'Getting paths for: {path_case}')
            # getting paths to different cine MRI (sessions)
            for path_cine in subdir_paths(os.path.join(path_case, 'raw_cine')):
                for _, _, file_list in os.walk(path_cine):
                    if moving_id is None:
                        # convert list from words back to number
                        file_list_numbers = get_frame_nr(file_list)
                        # sort list according to file names with numbers
                        file_list_sorted = [x for _,x in sorted(zip(file_list_numbers, file_list))]
                        # get first frame --> moving
                        moving_id_name = file_list_sorted[0]
                    else:
                        moving_id_name = moving_id
                    
                    nr_frames = 0  # nr of fixed images
                    for file_name in file_list:
                        if file_name != moving_id_name:
                            if file_name not in excluded_frames:
                                nr_frames += 1
                                # get path to fixed images according to moving_id
                                paths_fixed_image.append(os.path.join(path_cine, file_name))
                    # repeat path to moving image for every fixed image
                    for file_nr in range(nr_frames):
                        paths_moving_image.append(os.path.join(path_cine, moving_id_name))
    
        paths_dict = [
        {
            "fixed_image": paths_fixed_image[idx],
            "moving_image": paths_moving_image[idx],
        }
        for idx in range(len(paths_fixed_image))
        ]
        
    else:
        # getting all paths to fixed and moving images
        labels_folder = 'labels'
        
        # print(f'Getting paths for: {path_case}')
        # getting paths to different cine MRI (sessions)
        for path_case in path_data:
            print(f'Getting paths for: {path_case}')
            for path_cine in subdir_paths(path_case):
                current_dir_file_list = [file for file in os.listdir(path_cine) if os.path.isfile(os.path.join(path_cine, file))]
                if moving_id is None:
                    # convert list from words back to number
                    file_list_numbers = get_frame_nr(current_dir_file_list)
                    # sort list according to file names with numbers
                    file_list_sorted = [x for _,x in sorted(zip(file_list_numbers, current_dir_file_list))]
                    # get first frame --> moving
                    moving_id_name = file_list_sorted[0]
                else:
                    moving_id_name = moving_id
                    
                if os.path.basename(path_dataset) == 'lung_patient0041':
                    # as the first frame for this patient is empty, manually select the next which isn't
                    moving_id_name = 'frame_nine.nii.gz'
                
                nr_frames = 0   # nr of fixed images
                for file_name in current_dir_file_list:
                    if file_name != moving_id_name:
                        if file_name not in excluded_frames:
                            nr_frames += 1
                            # check if file to corresponding segmentation exists
                            if os.path.exists(os.path.join(path_cine, labels_folder, file_name)):
                                # get path to fixed images according to moving_id
                                paths_fixed_image.append(os.path.join(path_cine, file_name))
                                paths_fixed_seg.append(os.path.join(path_cine, labels_folder, file_name))
                    
                    for file_nr in range(nr_frames):
                        # check if file to corresponding segmentation exists
                        if os.path.exists(os.path.join(path_cine, labels_folder, moving_id_name)):                                
                            paths_moving_image.append(os.path.join(path_cine, moving_id_name))                            
                            paths_moving_seg.append(os.path.join(path_cine, labels_folder, moving_id_name))    
                                                
        paths_dict = [
        {
            "fixed_image": paths_fixed_image[idx],
            "moving_image": paths_moving_image[idx],
            "fixed_seg": paths_fixed_seg[idx],
            "moving_seg": paths_moving_seg[idx],
        }
        for idx in range(len(paths_fixed_image))
        ]       
    
    return paths_dict

def separate_ps_train_and_infer(all_files, excluded_frames, len_train=12):
    train_list = []
    infer_list = []
    
    for file_dict in all_files:
        # Check if any frame in the dictionary matches excluded_frames
        if os.path.basename(file_dict['fixed_image']) in excluded_frames:
            train_list.append(file_dict)
        else:
            infer_list.append(file_dict)

    # Check the length of train_list and move elements to infer_list if needed
    if len(train_list) != len_train:
        num_to_move = len(train_list) - len_train
        if num_to_move > 0:
            # Delete the last 'num_to_move' elements from train_list
            # infer_list.extend(train_list[-num_to_move:])
            del train_list[-num_to_move:]
        else:
            raise ValueError('len_train parameter is unexpectedly larger than len(train_list)... was it chosen <= 12 ?')
    
    return train_list, infer_list



def split_train_val(data, split_ratio, split_randomly=False, seed=None):
    if split_randomly:
        if seed is not None:
            random.seed(seed)
        random.shuffle(data)
    split_index = int(len(data) * split_ratio)
    train_data = data[:split_index]
    val_data = data[split_index:]
    return train_data, val_data


def results_dict_to_csv(results_dict, path_saving):
    # Extracting unique column names (A and B)
    columns = set()
    for patient_data in results_dict.values():
        columns.update(patient_data.keys())

    # Writing patient data to CSV file
    with open(os.path.join(path_saving), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Writing header row with column names
        header = ['Patient ID'] + sorted(columns)
        writer.writerow(header)
        
        # Writing data rows for each patient
        for patient, patient_data in results_dict.items():
            row = [patient]
            for column in sorted(columns):
                value = patient_data.get(column, '')
                row.append(value)
            writer.writerow(row)
            

def add_summary_row(csv_file, columns_to_summarize=['Dice', 'HD50', 'HD95', 'NegJ', 'RMSEAP', 'RMSESI']):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Calculate the mean and standard deviation for each specified column
    summary_data_param = {}
    summary_data_nonparam = {}
    for column in columns_to_summarize:
        if '-' in df[column + ' (mean)'].values:
            mean_value = std_value = '-'
            median_value = iqr_value = '-'
        else:
            mean_value = round(df[column + ' (mean)'].mean(), 2)
            std_value = round(df[column + ' (mean)'].std(), 2)
            median_value = round(df[column + ' (mean)'].median(), 2)
            Q1 = df[column + ' (mean)'].quantile(0.25)
            Q3 = df[column + ' (mean)'].quantile(0.75)
            iqr_value = round(Q3 - Q1, 2)

        summary_data_param[column + ' (mean)'] = mean_value
        summary_data_param[column + ' (std)'] = std_value
        summary_data_nonparam[column + ' (mean)'] = median_value
        summary_data_nonparam[column + ' (std)'] = iqr_value
        
    # Create a new row with 'all' as the 'Patient ID' and calculated values
    summary_data_param['Patient ID'] = 'all (parametric)'
    new_row_param = pd.Series(summary_data_param)
    summary_data_nonparam['Patient ID'] = 'all (non-parametric)'
    new_row_nonparam = pd.Series(summary_data_nonparam)
    
    # Append the new row to the DataFrame
    df = pd.concat([df, new_row_param.to_frame().T], ignore_index=True)
    df = pd.concat([df, new_row_nonparam.to_frame().T], ignore_index=True)

    # Save the updated DataFrame back to a CSV file
    df.to_csv(csv_file, index=False)
    

def jacobian_determinant(disp):
    """
    Compute jacobian determinant of a displacement field.
    NB: to compute the spatial gradients, we use np.gradient.
    Parameters:
        disp: 2D or 3D displacement field of size [*vol_shape, nb_dims],
              where vol_shape is of len nb_dims
    Returns:
        jacobian determinant (scalar)
    """

    # convert tensor to numpy array on cpu if needed
    if torch.is_tensor(disp):
        disp = disp.detach().cpu().numpy()
    elif isinstance(disp, np.ndarray):
        pass
    else:
        raise Exception('Attention: unknown type!')   
    
    # check inputs
    disp = disp.transpose(1, 2, 0)
    volshape = disp.shape[:-1]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'Displacement field has to be 2D or 3D'

    # compute grid
    grid_lst = nd.volsize2ndgrid(volshape)
    grid = np.stack(grid_lst, len(volshape))

    # compute gradients
    J = np.gradient(disp + grid)

    # 3D displacement field
    if nb_dims == 3:
        dx = J[0]
        dy = J[1]
        dz = J[2]

        # compute jacobian components
        Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
        Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
        Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

        return Jdet0 - Jdet1 + Jdet2

    # must be 2D displacement field
    else:  
        dfdx = J[0]
        dfdy = J[1]

        return dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]


def text2int(textnum, numwords={}):
    if not numwords:
      units = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
        "sixteen", "seventeen", "eighteen", "nineteen",
      ]

      tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

      scales = ["hundred", "thousand", "million", "billion", "trillion"]

      numwords["and"] = (1, 0)
      for idx, word in enumerate(units):    numwords[word] = (1, idx)
      for idx, word in enumerate(tens):     numwords[word] = (1, idx * 10)
      for idx, word in enumerate(scales):   numwords[word] = (10 ** (idx * 3 or 2), 0)

    current = result = 0
    for word in textnum.replace(',', '').split():
        if word not in numwords:
          raise Exception("Illegal word: " + word)

        scale, increment = numwords[word]
        current = current * scale + increment
        if scale > 100:
            result += current
            current = 0

    return result + current

def warp_image(img, grid):
    """Warp image with displacement field.

    Args:
        img (tensor): input image with size (b,c,h,w, (d))
        grid (tensor): input flow field with size (b, h, w, (d), 2)

    Returns:
        tensor: warped input image
    """
    
    # convert tensor to numpy array on cpu if needed
    if isinstance(img, np.ndarray):
        img = torch.tensor(img)
    elif torch.is_tensor(img):
        pass
    else:
        raise Exception('Attention: unknown type!') 
     
    if isinstance(grid, np.ndarray):
        grid = torch.tensor(grid)
    elif torch.is_tensor(grid):
        pass
    else:
        raise Exception('Attention: unknown type!')   
       
    # use grid_sample Pytorch function to warp image
    wrp = nnf.grid_sample(img, grid, mode='bilinear')
    
    return wrp

class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    Obtained from https://github.com/voxelmorph/voxelmorph
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)

def get_frame_nr(list_with_words):
    
    list_with_nums = []
    for el in list_with_words:
        el_no_extension_and_frame = os.path.splitext(os.path.splitext(el)[0])[0][6:].replace("-", " " )
        el_no_extension_and_frame_as_nr = text2int(el_no_extension_and_frame)
        list_with_nums.append(el_no_extension_and_frame_as_nr)
    
    return list_with_nums

def timelog(func):
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time() -start
        print("Function {} took {} seconds".format(func.__name__, end))
        return result
    return wrapper

def get_segmentation_com(outputs, targets, to_tensor=True):
    """Get center of mass for output segmentations and predicted segmentations.

    Args:
        outputs (array or tensor): predicted segmentations with shape (b, c, ...)
        targets (array or tensor): ground truth segmentations with shape (b, c, ...)

    Returns:
        list/tensor of tuple with floats: centroids positions in SI and AP.
    """
    
    # convert tensor to numpy arrays on cpu 
    if torch.is_tensor(outputs):
        outputs = outputs.detach().cpu().numpy()
    if torch.is_tensor(targets):
        targets = targets.detach().cpu().numpy()
    else:
        pass  
    
    com_outputs = []
    com_targets = []
    for el in range(outputs.shape[0]):
        com_outputs.append(scipy.ndimage.measurements.center_of_mass(outputs[el,0,...]))
        com_targets.append(scipy.ndimage.measurements.center_of_mass(targets[el,0,...]))
    
    if to_tensor:
        com_outputs, com_targets = torch.tensor(com_outputs, dtype=torch.float32), torch.tensor(com_targets, dtype=torch.float32)
    
    return com_outputs, com_targets

def breathhold_detection_vectorized(array1, array2=None, wdw_size=20, amp_threshold1=0.05, amp_threshold2=0.005):
    """ Given a sequence of data, subdivide it in windows and then slide over them to find breath-holds.
    Args:
        array1: input sequence (e.g. inf-sup)
        array2: second input sequence (e.g. post-ant motion)
        wdw_size: size of sliding window 
        amp_threshold: normalized amplitude threshold below which to consider the corresponding window as a breath-hold
    """
    
    # get normalized data to be used to find indices of breathholds -->
    # needed as threshold works best on all data if the amplitudes are comparable (i.e. normalized)
    array1_norm = normalize(array1, {'actual': {'lower': np.min(array1), 'upper': np.max(array1)}, 
                                        'desired': {'lower': -1, 'upper': 1}})
    if array2 is not None:
        array2_norm = normalize(array2, {'actual': {'lower': np.min(array2), 'upper': np.max(array2)}, 
                                            'desired': {'lower': -1, 'upper': 1}}) 
         
    start = 0
    stop = len(array1)
    if array2 is not None:
        if len(array1) != len(array2):
            raise Exception('Attention! Length of array1 and array2 different.')
    step_size = 1 

    # find indices of all possible windows using vectorized operations
    idx_windows = (start + 
        np.expand_dims(np.arange(wdw_size), 0) +
        # Create a rightmost vector as [0, step, 2*step, ...].
        np.expand_dims(np.arange(stop - wdw_size + 1, step=step_size), 0).T)

    #print(array[idx_windows]) # e.g. [[0.8,0.9,0.92,0.9],[0.8,0.74,0.42,0.44]]

    breathholds=np.zeros(len(array1))
    # loop over all windows
    for window in idx_windows:
        # compute distances from median for normalized curve
        d1 = np.abs(array1_norm[window] - np.median(array1_norm[window]))
        #print(d)

        # compute median of distances from median
        mdev1 = np.median(d1)
        #print(mdev)

        # consider sequence breathhold if median of the distances is below a normalized amplitude of amp_threshold
        if mdev1 < amp_threshold1:
            breathholds[window] = 1

        # do the same for the second dimension and consider breathhold if there is constancy in either one of the two dims
        if array2 is not None:
            # compute distances from median for normalized curve
            d2 = np.abs(array2_norm[window] - np.median(array2_norm[window]))

            # compute median of distances from median
            mdev2 = np.median(d2)

            # consider sequence breathhold if median of the distances is below a normalized amplitude of amp_threshold
            if mdev2 < amp_threshold2:
                breathholds[window] = 1           

    return breathholds
# %%

# if __name__ == "__main__":    
#     supervised = True
    
#     # path_dataset='/home/segmentation_at_MRIdian/data/training/converted/2023_05_10_test'
#     path_dataset = '/home/segmentation_at_MRIdian/data/testing/images/contoured_ground_truth/2023_05_05_all_sites/heart_patient0002'
    
#     # get paths to moving and fixed image
#     paths_dict = get_paths_dict(path_dataset=path_dataset,
#                    moving_id='frame_six.nii.gz', seg=True)
    
#     train_files, val_files = paths_dict[:1], paths_dict[1:]
    
#     if supervised is False:
#         train_transforms = Compose(
#             [
#                 LoadImaged(keys=["fixed_image", "moving_image"]),
#                 EnsureChannelFirstd(keys=("fixed_image", "moving_image"), channel_dim=-1),  # images have shape (270,270,1)
#                 ScaleIntensityRanged(
#                     keys=["fixed_image", "moving_image"],
#                     a_min=0,
#                     a_max=1000,
#                     b_min=0.0,
#                     b_max=1.0,
#                     clip=False,
#                 ),
#                 RandAffined(
#                     keys=["fixed_image", "moving_image"],
#                     mode=("bilinear", "bilinear"),
#                     prob=0.5,
#                     spatial_size=(224, 224),
#                     rotate_range=(0, np.pi / 15),
#                     shear_range=(0.2, 0.2), 
#                     translate_range=(20,20),
#                     scale_range=(0.2, 0.2),
#                 ),
#                 RandGibbsNoised(
#                     keys=["fixed_image", "moving_image"],
#                     prob=0.5, alpha=(0.1, 0.9)
#                     ),
#                 RandGaussianSmoothd(
#                     keys=["fixed_image", "moving_image"],
#                     prob=0.5, approx='erf',
#                     sigma_x=(0.25, 1.5), sigma_y=(0.25, 1.5), 
#                     ),
#                 CenterSpatialCropd(
#                     keys=["fixed_image", "moving_image"],
#                     roi_size=[224,224]
#                 )
#             ]
#         )    

#         val_transforms = Compose(
#             [
#                 LoadImaged(keys=["fixed_image", "moving_image"]),
#                 EnsureChannelFirstd(keys=("fixed_image", "moving_image"), channel_dim=-1),  # images have shape (270,270,1)
#                 ScaleIntensityRanged(
#                     keys=["fixed_image", "moving_image"],
#                     a_min=0,
#                     a_max=1000,
#                     b_min=0.0,
#                     b_max=1.0,
#                     clip=False,
#                 ),
#                 CenterSpatialCropd(
#                     keys=["fixed_image", "moving_image"],
#                     roi_size=[224,224]
#                 )
#             ]
#         )
#     else:
#         train_transforms = Compose(
#             [
#                 LoadImaged(keys=["fixed_image", "moving_image", "fixed_seg", "moving_seg"]),
#                 EnsureChannelFirstd(keys=("fixed_image", "moving_image", "fixed_seg", "moving_seg"), channel_dim=-1),  # images have shape (270,270,1)
#                 ScaleIntensityRanged(
#                     keys=["fixed_image", "moving_image"],
#                     a_min=0,
#                     a_max=1000,
#                     b_min=0.0,
#                     b_max=1.0,
#                     clip=False,
#                 ),
#                 RandAffined(
#                     keys=["fixed_image", "moving_image", "fixed_seg", "moving_seg"],
#                     mode=("bilinear", "bilinear", "nearest", "nearest"),
#                     prob=0.0,
#                     spatial_size=(224, 224),
#                     rotate_range=(0, np.pi / 15),
#                     shear_range=(0.2, 0.2), 
#                     translate_range=(20,20),
#                     scale_range=(0.2, 0.2),
#                 ),
#                 RandGibbsNoised(
#                     keys=["fixed_image", "moving_image"],
#                     prob=0.0, alpha=(0.1, 0.9)
#                     ),
#                 RandGaussianSmoothd(
#                     keys=["fixed_image", "moving_image"],
#                     prob=0.0, approx='erf',
#                     sigma_x=(0.25, 1.5), sigma_y=(0.25, 1.5), 
#                     ),
#                 CenterSpatialCropd(
#                     keys=["fixed_image", "moving_image", "fixed_seg", "moving_seg"],
#                     roi_size=[224,224]
#                 )
#             ]
#         )    

#         val_transforms = Compose(
#             [
#                 LoadImaged(keys=["fixed_image", "moving_image", "fixed_seg", "moving_seg"]),
#                 EnsureChannelFirstd(keys=("fixed_image", "moving_image", "fixed_seg", "moving_seg"), channel_dim=-1),  # images have shape (270,270,1)
#                 ScaleIntensityRanged(
#                     keys=["fixed_image", "moving_image"],
#                     a_min=0,
#                     a_max=1000,
#                     b_min=0.0,
#                     b_max=1.0,
#                     clip=False,
#                 ),
#                 CenterSpatialCropd(
#                     keys=["fixed_image", "moving_image", "fixed_seg", "moving_seg"],
#                     roi_size=[224,224]
#                 )
#             ]
#         )
#     # Dataset (vanilla) 
#     train_ds = Dataset(data=train_files, transform=train_transforms)
#     val_ds = Dataset(data=val_files, transform=val_transforms)
    
#     # CacheDataset precompute all non-random transforms of original data and store in memory, might run out if dataset large
#     # train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=4)
#     # use batch_size=2 to load images and use RandCropByPosNegLabeld to generate 2 x 4 images for network training
#     # val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=0)
    
#     # PersisteneDataset uses persistent storage of non-random transformed training and validation data computed once and stored in persistently across runs
#     path_persistent_cache = "/home/segmentation_at_MRIdian/data/training/persistent_cache/"
#     # train_ds = PersistentDataset(data=train_files, transform=train_transforms, cache_dir=path_persistent_cache)
#     # val_ds = PersistentDataset(data=val_files, transform=val_transforms, cache_dir=path_persistent_cache)
#     # CacheNTrans is an extendion opf PersistenDataset which caches the result of first N transforms, no matter it's random or not.
#     # train_ds = CacheNTransDataset(data=train_files, transform=train_transforms, cache_dir=path_persistent_cache, cache_n_trans=4)
#     # val_ds = CacheNTransDataset(data=val_files, transform=val_transforms, cache_dir=path_persistent_cache, cache_n_trans=4)
    
    
#     # get data into batches using Dataloaders
#     train_loader = DataLoader(train_ds, batch_size=1, shuffle=False, num_workers=1)
#     val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=1)
    
#     sample_nr = 0
#     check_data = monai.utils.first(train_loader)
#     fixed_image = check_data["fixed_image"][sample_nr][0]  # (2,1,h,w)
#     moving_image = check_data["moving_image"][sample_nr][0]  

#     print(f"moving_image shape: {moving_image.shape}")  # (h,w)
#     print(f"fixed_image shape: {fixed_image.shape}")
#     plt.figure("check", (12, 6))
#     plt.subplot(1, 2, 1)
#     plt.title("moving_image")
#     plt.imshow(moving_image, cmap="gray")
#     plt.subplot(1, 2, 2)
#     plt.title("fixed_image")
#     plt.imshow(fixed_image, cmap="gray")
#     plt.show()

#     if supervised:
#         fixed_seg = check_data["fixed_seg"][sample_nr][0]  # (2,1,h,w)
#         moving_seg = check_data["moving_seg"][sample_nr][0]     
        
#         print(f"moving_seg shape: {moving_seg.shape}")  # (h,w)
#         print(f"fixed_seg shape: {fixed_seg.shape}")
#         plt.figure("check", (12, 6))
#         plt.subplot(1, 2, 1)
#         plt.title("moving_seg")
#         plt.imshow(moving_seg, cmap="gray")
#         plt.subplot(1, 2, 2)
#         plt.title("fixed_seg")
#         plt.imshow(fixed_seg, cmap="gray")
#         plt.show()
        
