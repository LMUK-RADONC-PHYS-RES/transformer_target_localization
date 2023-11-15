"""
B-spline DIR using plastimatch wrapped in Python 3.
@authors: moritz.rabe, claudia.tejero, elia.lombardo (@med.uni-muenchen.de)
"""
#%%
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import nibabel as nb

import utils_models as utils

#%%
    
def run_xfconvert(input, output, xf):
    conv=utils.convert()
    conv.option['input']      = input
    conv.option['interpolation']      = 'nn'
    conv.option['output-img'] = output
    conv.option['xf']         = xf
    conv.run_convert()

    
def run_compose(input_file_1, input_file_2, outfile, log_file):
    
    compose=utils.compose()
    compose.input_files=[input_file_1, input_file_2]
    compose.log_file=log_file
    compose.outfile=outfile
    
    compose.run_compose()


def run_plastimatch_DIR(lambda_value, init_grid_spac, par_file, log_file, fixed, moving, output, 
                        xform_out=None, vf=None, metric='mse', max_its='500', impl='plastimatch', load_output=False):
    """Execute B-spline registration with Plastimatch using specified parameters lambda_value and init_grid_spac.

    Args:
        lambda_value (str): regularization value for iterative optimization
        init_grid_spac (str): b-spline grid spacing (in mm) of first stage, with the followign stages being smaller by certain factors
        par_file (str): path where file with paramters is saved
        log_file (str): path where registration log is saved
        fixed (str): path to fixed imaged, can be nifti or mha
        moving (str): path to moving image, can be nifti or mha
        output (str): path to warped moving image, can be nifti or mha
        vf (str, optional): path with txt where deformation vector field is saved. Defaults to None.
        xform_out (str, optional): path with txt where b-spline coefficients are saved. Defaults to None.
        metric (str, optional): optimization metric (mse, mi, gm)
        max_its (str, optional): maximum nr of iterations
        impl (str, optional): implementaion with plastimatch or itk engine
        load_output (bool, optional): if True, load the warped moving in python

    Returns:
        np.array: output array
    """
    
    # try to delete existing command and log files
    try:
        os.remove(par_file)
        os.remove(log_file)
        print('Deleted old parameter and log file.')
    except OSError:
        print('No files to delete!')
        pass
        
    # initialize registration class and set a params based on lambda and grid factor
    reg=utils.register() 
    reg.par_file = par_file
    reg.log_file = log_file
    
    reg.add_global_stage()
    reg.stages[0]['fixed']      = fixed
    reg.stages[0]['moving']     = moving
    reg.stages[0]['img_out']    = output
    if vf:
        reg.stages[0]['vf_out']  = vf  # Note: writing vf takes a lot of time. If no checks are necessary, this line can be commented
    if xform_out:
        reg.stages[0]['xform_out']  = xform_out

    print('Lambda value: ' + lambda_value)
    
    if init_grid_spac == '30':
        print('Initial grid spacing: ' + init_grid_spac)
        reg.add_stage()
        reg.stages[1]['xform']     = 'bspline'
        reg.stages[1]['impl']      = impl
        reg.stages[1]['threading'] = 'openmp'
        reg.stages[1]['regularization_lambda'] = lambda_value
        reg.stages[1]['metric']    = metric
        reg.stages[1]['max_its']   = max_its
        reg.stages[1]['grid_spac'] = '30 30 30'
        reg.stages[1]['res'] = '1 1 1'
        reg.add_stage()
        reg.stages[2]['grid_spac'] = '20 20 20'
        reg.add_stage()
        reg.stages[3]['grid_spac'] = '10 10 10'       
        reg.add_stage()
        reg.stages[4]['grid_spac'] = '8 8 8'
    
    elif init_grid_spac == '60':
        print('Initial grid spacing: ' + init_grid_spac)
        reg.add_stage()
        reg.stages[1]['xform']     = 'bspline'
        reg.stages[1]['impl']      = impl
        reg.stages[1]['threading'] = 'openmp'
        reg.stages[1]['regularization_lambda'] = lambda_value             
        reg.stages[1]['metric']    = metric
        reg.stages[1]['max_its']   = max_its
        reg.stages[1]['res'] = '1 1 1'
        reg.stages[1]['grid_spac'] = '60 60 60'
        reg.add_stage()
        reg.stages[2]['grid_spac'] = '40 40 40'
        reg.add_stage()
        reg.stages[3]['grid_spac'] = '20 20 20'      
        reg.add_stage()
        reg.stages[4]['grid_spac'] = '15 15 15'
        
    elif init_grid_spac == '90':
        print('Initial grid spacing: ' + init_grid_spac)
        reg.add_stage()
        reg.stages[1]['xform']     = 'bspline'
        reg.stages[1]['impl']      = impl
        reg.stages[1]['threading'] = 'openmp'
        reg.stages[1]['regularization_lambda'] = lambda_value
        reg.stages[1]['metric']    = metric
        reg.stages[1]['max_its']   = max_its
        reg.stages[1]['grid_spac'] = '90 90 90'
        reg.stages[1]['res'] = '1 1 1'
        reg.add_stage()
        reg.stages[2]['grid_spac'] = '60 60 60'
        reg.add_stage()
        reg.stages[3]['grid_spac'] = '30 30 30'      
        reg.add_stage()
        reg.stages[4]['grid_spac'] = '23 23 23'
    
    elif init_grid_spac == '120':
        print('Initial grid spacing: ' + init_grid_spac)
        reg.add_stage()
        reg.stages[1]['xform']     = 'bspline'
        reg.stages[1]['impl']      = impl
        reg.stages[1]['threading'] = 'openmp'
        reg.stages[1]['regularization_lambda'] = lambda_value
        reg.stages[1]['metric']    = metric
        reg.stages[1]['max_its']   = max_its
        reg.stages[1]['grid_spac'] = '120 120 120'
        reg.stages[1]['res'] = '1 1 1'
        reg.add_stage()
        reg.stages[2]['grid_spac'] = '80 80 80'
        reg.add_stage()
        reg.stages[3]['grid_spac'] = '40 40 40'      
        reg.add_stage()
        reg.stages[4]['grid_spac'] = '30 30 30'
    else:
        raise ValueError('Attention: unknow init_grid_spac specified, accepted strings are 30, 60, 90, 120 mm !')

    t0 = time.time()
    reg.run_registration()
    t1 = time.time()
    tot_t = (t1 - t0)
    print(f'Time needed for plastimatch registration: {tot_t} s')
    
    
    # get obtained output
    if load_output:
        if output.endswith('.gz'):
            output_array  = nb.load(output).get_fdata()
        else:
            raise Exception('Attention: only nii.gz files currently supported !')
    
        return output_array


def apply_existing_vf(log_file, vf, output, moving='', warp_grid=False, load_output=False):
    """Apply a dvf saved on disk to a specified moving image.

    Args:
        log_file (str): path to log file
        dvf (str): path to deformation vector field previously generated with plastimatch (can be coefficients or dense displacements)
        output (str): path where warped moving is saved
        moving (str, optional): path to moving image. Defaults to ''.
        warp_grid (bool, optional): if True, instead of the moving image, an image of a regular grid is warped. Defaults to False.
        load_output (bool, optional): if True, load the warped moving in python

    Returns:
        np.array: output and vf array
    """
    
    # initialize warping class
    warper=utils.warp() 
    warper.log_file = log_file
    if warp_grid:
        # create a temporary regular grid file and use it as moving
        grid_image = utils.get_grid_image(rows=270, cols=270)
        grid_image = grid_image[:,:,None]  # add fake 3rd dimension
        moving = os.path.join(os.path.dirname(output), 'grid_image.nii.gz') # strip filename of output and keep folder
        affine=np.eye(4) # define the affine matrix to correctly orient image in vv, no effect in python
        affine[0,0]=0
        affine[0,1]=-1
        affine[1,1]=0
        affine[1,0]=-1
        affine[2,2]=1
        affine[3,3]=1
        grid_image_nifti = nb.Nifti1Image(grid_image, affine=affine)
        nb.save(grid_image_nifti, moving)
    else:
        if moving == '':
            raise ValueError('Path to moving must be specified if warp_grid=False !')
        else:
            # in case path contains spaces, escape them for a linux terminal
            moving = escape_for_terminal(moving)
            
    warper.option['input'] = moving
    warper.option['output-img'] = output
    warper.option['xf'] = vf
    
    warper.run_warp()
    
    # get obtained output
    if load_output:
        if output.endswith('.gz'):
            output_array = nb.load(output).get_fdata()
            vf_array = nb.load(vf).get_fdata()
        else:
            raise Exception('Attention: only nii.gz files currently supported !')
    
        return output_array, vf_array
    

def escape_for_terminal(input_string):
    """Escape spaces in strings for linux terminals."""
    escaped_string = input_string.replace(' ', '\\ ')
    return escaped_string

#%%
if __name__ == '__main__':
    lambda_value = '1.0'
    init_grid_spac = '90'
    par_file = '/home/segmentation_at_MRIdian/data/training/converted/2023_08_28_test_plm/command_file.txt'
    log_file = '/home/segmentation_at_MRIdian/data/training/converted/2023_08_28_test_plm/log_file.txt'
    fixed = '/home/segmentation_at_MRIdian/data/training/converted/2023_08_28_test_plm/frame_eight hundred and eight.nii.gz'
    moving = '/home/segmentation_at_MRIdian/data/training/converted/2023_08_28_test_plm/frame_eight hundred and eighteen.nii.gz'
    output = '/home/segmentation_at_MRIdian/data/training/converted/2023_08_28_test_plm/output.nii.gz'
    xform_out = '/home/segmentation_at_MRIdian/data/training/converted/2023_08_28_test_plm/xf.txt'
    vf = '/home/segmentation_at_MRIdian/data/training/converted/2023_08_28_test_plm/vf.nii.gz'
    
    run_plastimatch_DIR(lambda_value, init_grid_spac, par_file, log_file, fixed, moving, output, xform_out, vf)
    
    apply_existing_vf(log_file, vf, output='/home/segmentation_at_MRIdian/data/training/converted/2023_08_28_test_plm/grid_output.nii.gz', 
                        moving='', warp_grid=True)