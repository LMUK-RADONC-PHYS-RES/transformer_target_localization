# %%

import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import scipy.ndimage
from scipy.interpolate import griddata
import nibabel as nib

import utils

# %%

def plot_example_augmentations(moving_image, fixed_image, path_saving):
    plt.figure("check", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Moving - augmented")
    plt.imshow(moving_image, cmap="gray")
    plt.subplot(1, 2, 2)
    plt.title("Fixed - augmented")
    plt.imshow(fixed_image, cmap="gray")
    plt.show()
    if path_saving is not None:
        plt.savefig(path_saving)
    plt.close()
    
def get_grid_image(rows=224, cols=224, spacing=12):
    # Create a grid pattern using numpy
    grid = np.zeros((rows, cols), dtype=np.float32)
    # Add grid lines to the array
    for i in range(rows):
        for j in range(cols):
            if (i%spacing == 0) and (j%spacing == 0):
                grid[i, :] = 1  # Horizontal lines
                grid[:, j] = 1  # Vertical lines
                
    return grid
    

def plot_moving_fixed_and_outputs(moving, fixed, output, ddf=None, coarse_ddf=10, grid_spacing=12, 
                                  sample_nr=-1, path_saving=None):
    
    # convert tensor to numpy array on cpu if needed, assuming all same as moving
    if torch.is_tensor(moving):
        moving = moving.detach().cpu().numpy()
        fixed = fixed.detach().cpu().numpy()
        output = output.detach().cpu().numpy()
        if ddf is not None:
            ddf = ddf.detach().cpu().numpy()
    elif isinstance(moving, np.ndarray):
        pass
    else:
        raise Exception('Attention: unknown type!')   
    
    plt.figure(figsize=(14,14))
    plt.title('Moving')
    plt.imshow(moving[sample_nr,0,...], cmap='gray')
    if path_saving is not None:
        plt.savefig(os.path.join(path_saving, 'moving.png'))
    plt.close()
    
    plt.figure(figsize=(14,14))
    plt.title('Fixed')
    plt.imshow(fixed[sample_nr,0,...], cmap='gray')
    if path_saving is not None:
        plt.savefig(os.path.join(path_saving, 'fixed.png'))
    plt.close()
    
    plt.figure(figsize=(14,14))
    plt.title('Output (warped moving)')
    plt.imshow(output[sample_nr,0,...], cmap='gray')
    if path_saving is not None:
        plt.savefig(os.path.join(path_saving, 'output.png'))
    plt.close()
     
    plt.figure(figsize=(14,14))
    plt.title('Fixed - Output')
    plt.imshow( fixed[sample_nr,0,...] - output[sample_nr,0,...], cmap='bwr')
    plt.colorbar()
    plt.clim(-1.1,1.1)
    if path_saving is not None:
        plt.savefig(os.path.join(path_saving, 'diff_fixed_output.png'))   
    plt.close()

    plt.figure(figsize=(14,14))
    plt.title('Fixed - Moving')
    plt.imshow( fixed[sample_nr,0,...] - moving[sample_nr,0,...], cmap='bwr')
    plt.colorbar()
    plt.clim(-1.1,1.1)
    if path_saving is not None:
        plt.savefig(os.path.join(path_saving, 'diff_fixed_moving.png'))   
    plt.close()
    

    plt.figure(figsize=(14,14))
    plt.title('Fixed and Output')
    blended_overlay = np.dstack((fixed[sample_nr,0,...], output[sample_nr,0,...], fixed[sample_nr,0,...])) # R,G,B ?
    blended_overlay = (blended_overlay - np.min(blended_overlay))/(np.max(blended_overlay) - np.min(blended_overlay))
    plt.imshow(blended_overlay)
    if path_saving is not None:
        plt.savefig(os.path.join(path_saving, 'overlay_fixed_output.png'))   
    plt.close()

    plt.figure(figsize=(14,14))
    plt.title('Fixed and Moving')
    blended_overlay = np.dstack((fixed[sample_nr,0,...], moving[sample_nr,0,...], fixed[sample_nr,0,...])) # R,G,B ?
    blended_overlay = (blended_overlay - np.min(blended_overlay))/(np.max(blended_overlay) - np.min(blended_overlay))
    plt.imshow(blended_overlay)
    if path_saving is not None:
        plt.savefig(os.path.join(path_saving, 'overlay_fixed_moving.png'))   
    plt.close()
    
    if ddf is not None:
        u = ddf[sample_nr, 0, ...]
        v = ddf[sample_nr, 1, ...]
        
        # plot displacement vectors
        if coarse_ddf is None:
            plt.figure(figsize=(14,14))
            plt.title('DDF')
            plt.quiver(u[::-1], v[::-1])
        else:
            x, y = np.meshgrid(np.linspace(0,ddf.shape[-2],ddf.shape[-2]),np.linspace(0,ddf.shape[-1],ddf.shape[-1]))
            coarse_x, coarse_y = np.meshgrid(np.linspace(0,ddf.shape[-2],ddf.shape[-2]//coarse_ddf),np.linspace(0,ddf.shape[-1],ddf.shape[-1]//coarse_ddf))
            # Linearly interpolate u and v onto the coarser grid
            coarse_u = griddata((x.ravel(), y.ravel()), u.ravel(), (coarse_x, coarse_y), method='linear')
            coarse_v = griddata((x.ravel(), y.ravel()), v.ravel(), (coarse_x, coarse_y), method='linear')
            
            plt.figure(figsize=(14,14))
            plt.title('DDF')
            plt.quiver(coarse_x, coarse_y, coarse_u[::-1], coarse_v[::-1])
        if path_saving is not None:
            plt.savefig(os.path.join(path_saving, 'ddf.png'))
        plt.close()
        
        fig,ax = plt.subplots(figsize=(14,14))
        plt.title('Moving with DDF')
        pa = ax.quiver(u,v)
        pb = ax.imshow(moving[sample_nr,0,...], cmap='gray', alpha=0.5)
        if path_saving is not None:
            plt.savefig(os.path.join(path_saving, 'overlay_ddf_moving.png'))
        plt.close()
        
        
        # create a grid image
        rows, cols = ddf.shape[-2], ddf.shape[-1]
        grid_image = get_grid_image(rows, cols, spacing=grid_spacing)
        # warp the image using the deformation field and torch spatial transform function
        spatial_trans = utils.SpatialTransformer([ddf.shape[-2], ddf.shape[-1]], mode='bilinear')
        warped_grid_image = spatial_trans(torch.tensor(grid_image[None, None, ...]), torch.tensor(ddf[None, sample_nr]))
        # convert back to numpy
        warped_grid_image = warped_grid_image[0,0,...].detach().cpu().numpy()

        plt.figure(figsize=(14,14))
        plt.title('Warped uniform grid')
        plt.imshow(warped_grid_image, cmap='gray')
        if path_saving is not None:
            plt.savefig(os.path.join(path_saving, 'ddf_on_grid.png'))
        plt.close()
        


def plot_moving_fixed_and_outputs_seg(moving, fixed, fixed_img, output, sample_nr=0, path_saving=None):
    
    # convert tensor to numpy array on cpu if needed
    if torch.is_tensor(moving):
        moving = moving.detach().cpu().numpy()
        fixed = fixed.detach().cpu().numpy()
        fixed_img = fixed_img.detach().cpu().numpy()
        output = output.detach().cpu().numpy()
    elif isinstance(moving, np.ndarray):
        pass
    else:
        raise Exception('Attention: unknown type!') 
    
    plt.figure(figsize=(14,14))
    plt.title('Moving')
    plt.imshow(moving[sample_nr,0,...], cmap='gray')
    if path_saving is not None:
        plt.savefig(os.path.join(path_saving, 'moving_seg.png'))
    plt.close()
    
    plt.figure(figsize=(14,14))
    plt.title('Fixed')
    plt.imshow(fixed[sample_nr,0,...], cmap='gray')
    if path_saving is not None:
        plt.savefig(os.path.join(path_saving, 'fixed_seg.png'))
    plt.close()
    
    plt.figure(figsize=(14,14))
    plt.title('Output (warped moving)')
    plt.imshow(output[sample_nr,0,...], cmap='gray')
    if path_saving is not None:
        plt.savefig(os.path.join(path_saving, 'output_seg.png'))
    plt.close()
         
      
    fig,ax = plt.subplots(figsize=(14,14))
    lw=2
    color_fixed_seg = 'blue'
    color_moving_seg = 'red'
    color_output_seg = 'green'
    plt.title('Fixed image overlayed with fixed, moving and output segmentations')
    ax.imshow(fixed_img[sample_nr,0,...], interpolation='none', cmap='gray', alpha=0.99)
    ax.contour(fixed[sample_nr,0,...], levels=[0.5], linewidths=[lw], colors=[color_fixed_seg])
    ax.contour(moving[sample_nr,0,...], levels=[0.5], linewidths=[lw], colors=[color_moving_seg])
    ax.contour(output[sample_nr,0,...], levels=[0.5], linewidths=[lw], colors=[color_output_seg])
    
    x=228
    y=8
    plt.annotate("Fixed", xy=(x, y), xycoords='data', annotation_clip=False, color=color_fixed_seg)
    plt.annotate("Moving", xy=(x, y+6), xycoords='data', annotation_clip=False, color=color_moving_seg)
    plt.annotate("Output", xy=(x, y+12), xycoords='data', annotation_clip=False, color=color_output_seg)

    
    if path_saving is not None:
        plt.savefig(os.path.join(path_saving, 'overlay_fixed_moving_output_seg.png'))
    plt.close()

        
def box_plot(boxdata, x_labels, y_label, display=True, path_saving=None, stats=False):    
    """
    https://www.geeksforgeeks.org/box-plot-in-python-using-matplotlib/
    """
       
    
    fig, ax = plt.subplots(figsize=(10, 9))
    bp = ax.boxplot(boxdata, labels=x_labels, patch_artist=True)
    
    for median in bp['medians']:
        median.set(color='red', linewidth=2)
    for flier in bp['fliers']:
        flier.set(marker='D', color='black', alpha=0.5)

    ax.set_xticks(np.arange(1, len(x_labels) + 1))
    ax.set_xticklabels(x_labels)
    plt.ylabel(y_label)
    plt.grid(True)

    if stats:
        for i in range(len(x_labels)):
            ax.text(1.03, 0.95 - i*0.05, f'{x_labels[i]} median: {round(np.median(boxdata[i]), 2)}', 
                    fontdict={'color': 'darkred',
                    'weight': 'normal'}, transform=ax.transAxes)    

      
    if path_saving is not None:
        plt.savefig(os.path.join(path_saving, f'{y_label}_boxplot.png'), bbox_inches="tight")
    if display:
        plt.show()   
    plt.close()
    
     
def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    
    return lower_adjacent_value, upper_adjacent_value


def set_axis_style(ax, labels):
    ax.get_xaxis().set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    
                
def violin_plot(boxdata, x_labels, y_label,
                display=True, path_saving=None):
    """
    Get violinplot.
    https://matplotlib.org/3.1.1/gallery/statistics/customized_violin.html
    https://eustomaqua.github.io/2020/2020-03-24-Matplotlib-Tutorial-Gallery/
    
    Args:
        boxdata: list with input data eg [x1, x2]
        x_labels: list with names of input data eg ['Model1', 'Model2']
        y_label: quantity to plot
    """
    
    fig, ax = plt.subplots(figsize=(10, 9))
    vp = ax.violinplot(boxdata, showmeans=False, showmedians=True, showextrema=False)
    for pc in vp['bodies']:
        # pc.set_facecolor('blue')
        pc.set_edgecolor('black')
        pc.set_alpha(0.5)
        
    quartile1, medians, quartile3 = np.percentile(boxdata, [25, 50, 75], axis=1)
    whiskers = np.array([
        adjacent_values(sorted_array, q1, q3) 
        for sorted_array, q1, q3 in zip(boxdata, quartile1, quartile3)])
    whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

    inds = np.arange(1, len(medians) + 1)
    ax.scatter(inds, medians, marker='o', color='white', s=10, zorder=3)
    ax.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=10)
    ax.vlines(inds, whiskers_min, whiskers_max, color='green', linestyle='-', lw=3)

    # set style for the axes
    # for ax in [ax]:
    #     set_axis_style(ax, x_labels)
    ax.set_xticks(np.arange(1, len(x_labels) + 1))
    ax.set_xticklabels(x_labels)

    # plt.title("Target motion")
    plt.ylabel(y_label)
    plt.grid(True)
    
    if path_saving is not None:
        plt.savefig(os.path.join(path_saving, f'{y_label}_violinplot.png'), bbox_inches="tight")
    if display:
        plt.show()
    plt.close()
        

def scatter_hist_3d(x, y, fn, tm, 
                  display=True, save=False, path_saving=None, variant=None):
    """ # Scatter histogram https://stackoverflow.com/questions/6063876/matplotlib-colorbar-for-scatter """
    
    plt.figure(figsize=(10, 7))
    
    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.03  # changing the distance between the histograms and the scatter plot


    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]

    ax = plt.axes(rect_scatter)
    ax.set_xlabel('Post-Ant [mm] \n ' + fn)
    ax.set_ylabel('Inf-Sup [mm]')
    ax_histx = plt.axes(rect_histx)
    ax_histy = plt.axes(rect_histy)
    
    
    # nolabels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)
    
    # the scatter plot
    cm = plt.cm.get_cmap('viridis')
    sc = ax.scatter(x, y, c=tm, cmap=cm)
    cbar = plt.colorbar(sc)
    cbar.set_label('Time [s]', rotation=270, labelpad=22)

    binwidth = 0.15
    # xymax = np.max([np.max(np.fabs(x)), np.max(np.fabs(y))])
    # lim = (int(xymax/binwidth) + 1) * binwidth
    lim = 6
    
   
    bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histx.hist(x, bins=bins, color='#D7191C')
    ax_histy.hist(y, bins=bins, orientation='horizontal')

    ax_histx.set_xlim(ax.get_xlim())
    ax_histy.set_ylim(ax.get_ylim())

    if save:
        plt.savefig(os.path.join(path_saving, fn[:-4] + '_motion_scatterhist3d_' + variant + '.png'),
                    bbox_inches="tight")
    if display:
        plt.show()
    plt.close()
        
        
def random_frame_filling(nf, target, original_target, fn,
                         display=True, save=False, path_saving=None):
    """ Plot a few random frames to check if they were filled. """

    random_frames = np.array([[nf // 5, nf // 3], [nf // 2, nf // 1.3]], dtype=np.int)
    # print(random_frames)
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(16, 16), sharex=True, sharey=True)
    for col in range(2):
        for row in range(2):
            ax[col][row].imshow(target[random_frames[col][row]] + original_target[random_frames[col][row]])
            ax[col][row].set_title(f'Frame {random_frames[col][row]}/{nf}', fontsize=18)

    if save:
        plt.savefig(os.path.join(path_saving, fn[:-4] + '_random_frame_filling.png'), 
                    bbox_inches="tight")
    if display:
        plt.show()
    plt.close()


def motion_plot(tm, cxm, cym, fn, display=True, save=False, path_saving=None):
    """ Plot motion in millimters. """

    fig, axs = plt.subplots(2, sharex=True, sharey=False, figsize=(15, 10))
    axs[0].plot(tm, cxm, 'ko', linestyle='-')
    axs[0].set_ylabel('Post-ant motion [mm]')
    axs[1].plot(tm, cym, 'ko', linestyle='-')
    axs[1].set_ylabel('Inf-sup motion [mm]')  
    axs[1].set_xlabel('Time [s]')
    axs[1].set_xlim(tm[0] - 1, tm[-1] + 1)
    axs[0].grid(True)
    axs[1].grid(True)

    if save:
        plt.savefig(os.path.join(path_saving, fn[:-4] + '_motion_in_mm.png'), bbox_inches="tight")
    if display:
        plt.show()
    plt.close()
            
         
def motion_smoothing_comparison(tm, cx, cx_or, cx_f_or, cy, cy_or, cy_f_or,
                                fn, fps, display=True, save=False, path_saving=None):
    """ Plot original, outlier replaced and filterd motion curves in same subplot to allow for a comparison."""
    
    fig, axs = plt.subplots(2, sharex=True, sharey=False, figsize=(15, 10))

    axs[0].plot(tm, cx, 'ko', linestyle='-', color='black', label='original')
    axs[0].plot(tm, cx_or, 'ko', linestyle='--', color='red', label='replaced')
    axs[0].plot(tm, cx_f_or, 'ko', linestyle='--', color='blue', label='replaced and filtered')
    axs[0].legend()
    axs[0].grid(True)
    axs[0].set_ylabel('Post-ant motion [mm]')

    axs[1].plot(tm, cy, 'ko', linestyle='-', color='black', label='original')
    axs[1].plot(tm, cy_or, 'ko', linestyle='--', color='red', label='replaced')
    axs[1].plot(tm, cy_f_or, 'ko', linestyle='--', color='blue', label='replaced and filtered')
    axs[1].legend()
    axs[1].grid(True)
    axs[1].set_ylabel('Inf-sup motion [mm]')  

    axs[1].set_xlabel('Time [s]')
    axs[1].set_xlim(tm[0], tm[-1] + 1)
    
    if save:
        plt.savefig(os.path.join(path_saving, fn[:-4] + '_eff_motion_smoothing_comparison.png'), 
                    bbox_inches="tight")
    if display:
        plt.show()
        
    # if sequence particularly long, plot only first 100 seconds
    if len(tm) > 100 * fps:
        fig, axs = plt.subplots(2, sharex=True, sharey=False, figsize=(15, 10))

        axs[0].plot(tm[:int(100 * fps)], cx[:int(100 * fps)], 'ko', linestyle='-', 
                    color='black', label='original')
        axs[0].plot(tm[:int(100 * fps)], cx_or[:int(100 * fps)], 'ko', linestyle='--', 
                    color='red', label='replaced')
        axs[0].plot(tm[:int(100 * fps)], cx_f_or[:int(100 * fps)], 'ko', linestyle='--', 
                    color='blue', label='replaced and filtered')
        axs[0].legend()
        axs[0].grid(True)
        axs[0].set_ylabel('Post-ant motion [mm]')

        axs[1].plot(tm[:int(100 * fps)], cy[:int(100 * fps)], 'ko', linestyle='-', 
                    color='black', label='original')
        axs[1].plot(tm[:int(100 * fps)], cy_or[:int(100 * fps)], 'ko', linestyle='--', 
                    color='red', label='replaced')
        axs[1].plot(tm[:int(100 * fps)], cy_f_or[:int(100 * fps)], 'ko', linestyle='--', 
                    color='blue', label='replaced and filtered')
        axs[1].legend()
        axs[1].grid(True)
        axs[1].set_ylabel('Inf-sup motion [mm]') 

        axs[1].set_xlabel('Time [s]')

        if save:
            plt.savefig(os.path.join(path_saving, fn[:-4] + '_eff_motion_smoothing_comparison_100s.png'), 
                        bbox_inches="tight")
        if display:
            plt.show()
    plt.close()
                
            
def motion_with_info(tm, cx, cx_or, cx_f_or, cy, cy_or, cy_f_or,
                    status, breathholds, framenumb_all, imagepauses, fn, 
                    fps, display=True, save=False, path_saving=None):
    
    """" Plot motion curves plus beam status, image pauses and breathhold information. """
    
    # defined globally for all figures
    plt.rcParams['axes.grid'] = True
    plt.rc('xtick', labelsize=25) 
    plt.rc('ytick', labelsize=25)
    plt.rc('axes', labelsize=25)  # to change the size of the letters

    fig, axs = plt.subplots(5, sharex=True, sharey=False, figsize=(25, 20), 
                            gridspec_kw={'height_ratios': [2, 2, 1, 1, 1]})
    axs[0].plot(tm, cx, 'ko', linestyle='-', color='black', label='original')
    axs[0].plot(tm, cx_or, 'ko', linestyle='-', color='red', label='replaced')
    axs[0].plot(tm, cx_f_or, 'ko', linestyle='-', color='blue', label='replaced and filtered')
    axs[0].set_ylabel('Post-ant motion [mm]')

    axs[1].plot(tm, cy, 'ko', linestyle='-', color='black', label='original')
    axs[1].plot(tm, cy_or, 'ko', linestyle='-', color='red', label='replaced')
    axs[1].plot(tm, cy_f_or, 'ko', linestyle='-', color='blue', label='replaced and filtered')
    axs[1].set_ylabel('Inf-sup motion [mm]')  

    axs[1].set_xlim(tm[0], tm[-1] + 1)

    for i in range(len(cy_f_or)):
        # beam status
        if status[i] == 'on':
            axs[2].axvline(x=tm[i], ymin=0, ymax=1, color='g', linewidth=4)
            axs[2].set_ylabel('Beam status On')
            plt.setp(axs[2].get_yticklabels(), visible=False)
        
        # breath-holds
        if breathholds[i] == 1:
            axs[3].axvline(x=tm[i], ymin=0, ymax=1, color='r', linewidth=4)
            axs[3].set_ylabel('Breath-holds')
            plt.setp(axs[3].get_yticklabels(), visible=False)

    if np.sum(imagepauses) > 0:
        for i in range(len(framenumb_all)):
            # image pauses
            if imagepauses[i] == 1:
                axs[4].axvline(x=framenumb_all[i] / fps, ymin=0, ymax=1, color='r', linewidth=4)
                axs[4].set_ylabel('Imaging paused')
                axs[4].set_xlabel('Time [s]')
                plt.setp(axs[4].get_yticklabels(), visible=False)

    plt.tight_layout()

    if save:
        plt.savefig(os.path.join(path_saving, fn[:-4] + '_eff_motion_info.png'), 
                    bbox_inches="tight")
    if display:
        plt.show()
        
    # if sequence particularly long, plot only first 100 seconds
    if len(tm) > 100 * fps: 
        # defined globally for all figures
        plt.rcParams['axes.grid'] = True
        plt.rc('xtick', labelsize=25) 
        plt.rc('ytick', labelsize=25)
        plt.rc('axes', labelsize=25)  # to change the size of the letters

        fig, axs = plt.subplots(5, sharex=True, sharey=False, figsize=(25, 20),
                                gridspec_kw={'height_ratios': [1, 1, 0.5, 0.5, 0.5]})
        axs[0].plot(tm[:int(100 * fps)], cx[:int(100 * fps)], 'ko', linestyle='-', color='black', label='original')
        axs[0].plot(tm[:int(100 * fps)], cx_or[:int(100 * fps)], 'ko', linestyle='-', 
                    color='red', label='replaced and filtered')
        axs[0].plot(tm[:int(100 * fps)], cx_f_or[:int(100 * fps)], 'ko', linestyle='-', 
                    color='blue', label='replaced and filtered')
        axs[0].set_ylabel('Post-ant motion [mm]')

        axs[1].plot(tm[:int(100 * fps)], cy[:int(100 * fps)], 'ko', linestyle='-', 
                    color='black', label='original')
        axs[1].plot(tm[:int(100 * fps)], cy_or[:int(100 * fps)], 'ko', linestyle='-', 
                    color='red', label='replaced and filtered')
        axs[1].plot(tm[:int(100 * fps)], cy_f_or[:int(100 * fps)], 'ko', linestyle='-', 
                    color='blue', label='replaced and filtered')
        axs[1].set_ylabel('Inf-sup motion [mm]')  


        for i in range(int(100 * fps)):
            if status[i] == 'on':
                axs[2].axvline(x=tm[i], ymin=0, ymax=1, color='g', linewidth=4)
                axs[2].set_ylabel('Beam status On')
                plt.setp(axs[2].get_yticklabels(), visible=False)

        # breath-holds
            if breathholds[i] == 1:
                axs[3].axvline(x=tm[i], ymin=0, ymax=1, color='r', linewidth=4)
                axs[3].set_ylabel('Breath-holds')
                plt.setp(axs[3].get_yticklabels(), visible=False)

            if np.sum(imagepauses) > 0:
                # image pauses
                if imagepauses[i] == 1:
                    axs[4].axvline(x=framenumb_all[i] / fps, ymin=0, ymax=1, color='r', linewidth=4)
                    axs[4].set_ylabel('Imaging paused')
                    axs[4].set_xlabel('Time [s]')
                    plt.setp(axs[4].get_yticklabels(), visible=False)


        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(path_saving, fn[:-4] + '_eff_motion_info_100s.png'), 
                        bbox_inches="tight")
        if display:
            plt.show()
    
    plt.close()


def motion_with_status_pause_info(tm, cx, cy,
                    status, framenumb_all, imagepauses, fn, 
                    fps, display=True, save=False, path_saving=None):
    
    """" Plot motion curves plus beam status, image pauses and breathhold information. """
    
    # defined globally for all figures
    plt.rcParams['axes.grid'] = True
    plt.rc('xtick', labelsize=25) 
    plt.rc('ytick', labelsize=25)
    plt.rc('axes', labelsize=25)  # to change the size of the letters

    if np.sum(imagepauses) > 0:
        fig, axs = plt.subplots(4, sharex=True, sharey=False, figsize=(25,20),gridspec_kw={'height_ratios': [2,2,1,1]})
    else:
        fig, axs = plt.subplots(3, sharex=True, sharey=False, figsize=(25,20),gridspec_kw={'height_ratios': [2,2,1]})

    axs[0].plot(tm, cx, 'ko', linestyle='-', color='black')
    axs[0].set_ylabel('Post-ant motion [mm]')

    axs[1].plot(tm, cy, 'ko', linestyle='-', color='black')
    axs[1].set_ylabel('Inf-sup motion [mm]')  

    axs[1].set_xlim(tm[0], tm[-1] + 1)

    for i in range(len(cy)):
        # beam status
        if status[i] == 'on':
            axs[2].axvline(x=tm[i], ymin=0, ymax=1, color='g', linewidth=4)
            axs[2].set_ylabel('Beam status On')
            if len(imagepauses) == 0:
                axs[2].set_xlabel('Time [s]')
            plt.setp(axs[2].get_yticklabels(), visible=False)
    
    if np.sum(imagepauses) > 0:
        for i in range(len(framenumb_all)):
            # image pauses
            if imagepauses[i] == 1:
                axs[3].axvline(x=framenumb_all[i] / fps, ymin=0, ymax=1, color='r', linewidth=4)
                axs[3].set_ylabel('Imaging paused')
                axs[3].set_xlabel('Time [s]')
                plt.setp(axs[3].get_yticklabels(), visible=False)

    plt.tight_layout()

    if save:
        plt.savefig(os.path.join(path_saving, fn[:-4] + '_eff_motion_info.png'), 
                    bbox_inches="tight")
    if display:
        plt.show()
        
    # if sequence particularly long, plot only first 100 seconds
    if len(tm) > 100 * fps: 
        # defined globally for all figures
        plt.rcParams['axes.grid'] = True
        plt.rc('xtick', labelsize=25) 
        plt.rc('ytick', labelsize=25)
        plt.rc('axes', labelsize=25)  # to change the size of the letters

        if len(imagepauses) > 0:
            fig, axs = plt.subplots(4, sharex=True, sharey=False, figsize=(25,20),gridspec_kw={'height_ratios': [2,2,1,1]})
        else:
            fig, axs = plt.subplots(3, sharex=True, sharey=False, figsize=(25,20),gridspec_kw={'height_ratios': [2,2,1]})
            
            
        axs[0].plot(tm[:int(100 * fps)], cx[:int(100 * fps)], 'ko', linestyle='-', color='black')
        axs[0].set_ylabel('Post-ant motion [mm]')

        axs[1].plot(tm[:int(100 * fps)], cy[:int(100 * fps)], 'ko', linestyle='-', 
                    color='black')
        axs[1].set_ylabel('Inf-sup motion [mm]')  


        for i in range(int(100 * fps)):
            if status[i] == 'on':
                axs[2].axvline(x=tm[i], ymin=0, ymax=1, color='g', linewidth=6)
                axs[2].set_ylabel('Beam status On')
                if len(imagepauses) == 0:
                    axs[2].set_xlabel('Time [s]')
                plt.setp(axs[2].get_yticklabels(), visible=False)

            if len(imagepauses) > 0:
                # image pauses
                if imagepauses[i] == 1:
                    axs[3].axvline(x=framenumb_all[i] / fps, ymin=0, ymax=1, color='r', linewidth=6)
                    axs[3].set_ylabel('Imaging paused')
                    axs[3].set_xlabel('Time [s]')
                    plt.setp(axs[3].get_yticklabels(), visible=False)


        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(path_saving, fn[:-4] + '_eff_motion_info_100s.png'), 
                        bbox_inches="tight")
        if display:
            plt.show()
    
    plt.close()

        

def losses_plot(train_losses=None, val_losses=None, loss_name=None,
                display=False, last_epochs=50, 
                path_saving=None):
    
    plt.figure(figsize=(10, 7))
    plt.rcParams.update({'font.size': 20})
    
    if train_losses is not None:
        plt.plot(train_losses, 'o-', label="Training loss")
    if val_losses is not None:
        plt.plot(val_losses, 'o-', label="Validation loss")
    
    plt.ylabel(f"{loss_name}")
    plt.xlabel("Epoch number")
    plt.legend()    
    
    if path_saving is not None:
        plt.savefig(path_saving, bbox_inches="tight")
    if display:
        plt.show()
    plt.close()
    
    if last_epochs is not None:
        plt.figure(figsize=(10, 7))
    
        plt.plot(train_losses[-last_epochs:], 'o-', label="Training loss")
        if val_losses is not None:
            plt.plot(val_losses[-last_epochs:], 'o-', label="Validation loss")
        
        plt.ylabel(f"{loss_name}")
        plt.xlabel("Epoch number")
        plt.legend() 
        
        if path_saving is not None:
            plt.savefig(path_saving, bbox_inches="tight")
        if display:
            plt.show()
        plt.close() 


def losses_plot_detailed(train_losses=None, val_losses=None, 
                        loss_name=None,
                        log=False,
                        display=False, 
                        save=False, path_saving=None, 
                        info_loss=''):
 
    plt.rcParams.update({'font.size': 22})   
    plt.figure(figsize=(10, 7))
    
    if log:
        # set logarithmic axis
        axs = plt.axes(yscale='log')
    else:
        axs = plt.axes()        
    
    if train_losses is not None:
        axs.plot(train_losses, '-', label="Training loss")
    if val_losses is not None:
        axs.plot(val_losses, '-', label="Validation loss")
    
    axs.set_ylabel(f"Normalized {loss_name}")
    # axs.set_ylim(min(train_losses), max(train_losses));
    axs.set_xlabel("Epoch number")
    # axs.set_xlim(-1, len(train_losses) + 1);
    
    # set legend and grid
    axs.legend()    
    axs.grid(linestyle='dashed')
    
    # set minor ticks on and ticks on both sides
    # axs.xaxis.set_major_locator(plt.MultipleLocator(1))
    # axs.yaxis.set_major_locator(plt.MultipleLocator(1))
    axs.minorticks_on()
    axs.xaxis.set_minor_locator(plt.MaxNLocator())
    axs.tick_params(labeltop=False, labelright=False)
    axs.tick_params(which='both', top=True, right=True)
    
    if log is False:
        # set scientific notation for y axis
        axs.ticklabel_format(axis='y', 
                             style='sci',
                             scilimits=(0, 0))

    if save:
        plt.savefig(os.path.join(path_saving, info_loss + 'losses.png'), bbox_inches="tight")
    if display:
        plt.show()
    plt.close()
    
           
        
def predicted_wdw_plot(x, y, y_pred, wdw_nr=-1, last_pred=True,
                       display=True, path_saving=None):
    """ Plot ground truth vs predict time series window.

    Args:
        x (Pytorch tensor or np.array): ground truth input windows, shape = [batch_size, wdw_size_o] 
        y (Pytorch tensor or np.array): ground truth output windows, shape = [batch_size, wdw_size_o] 
        y_pred (Pytorch tensor): predicted output windows, shape = [batch_size, wdw_size_o] 
        wdw_nr (int, optional): window nr in list with windows to be plotted. Defaults to -1.
        last_pred (bool, optional): whether to plot only the last prediction. Defaults to True.
        display (bool, optional): whether to dislpay plot (works with Ipython). Defaults to True.
        save (bool, optional): whether to save plot. Defaults to False.
        path_saving ([type], optional): path where plot is saved. Defaults to None.
    """
    
    # take last wdw, changing shape to (wdw_size_i,)
    x = x[wdw_nr, ...]  
    y = y[wdw_nr, ...]
    y_pred = y_pred[wdw_nr, ...]    

    # convert tensor to numpy arrays on cpu 
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        pass
    else:
        raise Exception('Attention: unknown type')
    
    # create time axis
    t = np.arange(len(x) + len(y))
    
    # print(np.shape(x))  # (8,)
    # print(np.shape(y))  # (1,)
    # print(np.shape(y_pred))  # (1,)
  
    plt.figure(figsize=(10, 7))  
    plt.plot(t[:len(x)], x, 'o-', color='black', label="True input")
    plt.plot(t[len(x):len(x) + len(y)], y, 'o-', color='blue', label="True output")
    
    if last_pred:
        plt.plot(t[len(x) + len(y) - 1], y_pred[-1], 'o-', color='red', label="Predicted output")
    else:
        plt.plot(t[len(x):len(x) + len(y)], y_pred, 'o-', color='red', label="Predicted output")

    
    plt.ylabel("Relative amplitude")
    plt.xlabel("Time step")
    plt.ylim([-1, 1])
    plt.legend()
    plt.grid()
    
    plt.title(f"Ground truth and predicted sequences")
    
    if path_saving is not None:
        plt.savefig(path_saving, bbox_inches="tight")
    if display:
        plt.show() 
    plt.close()
            
        
        
def predicted_snippets_plot(y_pred, y_batch, normalization=True, 
                            first_points=None, last_points=None,
                            display=True, save=False, path_saving=None):
    """ Plot ground truth vs predict time series.

    Args:
        y_batch (list of Pytorch tensors or or np.array): ground truth output series
        y_pred (list of Pytorch tensors or np.array): predicted output series
        first_points(int, optional): whether to plot only specified nr of first time points
        last_points(int, optional): whether to plot only specified nr of last time points
        display (bool, optional): whether to dislpay plot (works with Ipython). Defaults to True.
        save (bool, optional): whether to save plot. Defaults to False.
        path_saving ([type], optional): path where plot is saved. Defaults to None.
    """
  
    if torch.is_tensor(y_pred[0]):
        # get tensor out of list of tensors
        y_pred = torch.stack(y_pred)
        y_batch = torch.stack(y_batch)
        # get numpy arrays on CPU
        y_pred = y_pred.detach().cpu().numpy()
        y_batch = y_batch.detach().cpu().numpy()
    elif isinstance(y_pred[0], np.ndarray):
        # get array out of list of arrays
        y_pred = np.concatenate(y_pred)
        y_batch = np.concatenate(y_batch)
    else:
        print(y_pred[0])
        raise Exception('Attention: unknown type')

    # create time axis
    t = np.arange(len(y_pred)) / 4
    

    # print(np.shape(y_pred))  # (324, 1)
    # print(np.shape(y_batch))  # (324, 1)
  
    plt.figure(figsize=(10, 7))  
    if last_points is not None:
        plt.plot(t[:last_points], y_batch[-last_points:], 'o-', color='black', label="True")
        plt.plot(t[:last_points], y_pred[-last_points:], '*-', color='blue', label="Predicted")
    elif first_points is not None:
        plt.plot(t[:first_points], y_batch[:first_points], 'o-', color='black', label="True")
        plt.plot(t[:first_points], y_pred[:first_points], '*-', color='blue', label="Predicted")
    else:
        plt.plot(t, y_batch, 'o-', color='black', label="True")
        plt.plot(t, y_pred, '*-', color='blue', label="Predicted")
        
            
    if normalization:
        plt.ylabel("Relative amplitude")
        plt.ylim([-1, 1])
    else:
        plt.ylabel("Amplitude [mm]")      
          
    plt.xlabel("Time [s]")
    plt.legend()
    
    plt.title(f"Ground truth vs. predicted snippets")
    
    if save:
        plt.savefig(os.path.join(path_saving, 
                                 f'predicted_snippets_norm{normalization}_last_points{last_points}_first_points{first_points}.png'), 
                    bbox_inches="tight")
    if display:
        plt.show() 
        
    plt.close()


def predicted_snippets_comparison(y_pred_1, y_pred_2, y_batch, 
                            video_nr, 
                            normalization=True, 
                            first_points=None, last_points=None,
                            display=True, save=False, 
                            legend=True,
                            info='', path_saving=None,
                            fs=16):
    """ Plot ground truth vs predict time series for two different models.

    Args:
        y_batch (list of Pytorch tensors or or np.array): ground truth output series
        y_pred_1 (list of Pytorch tensors or np.array): predicted output series for LR model
        y_pred_2 (list of Pytorch tensors or np.array): predicted output series for LSTM model
        video_nr (int): snippet number picked for plotting
        first_points(int, optional): whether to plot only specified nr of first time points
        last_points(int, optional): whether to plot only specified nr of last time points
        display (bool, optional): whether to dislpay plot (works with Ipython). Defaults to True.
        save (bool, optional): whether to save plot. Defaults to False.
        legend (bool, optional): whether to plot legend or not. Defaults to True
        info (str): additional info for name of saved plot  
        path_saving ([type], optional): path where plot is saved. Defaults to None.
        fs (int): fontsize for labels,legend etc.
    """
    # change fontsize for label, legend etc.
    plt.rcParams.update({'font.size': fs})
    
    if torch.is_tensor(y_pred_1[0]):
        # get tensor out of list of tensors
        y_pred_1 = torch.stack(y_pred_1)
        # get numpy arrays on CPU
        y_pred_1 = y_pred_1.detach().cpu().numpy()
    elif isinstance(y_pred_1[0], np.ndarray):
        # get array out of list of arrays
        y_pred_1 = np.concatenate(y_pred_1)
    else:
        print(y_pred_1[0])
        raise Exception('Attention: unknown type for y_pred_1')
    
    if torch.is_tensor(y_pred_2[0]):
        # get tensor out of list of tensors
        y_pred_2 = torch.stack(y_pred_2)
        # get numpy arrays on CPU
        y_pred_2 = y_pred_2.detach().cpu().numpy()
    elif isinstance(y_pred_2[0], np.ndarray):
        # get array out of list of arrays
        y_pred_2 = np.concatenate(y_pred_2)
    else:
        print(y_pred_2[0])
        raise Exception('Attention: unknown type for y_pred_2')
    
    
    if torch.is_tensor(y_batch[0]):
        # get tensor out of list of tensors
        y_batch = torch.stack(y_batch)
        # get numpy arrays on CPU
        y_batch = y_batch.detach().cpu().numpy()
    elif isinstance(y_batch[0], np.ndarray):
        # get array out of list of arrays
        y_batch = np.concatenate(y_batch)
    else:
        print(y_batch[0])
        raise Exception('Attention: unknown type for y_batch')
    
    # create time axis
    t = np.arange(len(y_pred_1)) / 4
    
    # print(np.shape(y_pred_1))  # (324, 1)
    # print(np.shape(y_batch))  # (324, 1)
  
    plt.figure(figsize=(10, 7))  
    if last_points is not None:
        plt.plot(t[:last_points], y_batch[-last_points:], 'o-', color='black', label="True")
        plt.plot(t[:last_points], y_pred_1[-last_points:], '*--', color='blue', label="LR")
        plt.plot(t[:last_points], y_pred_2[-last_points:], 'd--', color='red', label="LSTM")
    elif first_points is not None:
        plt.plot(t[:first_points], y_batch[:first_points], 'o-', color='black', label="True")
        plt.plot(t[:first_points], y_pred_1[:first_points], '*--', color='blue', label="LR")
        plt.plot(t[:first_points], y_pred_2[:first_points], 'd--', color='red', label="LSTM")    
    else:
        start = 0
        stop = 57
        plt.plot(t[start:stop], y_batch[start:stop], 'o-', color='black', label="True")
        plt.plot(t[start:stop], y_pred_1[start:stop], '*--', color='blue', label="LR")        
        plt.plot(t[start:stop], y_pred_2[start:stop], 'd--', color='red', label="LSTM")    
           
    if normalization:
        plt.ylabel("Relative amplitude")
        plt.ylim([-1, 1])
    else:
        plt.ylabel("Amplitude [mm]")      
          
    plt.xlabel("Time [s]")
    if legend:
        plt.legend()
    plt.grid()
    
    
    if save:
        if (first_points is None) and (last_points is None):
            plt.savefig(os.path.join(path_saving, 
                                    f'predicted_snippet{video_nr}_norm{normalization}_start{start}_stop{stop}_{info}.png'), 
                        bbox_inches="tight")
        else:            
            plt.savefig(os.path.join(path_saving, 
                                    f'predicted_snippet{video_nr}_norm{normalization}_last_points{last_points}_first_points{first_points}_{info}.png'), 
                        bbox_inches="tight")
    if display:
        plt.show() 
        
    plt.close()


def diff_violinplot(y_pred_1, y_pred_2, y_batch,
                            display=True, save=False, 
                            info='', path_saving=None):
    """ Plot ground truth vs predict time series.

    Args:
        y_batch (list of Pytorch tensors or or np.array): ground truth output series
        y_pred (list of Pytorch tensors or np.array): predicted output series
        first_points(int, optional): whether to plot only specified nr of first time points
        last_points(int, optional): whether to plot only specified nr of last time points
        display (bool, optional): whether to dislpay plot (works with Ipython). Defaults to True.
        save (bool, optional): whether to save plot. Defaults to False.
        info (str): additional info for name of saved plot          
        path_saving ([type], optional): path where plot is saved. Defaults to None.
    """
    diff_true_1 = np.array([])
    diff_true_2 = np.array([])
    
    for y_pred_case_1, y_pred_case_2, y_batch_case in zip(y_pred_1, y_pred_2, y_batch):   
        if torch.is_tensor(y_pred_case_1[0]):
            # get tensor out of list of tensors
            y_pred_case_1 = torch.stack(y_pred_case_1)
            # get numpy arrays on CPU
            y_pred_case_1 = y_pred_case_1.detach().cpu().numpy()
        elif isinstance(y_pred_case_1[0], np.ndarray):
            # get array out of list of arrays
            y_pred_case_1 = np.concatenate(y_pred_case_1)[:, np.newaxis] 
        else:
            print(y_pred_case_1[0])
            raise Exception('Attention: unknown type for y_pred_case_1')
        
        if torch.is_tensor(y_pred_case_2[0]):
            # get tensor out of list of tensors
            y_pred_case_2 = torch.stack(y_pred_case_2)
            # get numpy arrays on CPU
            y_pred_case_2 = y_pred_case_2.detach().cpu().numpy()
        elif isinstance(y_pred_case_2[0], np.ndarray):
            # get array out of list of arrays
            y_pred_case_2 = np.concatenate(y_pred_case_2)[:, np.newaxis] 
        else:
            print(y_pred_case_2[0])
            raise Exception('Attention: unknown type for y_pred_case_2')
        
        if torch.is_tensor(y_batch_case[0]):
            # get tensor out of list of tensors
            y_batch_case = torch.stack(y_batch_case)
            # get numpy arrays on CPU
            y_batch_case = y_batch_case.detach().cpu().numpy()
        elif isinstance(y_batch_case[0], np.ndarray):
            # get array out of list of arrays
            y_batch_case = np.concatenate(y_batch_case)[:, np.newaxis] 
        else:
            print(y_batch_case[0])
            raise Exception('Attention: unknown type for y_batch_case')
        
        # print(f'shape y_pred_case_1: {np.shape(y_pred_case_1)}')  # (528, 1)
        # print(f'shape y_pred_case_2: {np.shape(y_pred_case_2)}')  # (528, 1)
        # print(f'shape y_batch_case: {np.shape(y_batch_case)}')  # (528, 1)

        # build difference between ground truth and predicted curves for video
        current_diff_true_1 = np.subtract(y_batch_case, y_pred_case_1)
        current_diff_true_2 = np.subtract(y_batch_case, y_pred_case_2)
        
        # append to array with all differences
        diff_true_1 = np.append(diff_true_1, current_diff_true_1)
        diff_true_2 = np.append(diff_true_2, current_diff_true_2)
    
    # print(f'shape diff_true_1: {np.shape(diff_true_1)}')  # (13359,)
    # print(f'shape diff_true_2: {np.shape(diff_true_2)}')  # (13359,)

    # do violin plot of signed difference between ground truth and predicted curves
    boxdata = [sorted(diff_true_1), sorted(diff_true_2)] 

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_axes([0, 0, 1, 1])
    vp = ax.violinplot(boxdata, showmeans=False, showmedians=True, showextrema=True)
    for pc in vp['bodies']:
        # pc.set_facecolor('blue')
        pc.set_edgecolor('black')
        pc.set_alpha(0.5)
        
    q1_1, med_1, q3_1 = np.percentile(diff_true_1, [25, 50, 75], axis=0)  
    q1_2, med_2, q3_2 = np.percentile(diff_true_2, [25, 50, 75], axis=0)  

    
    quartiles1, medians, quartiles3 = [q1_1, q1_2], [med_1, med_2], [q3_1, q3_2]   

    inds = np.arange(1, len(medians) + 1)
    ax.scatter(inds, medians, marker='o', color='white', s=10, zorder=3)
    ax.vlines(inds, quartiles1, quartiles3, color='k', linestyle='-', lw=10)
    # ax.vlines(inds, whiskers_min, whiskers_max, color='green', linestyle='-', lw=3)

    # set style for the axes
    labels = ['True - LR', 'True - LSTM']
    for ax in [ax]:
        set_axis_style(ax, labels)

    # plt.title(" Target motion", fontsize=25)
    plt.ylabel("Difference between ground truth and prediction [mm]")
    plt.grid(True)

    if save:
        plt.savefig(os.path.join(path_saving, 
                                 f'difference_violinplots_{info}.png'), 
                    bbox_inches="tight")
    if display:
        plt.show() 
        
    plt.close()
    

def plot_random_dvfs(grid_size=10):
    # Generate a random deformation vector
    deformation_x_original = np.random.rand() - 0.5
    deformation_y_original = np.random.rand() - 0.5

    # Replicate the deformation vector across the grid
    deformation_x = np.full((grid_size, grid_size), deformation_x_original) + np.random.rand(grid_size, grid_size)/5
    deformation_y = np.full((grid_size, grid_size), deformation_y_original) + np.random.rand(grid_size, grid_size)/5

    deformation_x_larger = np.full((grid_size, grid_size), deformation_x_original) + np.random.rand(grid_size, grid_size)/4
    deformation_y_larger = np.full((grid_size, grid_size), deformation_y_original) + np.random.rand(grid_size, grid_size)/4

    # Create a grid for plotting
    x = np.arange(grid_size)
    y = np.arange(grid_size)
    X, Y = np.meshgrid(x, y)

    # Plot the deformation vector field in black
    plt.figure()
    plt.quiver(X, Y, deformation_x, deformation_y, color='black')
    plt.title('Uniform Deformation Vector Field (Black)')
    plt.axis('off')

    # Plot the deformation vector field in red
    plt.figure()
    plt.quiver(X, Y, deformation_x_larger, deformation_y_larger, color='red')
    plt.title('Uniform Deformation Vector Field (Red)')
    plt.axis('off')

    # Display the plots
    plt.show()
    
    
def motion_analysis_plot(path_labels_folder, only_first=40, display=True, path_saving=None):
    """ Plot motion curve as a function of frames for eg patients binary masks """
    
    # get imaging frequency
    if '8FPS' in path_labels_folder:
        f = 8
    else:
        f = 4
    
    # Get a list of all NIfTI files in the given folder
    nifti_files = [file for file in os.listdir(path_labels_folder) if file.endswith('.nii.gz')]

    if not nifti_files:
        raise ValueError("No NIfTI files found in the specified folder!")

    # Initialize lists to store centers of mass in X and Y directions
    center_of_mass_x = []
    center_of_mass_y = []
    
    # get actual numbers of each file path
    nifti_files_numbers = utils.get_frame_nr(nifti_files)
    
    # sort the lists according to frame numbers
    nifti_files_numbers, nifti_files = zip(*sorted(zip(nifti_files_numbers, nifti_files)))
    frame_numbers = list(nifti_files_numbers)
    
    # Iterate through each NIfTI file
    for nr, nifti_file in zip(frame_numbers, nifti_files):
        # print(nifti_file)
        # Load the NIfTI file
        nifti_image = nib.load(os.path.join(path_labels_folder, nifti_file))

        # Get the data array from the NIfTI file
        mask = nifti_image.get_fdata()
        # print(mask.shape)  # 270,270,1
        # plt.figure()
        # plt.imshow(mask[:,:,0])

        # Compute the center of mass in X and Y directions
        center_x, center_y = scipy.ndimage.center_of_mass(mask[:,:,0])


        if np.isnan(center_x) or np.isnan(center_y):
            print(f'Attention: center of mass is nan for {nifti_file} ! Skipping this one')
            frame_numbers.remove(nr)
        else:
            center_of_mass_x.append(center_x)
            center_of_mass_y.append(center_y)
        
    # zero center
    # center_of_mass_x = center_of_mass_x - np.mean(center_of_mass_x)
    # center_of_mass_y = center_of_mass_y - np.mean(center_of_mass_y)
    
    # compute IQR of motion
    q3_x, q1_x = np.percentile(np.array(center_of_mass_x), [75 ,25])
    iqr_x = q3_x - q1_x
    q3_y, q1_y = np.percentile(np.array(center_of_mass_y), [75 ,25])
    iqr_y = q3_y - q1_y

    # plot
    fig, axs = plt.subplots(2, sharex=True, sharey=False, figsize=(9,7))
    axs[0].set_title(f'IQR = {iqr_x} mm - {f} Hz')
    if only_first is not None:
        axs[0].plot(frame_numbers[:only_first], center_of_mass_x[:only_first], 'ko', linestyle='-')
    else:
       axs[0].plot(frame_numbers, center_of_mass_x, 'ko', linestyle='-') 
    axs[0].set_ylabel('SI motion [mm]')
    axs[1].set_title(f'IQR = {iqr_y} mm - {f} Hz')
    if only_first is not None:
        axs[1].plot(frame_numbers[:only_first], center_of_mass_y[:only_first], 'ko', linestyle='-')
    else:
        axs[1].plot(frame_numbers, center_of_mass_y, 'ko', linestyle='-')
    axs[1].set_ylabel('AP motion [mm]')  
    axs[1].set_xlabel('Frame number')
    # axs[1].set_xlim(frame_numbers[0], frames[-1] + 1)
    axs[0].grid(True)
    axs[1].grid(True)

    print(f'Total nr of frames: {len(frame_numbers)}')
    
    if path_saving is not None:
        plt.savefig(os.path.join(path_saving, 'motion_analysis_plot.png'), bbox_inches="tight")
    if display:
        plt.show()
    plt.close()


# %%

if __name__ == "__main__":   
    path_to_labels = '/home/segmentation_at_MRIdian/data/testing/images/final_with_contours/contoured_ground_truth_LV/abdomen_patient0006/raw_cine/20210510150232_converted_output_scan_6_2D_rad_FOV270_8FPS_th5_sag/labels'
    # path_to_labels = '/home/segmentation_at_MRIdian/data/testing/images/final_with_contours/contoured_ground_truth_VG/abdomen_patient0008/raw_cine/20210727105422_converted_output_scan_8_2D_rad_FOV270_8FPS_th5_sag/labels'
    # path_to_labels = '/home/segmentation_at_MRIdian/data/testing/images/final_with_contours/contoured_ground_truth_VG/abdomen_patient0017/raw_cine/20211202130059_converted_output_scan_5_2D_rad_FOV270_8FPS_th5_sag/labels'
    # path_to_labels = '/home/segmentation_at_MRIdian/data/testing/images/final_with_contours/contoured_ground_truth_LV/abdomen_patient0017/raw_cine/20211202130059_converted_output_scan_5_2D_rad_FOV270_8FPS_th5_sag/labels'
    
    motion_analysis_plot(path_labels_folder=path_to_labels, only_first=None)
    
# %%
