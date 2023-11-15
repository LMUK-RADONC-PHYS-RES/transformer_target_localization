import numpy as np
import time
import torch
import os, gc
import tqdm
from torch.nn import MSELoss
from monai.losses import DiceLoss, LocalNormalizedCrossCorrelationLoss
from monai.losses.ssim_loss import SSIMLoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric, RMSEMetric, MSEMetric
from monai.metrics.regression import SSIMMetric
import wandb

# import self written modules
import losses, metrics, plotting, utils


def train_val_model_unsupervised(model, train_loader, optimizer, loss_name, loss_weights, epoch_nr, device,
                                    lr_scheduler=None, early_stopping_patience=None, epoch_start=0, val_loader=None,
                                    val_loader_supervised=None, path_saving=None, wandb_usage=False, plot=False):
    """Function to train a Pytorch model in an unsupervised fashion and validate (unsupervised and supervised).

    Args:
        model (Pytorch model): Network architecture
        train_loader (Pytorch laoder): pytorch data loader with training input and output
        loss_name (str): name of loss function, eg 'MSE'
        loss_weights (list): list with weights of combined loss if used, e.g. [1, 1] where 0th element is for image and 1st for displacmements
        optimizer (Pytorch optimizer): optimizer used to update network's weights
        lr_scheduler (Pytorch scheduler): scheduler for learning rate
        epoch_nr (int): maximum number of training epochs
        device (Pytorch device): cuda device
        early_stopping_patience (int, optional): whether to stop the optimization if the loss does
                                not get better after early_stopping nr of epochs. Defaults to None.
        epoch_start: starting epoch for loop
        val_loader (Pytorch laoder): pytorch data loader with validation input and output
        val_loader_supervised (Pytorch laoder): pytorch data loader with validation input and output with labels
        path_saving (str): path to folder where models are saved. Defaults to None.
        plot (str): plot some results
    """
    
    # set data range for structural similiraty
    data_range = torch.tensor([1.0], device=device)
        
    # initialize metrics
    mse_metric_train = MSEMetric(reduction="mean", get_not_nans=False)
    mse_metric_val = MSEMetric(reduction="mean", get_not_nans=False)
    ssim_metric_train = SSIMMetric(data_range=data_range, win_size=7, k1=0.01, k2=0.03, spatial_dims=2, reduction="mean", get_not_nans=False)
    ssim_metric_val = SSIMMetric(data_range=data_range, win_size=7, k1=0.01, k2=0.03, spatial_dims=2, reduction="mean", get_not_nans=False)
    if val_loader_supervised is not None:
        dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False, ignore_empty=True)
        hd95_metric = HausdorffDistanceMetric(include_background=False, distance_metric='euclidean', percentile=95, 
                                                        directed=False, reduction="mean", get_not_nans=False)
        hd50_metric = HausdorffDistanceMetric(include_background=False, distance_metric='euclidean', percentile=50, 
                                                        directed=False, reduction="mean", get_not_nans=False)
        negj_metric = metrics.NegativeJacobianMetric(reduction="mean", get_not_nans=False)

    best_val_value = 1000000  # to be sure loss decreases

    t0 = time.time()
    
    # loop over all epochs
    for epoch in range(epoch_start, epoch_start + epoch_nr):
        model.train()
        losses_train = [] 

        # loop over all batches of data
        with tqdm.tqdm(train_loader, unit="batch", initial=0) as tepoch:
            for train_batch_data in tepoch:
                tepoch.set_description(f"Epoch {epoch}/{epoch_start + epoch_nr - 1}")
                if wandb_usage:
                    wandb.log({"lr": optimizer.param_groups[0]['lr']})  
                
                # clear stored gradients
                optimizer.zero_grad()
                
                # get batch of data
                train_inputs, train_targets = train_batch_data["moving_image"].to(device), train_batch_data["fixed_image"].to(device)
                # print(f'Shape of training inputs (moving image): {train_inputs.shape}')   #  (b,c,h,w)
                # print(f'Shape of training targets (fixed image): {train_targets.shape}')  #  (b,c,h,w) 

                # forward pass
                t0_forward = time.time()
                train_outputs, train_ddf = model(torch.cat((train_inputs, train_targets), dim=1))
                t1_forward = time.time()
                tot_t_forward = (t1_forward - t0_forward) * 1000
                # print(f'Time needed for forward pass: {tot_t_forward} ms')

                # loss computation
                if loss_name == 'MSE':
                    train_loss = MSELoss(reduction="mean")(train_outputs, train_targets)
                elif loss_name == 'SSIM':
                    train_loss = SSIMLoss(win_size=7, k1=0.01, k2=0.03, spatial_dims=2)(train_outputs, train_targets, data_range=data_range)
                elif loss_name == 'LNCC':
                    train_loss = LocalNormalizedCrossCorrelationLoss(spatial_dims=2, kernel_size=9, reduction='mean')(train_outputs, train_targets)
                elif loss_name == 'MSE-Diffusion':
                    train_loss = losses.CombinedImageDisplacementLoss(MSELoss(reduction="mean"),
                                                                      losses.DisplacementRegularizer2D(reg_type='gradient-l2'),
                                                                      image_weight=loss_weights[0], displacement_weight=loss_weights[1])(train_outputs, train_targets, train_ddf)
                elif loss_name == 'MSE-Bending':
                    train_loss = losses.CombinedImageDisplacementLoss(MSELoss(reduction="mean"),
                                                                      losses.DisplacementRegularizer2D(reg_type='bending'),
                                                                      image_weight=loss_weights[0], displacement_weight=loss_weights[1])(train_outputs, train_targets, train_ddf)
                elif loss_name == 'LNCC-Diffusion':
                    train_loss = losses.CombinedImageDisplacementLoss(LocalNormalizedCrossCorrelationLoss(spatial_dims=2, kernel_size=9, reduction='mean'),
                                                                      losses.DisplacementRegularizer2D(reg_type='gradient-l2'),
                                                                      image_weight=loss_weights[0], displacement_weight=loss_weights[1])(train_outputs, train_targets, train_ddf)
                elif loss_name == 'LNCC-Bending':
                    train_loss = losses.CombinedImageDisplacementLoss(LocalNormalizedCrossCorrelationLoss(spatial_dims=2, kernel_size=9, reduction='mean'),
                                                                      losses.DisplacementRegularizer2D(reg_type='bending'),
                                                                      image_weight=loss_weights[0], displacement_weight=loss_weights[1])(train_outputs, train_targets, train_ddf)
                else:
                    raise ValueError('Unkown loss_name specified!')
                
                # back propagate the errors and update the weights within batch    
                train_loss.backward()
                optimizer.step()
                losses_train.append(train_loss.item())
                # print(model.state_dict()['c2.0.weight'])
                # print(train_loss.item())
                
                if lr_scheduler is not None:
                    lr_scheduler.step()
                
                # compute metrics
                mse_metric_train(train_outputs, train_targets)
                ssim_metric_train(train_outputs, train_targets)


            # compute loss/metric for current epoch by averaging over batch losses and store results
            mean_loss_train_epoch = np.mean(losses_train) 
            mean_mse_metric_train_epoch = mse_metric_train.aggregate().item()
            mean_ssim_metric_train_epoch = ssim_metric_train.aggregate().item()
            
            if wandb_usage:
                wandb.log({f"train_{loss_name}_loss": mean_loss_train_epoch, 
                            "train_MSE_metric": mean_mse_metric_train_epoch,
                            "train_SSIM_metric": mean_ssim_metric_train_epoch,
                            "forward_pass_time (ms)": tot_t_forward})    
            

        model.eval()
        # unsupervised validation 
        if val_loader is not None:
            losses_val = []          
          
            print('Unsupervised validation...')            
            with torch.no_grad():
                for val_batch_data in val_loader:
                    val_inputs, val_targets = val_batch_data["moving_image"].to(device), val_batch_data["fixed_image"].to(device)
                    # print(f'Shape of validation inputs: {val_inputs.shape}')  # (1,1,224,224)
                    # print(f'Shape of validation targets: {val_targets.shape}')  # (1,1,224,224)
                    
                    # forward pass
                    val_outputs, val_ddf = model(torch.cat((val_inputs, val_targets),dim=1))

                    # loss computation
                    if loss_name == 'MSE':
                        val_loss = MSELoss(reduction="mean")(val_outputs, val_targets)
                    elif loss_name == 'SSIM':
                        val_loss = SSIMLoss(win_size=7, k1=0.01, k2=0.03, spatial_dims=2)(val_outputs, val_targets, data_range=data_range)
                    elif loss_name == 'LNCC':
                        val_loss = LocalNormalizedCrossCorrelationLoss(spatial_dims=2, kernel_size=9, reduction='mean')(val_outputs, val_targets)
                    elif loss_name == 'MSE-Diffusion':
                        val_loss = losses.CombinedImageDisplacementLoss(MSELoss(reduction="mean"),
                                                                        losses.DisplacementRegularizer2D(reg_type='gradient-l2'),
                                                                        image_weight=loss_weights[0], displacement_weight=loss_weights[1])(val_outputs, val_targets, val_ddf)
                    elif loss_name == 'MSE-Bending':
                        val_loss = losses.CombinedImageDisplacementLoss(MSELoss(reduction="mean"),
                                                                        losses.DisplacementRegularizer2D(reg_type='bending'),
                                                                        image_weight=loss_weights[0], displacement_weight=loss_weights[1])(val_outputs, val_targets, val_ddf) 
                    elif loss_name == 'LNCC-Diffusion':
                        val_loss = losses.CombinedImageDisplacementLoss(LocalNormalizedCrossCorrelationLoss(spatial_dims=2, kernel_size=9, reduction='mean'),
                                                                        losses.DisplacementRegularizer2D(reg_type='gradient-l2'),
                                                                        image_weight=loss_weights[0], displacement_weight=loss_weights[1])(val_outputs, val_targets, val_ddf)
                    elif loss_name == 'LNCC-Bending':
                        val_loss = losses.CombinedImageDisplacementLoss(LocalNormalizedCrossCorrelationLoss(spatial_dims=2, kernel_size=9, reduction='mean'),
                                                                        losses.DisplacementRegularizer2D(reg_type='bending'),
                                                                        image_weight=loss_weights[0], displacement_weight=loss_weights[1])(val_outputs, val_targets, val_ddf)  
                    else:
                        raise ValueError('Unkown loss_name specified!')
        
                    losses_val.append(val_loss.item())
                    
                    # compute metric
                    mse_metric_val(val_outputs, val_targets)
                    ssim_metric_val(val_outputs, val_targets)

                    
                # compute loss for current epoch by averaging over batch losses and store results
                mean_loss_val_epoch = np.mean(losses_val) 
                mean_mse_metric_val_epoch = mse_metric_val.aggregate().item()
                mean_ssim_metric_val_epoch = ssim_metric_val.aggregate().item()

            if wandb_usage:
                wandb.log({f"val_{loss_name}_loss": mean_loss_val_epoch, 
                            "val_MSE_metric": mean_mse_metric_val_epoch,
                            "val_SSIM_metric": mean_ssim_metric_val_epoch})   
            
            # if there is no supervised val loader, use loss to save best model/early stop the optimization
            if val_loader_supervised is None:        
                # save model if validation loss improves
                if mean_loss_val_epoch < best_val_value:
                    best_epoch = epoch
                    best_val_value = mean_loss_val_epoch
                    if path_saving is not None:
                        torch.save(model.state_dict(), os.path.join(
                            path_saving, f'best_model_epoch_{best_epoch:03d}_' + \
                            f'val_loss_{best_val_value:.6f}.pth'))
                        torch.save(optimizer.state_dict(), os.path.join(
                            path_saving, f'best_optim_epoch_{best_epoch:03d}_' + \
                            f'val_loss_{best_val_value:.6f}.pt'))
                        print('...saved model/optimizer based on new best validation loss.') 
                        
                if epoch % 1 == 0:
                    print(f'Train {loss_name} loss: {mean_loss_train_epoch} - '
                        f'Val {loss_name} loss: {mean_loss_val_epoch} - '
                        f'Best val loss: {best_val_value} - \n'      
                        f'Train MSE metric: {mean_mse_metric_train_epoch} - '
                        f'Val MSE metric: {mean_mse_metric_val_epoch} - \n'
                        f'Train SSIM metric: {mean_ssim_metric_train_epoch} - '
                        f'Val SSIM metric: {mean_ssim_metric_val_epoch}') 

                # stop the optimization if the loss didn't decrease after early_stopping nr of epochs
                if early_stopping_patience is not None:
                    if (epoch - best_epoch) > early_stopping_patience:
                        print('Early stopping the optimization!')
                        break
        
        # supervised validation
        if val_loader_supervised is not None:
            losses_sup = []          
            
            print('Supervised validation...')
            with torch.no_grad():
                for val_batch_data_supervised in val_loader_supervised:
                    inputs, targets, \
                        inputs_seg, targets_seg = val_batch_data_supervised["moving_image"].to(device), val_batch_data_supervised["fixed_image"].to(device), \
                                                            val_batch_data_supervised["moving_seg"].to(device), val_batch_data_supervised["fixed_seg"].to(device)
                    # print(f'Shape of inference inputs: {inputs.shape}')  # (b,1,224,224)
                    # print(f'Shape of inference outputs: {outputs.shape}')  # (b,1,224,224)
                    
                    # forward pass
                    outputs, outputs_seg, ddf = model(moving_img=torch.cat((inputs, targets),dim=1), moving_seg=inputs_seg, eval_warping=True)
                    
                    # loss computation
                    if loss_name == 'MSE':
                        loss = MSELoss(reduction="mean")(outputs, targets)
                    elif loss_name == 'SSIM':
                        loss = SSIMLoss(win_size=7, k1=0.01, k2=0.03, spatial_dims=2)(outputs, targets, data_range=data_range)
                    elif loss_name == 'LNCC':
                        loss = LocalNormalizedCrossCorrelationLoss(spatial_dims=2, kernel_size=9, reduction='mean')(outputs, targets)
                    elif loss_name == 'MSE-Diffusion':
                        loss = losses.CombinedImageDisplacementLoss(MSELoss(reduction="mean"),
                                                                        losses.DisplacementRegularizer2D(reg_type='gradient-l2'),
                                                                        image_weight=loss_weights[0], displacement_weight=loss_weights[1])(outputs, targets, ddf)
                    elif loss_name == 'MSE-Bending':
                        loss = losses.CombinedImageDisplacementLoss(MSELoss(reduction="mean"),
                                                                        losses.DisplacementRegularizer2D(reg_type='bending'),
                                                                        image_weight=loss_weights[0], displacement_weight=loss_weights[1])(outputs, targets, ddf) 
                    elif loss_name == 'LNCC-Diffusion':
                        loss = losses.CombinedImageDisplacementLoss(LocalNormalizedCrossCorrelationLoss(spatial_dims=2, kernel_size=9, reduction='mean'),
                                                                        losses.DisplacementRegularizer2D(reg_type='gradient-l2'),
                                                                        image_weight=loss_weights[0], displacement_weight=loss_weights[1])(outputs, targets, ddf)
                    elif loss_name == 'LNCC-Bending':
                        loss = losses.CombinedImageDisplacementLoss(LocalNormalizedCrossCorrelationLoss(spatial_dims=2, kernel_size=9, reduction='mean'),
                                                                        losses.DisplacementRegularizer2D(reg_type='bending'),
                                                                        image_weight=loss_weights[0], displacement_weight=loss_weights[1])(outputs, targets, ddf)  
                    else:
                        raise ValueError('Unkown loss_name specified!')
        
                    losses_sup.append(loss.item())
                    
                    # compute metric
                    dice_metric(outputs_seg, targets_seg)
                    hd95_metric(outputs_seg, targets_seg)
                    hd50_metric(outputs_seg, targets_seg)
                    negj_metric(ddf)
                
                
                # print(torch.sum(targets_seg[-2,...]))  
                # print(torch.sum(outputs_seg[-2,...]))  
                # compute metric by averaging over batchses and store results
                mean_loss_epoch = np.mean(losses_sup) 
                mean_dice_metric_epoch = round(dice_metric.aggregate().item(), 2)
                mean_hd95_metric_epoch = round(hd95_metric.aggregate().item(), 2)
                mean_hd50_metric_epoch = round(hd50_metric.aggregate().item(), 2)
                mean_negj_metric_epoch = round(negj_metric.aggregate().item(), 2)

            if wandb_usage:
                wandb.log({f"val_supervised_{loss_name}_loss": mean_loss_epoch, 
                            "val_supervised_DSC_metric": mean_dice_metric_epoch,
                            "val_supervised_HD95_metric": mean_hd95_metric_epoch,
                            "val_supervised_HD50_metric": mean_hd50_metric_epoch,
                            "val_supervised_NegJ_metric": mean_negj_metric_epoch})   
                    
                                        
            # save model if supervised validation HD95+HD50 improves
            if (mean_hd95_metric_epoch + mean_hd50_metric_epoch)  < best_val_value:
                best_epoch = epoch
                best_val_value = mean_hd95_metric_epoch + mean_hd50_metric_epoch
                if path_saving is not None:
                    torch.save(model.state_dict(), os.path.join(
                        path_saving, f'best_model_epoch_{best_epoch:03d}_' + \
                        f'val_mtrc_{best_val_value:.6f}.pth'))
                    torch.save(optimizer.state_dict(), os.path.join(
                        path_saving, f'best_optim_epoch_{best_epoch:03d}_' + \
                        f'val_mtrc_{best_val_value:.6f}.pt'))
                    print('...saved model/optimizer based on new best validation metric.') 
                    
            if epoch % 1 == 0:
                print(f'Train {loss_name} loss: {mean_loss_train_epoch} - '
                    # f'Val {loss_name} loss: {mean_loss_val_epoch} - '
                    f'Val supervised {loss_name} loss: {mean_loss_epoch} - \n'
                    f'Val supervised NegJ metric: {mean_negj_metric_epoch} - \n'
                    f'Val supervised DSC metric: {mean_dice_metric_epoch} - \n'
                    f'Val supervised HD50 metric: {mean_hd50_metric_epoch} - \n'
                    f'Val supervised HD95 metric: {mean_hd95_metric_epoch} - \n'
                    f'Current supervised val combined metric: {mean_hd50_metric_epoch + mean_hd95_metric_epoch} - '
                    f'Best supervised val combined metric: {best_val_value}') 

            # stop the optimization if the loss didn't decrease after early_stopping nr of epochs
            if early_stopping_patience is not None:
                if (epoch - best_epoch) > early_stopping_patience:
                    print('Early stopping the optimization!')
                    break
        
        # no validation set                                               
        else:
            if epoch % 1 == 0:
                print(f'Train {loss_name} loss: {mean_loss_train_epoch} - '
                    f'Train MSE metric: {mean_mse_metric_train_epoch} - '
                    f'Train SSIM metric: {mean_ssim_metric_train_epoch}')
                    
    
    if plot:
        print('Plotting last validation results...')
        if val_loader_supervised is None:
            plotting.plot_moving_fixed_and_outputs(moving=val_inputs, fixed=val_targets, 
                                                output=val_outputs, ddf=val_ddf, 
                                                sample_nr=0, path_saving=path_saving)
        else: 
            plotting.plot_moving_fixed_and_outputs(moving=inputs, fixed=targets, 
                                                    output=outputs, ddf=ddf, 
                                                    sample_nr=-1, path_saving=path_saving)
            
            plotting.plot_moving_fixed_and_outputs_seg(moving=inputs_seg, fixed=targets_seg, fixed_img=targets,
                                                        output=outputs_seg,
                                                        sample_nr=-1, path_saving=path_saving)
    
      
    t1 = time.time()
    tot_t = (t1 - t0) / 60
    print(f'\n------------ Total time needed for optimization: {tot_t} min --------- ') 
    
    # clean up
    gc.collect()
    torch.cuda.empty_cache() 
            


def train_val_model_supervised(model, train_loader, optimizer, loss_name, loss_weights, epoch_nr, device,
                                lr_scheduler=None, early_stopping_patience=None, epoch_start=0, val_loader=None,
                                path_saving=None, wandb_usage=False, plot=False):
    """Function to train and validate a Pytorch model in a supervised fashion.

    Args:
        model (Pytorch model): Network architecture
        train_loader (Pytorch laoder): pytorch data loader with training input and output
        loss_name (str): name of loss function, eg 'MSE'
        loss_weights (list): list with weights of combined loss if used, e.g. [1, 1] where 0th element is for image and 1st for displacmements
        optimizer (Pytorch optimizer): optimizer used to update network's weights
        lr_scheduler (Pytorch scheduler): scheduler for learning rate
        epoch_nr (int): maximum number of training epochs
        device (Pytorch device): cuda device
        early_stopping_patience (int, optional): whether to stop the optimization if the loss does
                                not get better after early_stopping nr of epochs. Defaults to None.
        epoch_start: starting epoch for loop
        val_loader (Pytorch laoder): pytorch data loader with validation input and output
        path_saving (str): path to folder where models are saved. Defaults to None.
        plot (str): plot some results
    """
    
    # set data range for structural similiraty
    data_range = torch.tensor([1.0], device=device)
        
    # initialize metrics
    mse_metric_train = MSEMetric(reduction="mean", get_not_nans=False)
    mse_metric_val = MSEMetric(reduction="mean", get_not_nans=False)
    ssim_metric_train = SSIMMetric(data_range=data_range, win_size=7, k1=0.01, k2=0.03, spatial_dims=2, reduction="mean", get_not_nans=False)
    ssim_metric_val = SSIMMetric(data_range=data_range, win_size=7, k1=0.01, k2=0.03, spatial_dims=2, reduction="mean", get_not_nans=False)
    dice_metric_train = DiceMetric(include_background=True, reduction="mean", get_not_nans=False, ignore_empty=True)
    dice_metric_val = DiceMetric(include_background=True, reduction="mean", get_not_nans=False, ignore_empty=True)
    hd95_metric_train = HausdorffDistanceMetric(include_background=False, distance_metric='euclidean', percentile=95, 
                                                    directed=False, reduction="mean", get_not_nans=False)
    hd95_metric_val = HausdorffDistanceMetric(include_background=False, distance_metric='euclidean', percentile=95, 
                                                    directed=False, reduction="mean", get_not_nans=False)
    hd50_metric_train = HausdorffDistanceMetric(include_background=False, distance_metric='euclidean', percentile=50, 
                                                    directed=False, reduction="mean", get_not_nans=False)
    hd50_metric_val = HausdorffDistanceMetric(include_background=False, distance_metric='euclidean', percentile=50, 
                                                    directed=False, reduction="mean", get_not_nans=False)
    negj_metric_train = metrics.NegativeJacobianMetric(reduction="mean", get_not_nans=False)
    negj_metric_val = metrics.NegativeJacobianMetric(reduction="mean", get_not_nans=False)

    best_val_value = 1000000  # to be sure loss decreases

    t0 = time.time()
    
    # loop over all epochs
    for epoch in range(epoch_start, epoch_start + epoch_nr):
        model.train()
        losses_train = [] 

        # loop over all batches of data
        with tqdm.tqdm(train_loader, unit="batch", initial=0) as tepoch:
            for train_batch_data in tepoch:
                tepoch.set_description(f"Epoch {epoch}/{epoch_start + epoch_nr - 1}")
                if wandb_usage:
                    wandb.log({"lr": optimizer.param_groups[0]['lr']})  
                
                # clear stored gradients
                optimizer.zero_grad()
                
                # get batch of data
                train_inputs, train_targets, \
                    train_inputs_seg, train_targets_seg = train_batch_data["moving_image"].to(device), train_batch_data["fixed_image"].to(device), \
                                                        train_batch_data["moving_seg"].to(device), train_batch_data["fixed_seg"].to(device)                    
                # print(f'Shape of validation inputs: {train_inputs.shape}')  # (1,1,224,224)
                # print(f'Shape of validation targets: {train_targets.shape}')  # (1,1,224,224)
                
                # forward pass
                t0_forward = time.time()
                train_outputs, train_outputs_seg, train_ddf = model(torch.cat((train_inputs, train_targets), dim=1), moving_seg=train_inputs_seg)
                t1_forward = time.time()
                tot_t_forward = (t1_forward - t0_forward) * 1000
                # print(f'Time needed for forward pass: {tot_t_forward} ms')

                # loss computation
                if loss_name == 'MSE':
                    train_loss = MSELoss(reduction="mean")(train_outputs, train_targets)
                elif loss_name == 'SSIM':
                    train_loss = SSIMLoss(win_size=7, k1=0.01, k2=0.03, spatial_dims=2)(train_outputs, train_targets, data_range=data_range)
                elif loss_name == 'LNCC':
                    train_loss = LocalNormalizedCrossCorrelationLoss(spatial_dims=2, kernel_size=9, reduction='mean')(train_outputs, train_targets)
                elif loss_name == 'Dice':
                    train_loss = DiceLoss(include_background=True, sigmoid=False, reduction="mean")(train_outputs_seg, train_targets_seg)
                elif loss_name == 'MSE-Diffusion':
                    train_loss = losses.CombinedImageDisplacementLoss(MSELoss(reduction="mean"),
                                                                      losses.DisplacementRegularizer2D(reg_type='gradient-l2'),
                                                                      image_weight=loss_weights[0], displacement_weight=loss_weights[1])(train_outputs, train_targets, train_ddf)
                elif loss_name == 'MSE-Bending':
                    train_loss = losses.CombinedImageDisplacementLoss(MSELoss(reduction="mean"),
                                                                      losses.DisplacementRegularizer2D(reg_type='bending'),
                                                                      image_weight=loss_weights[0], displacement_weight=loss_weights[1])(train_outputs, train_targets, train_ddf)
                elif loss_name == 'LNCC-Diffusion':
                    train_loss = losses.CombinedImageDisplacementLoss(LocalNormalizedCrossCorrelationLoss(spatial_dims=2, kernel_size=9, reduction='mean'),
                                                                      losses.DisplacementRegularizer2D(reg_type='gradient-l2'),
                                                                      image_weight=loss_weights[0], displacement_weight=loss_weights[1])(train_outputs, train_targets, train_ddf)
                elif loss_name == 'LNCC-Bending':
                    train_loss = losses.CombinedImageDisplacementLoss(LocalNormalizedCrossCorrelationLoss(spatial_dims=2, kernel_size=9, reduction='mean'),
                                                                      losses.DisplacementRegularizer2D(reg_type='bending'),
                                                                      image_weight=loss_weights[0], displacement_weight=loss_weights[1])(train_outputs, train_targets, train_ddf)
                elif loss_name == 'Dice-Diffusion':
                    train_loss = losses.CombinedImageDisplacementLoss(DiceLoss(include_background=True, sigmoid=False, reduction="mean"),
                                                                      losses.DisplacementRegularizer2D(reg_type='gradient-l2'),
                                                                      image_weight=loss_weights[0], displacement_weight=loss_weights[1])(train_outputs_seg, train_targets_seg, train_ddf)
                elif loss_name == 'Dice-Bending':
                    train_loss = losses.CombinedImageDisplacementLoss(DiceLoss(include_background=True, sigmoid=False, reduction="mean"),
                                                                      losses.DisplacementRegularizer2D(reg_type='bending'),
                                                                      image_weight=loss_weights[0], displacement_weight=loss_weights[1])(train_outputs_seg, train_targets_seg, train_ddf)
                elif loss_name == 'MSE-Dice-Diffusion':
                    train_loss = losses.CombinedImageSegmentationDisplacementLoss(MSELoss(reduction="mean"),
                                                                      DiceLoss(include_background=True, sigmoid=False, reduction="mean"),
                                                                      losses.DisplacementRegularizer2D(reg_type='gradient-l2'),
                                                                      image_weight=loss_weights[0], 
                                                                      segmentation_weight=loss_weights[1], 
                                                                      displacement_weight=loss_weights[2])(train_outputs, train_targets, train_outputs_seg, train_targets_seg, train_ddf)
                elif loss_name == 'MSE-Dice-Bending':
                    train_loss = losses.CombinedImageSegmentationDisplacementLoss(MSELoss(reduction="mean"),
                                                                      DiceLoss(include_background=True, sigmoid=False, reduction="mean"),
                                                                      losses.DisplacementRegularizer2D(reg_type='bending'),
                                                                      image_weight=loss_weights[0], 
                                                                      segmentation_weight=loss_weights[1], 
                                                                      displacement_weight=loss_weights[2])(train_outputs, train_targets, train_outputs_seg, train_targets_seg, train_ddf)
                else:
                    raise ValueError('Unkown loss_name specified!')
                
                # back propagate the errors and update the weights within batch    
                train_loss.backward()
                optimizer.step()
                losses_train.append(train_loss.item())
                # print(model.state_dict()['conv_enc1.0.bias'])
                # print(train_loss.item())
                
                if lr_scheduler is not None:
                    lr_scheduler.step()
                
                # compute metric
                mse_metric_train(train_outputs, train_targets)
                ssim_metric_train(train_outputs, train_targets)
                dice_metric_train(train_outputs_seg, train_targets_seg)
                hd95_metric_train(train_outputs_seg, train_targets_seg)
                hd50_metric_train(train_outputs_seg, train_targets_seg)
                negj_metric_train(train_ddf)


            # compute loss/metric for current epoch by averaging over batch losses and log results
            mean_loss_train_epoch = np.mean(losses_train) 
            mean_mse_metric_train_epoch = mse_metric_train.aggregate().item()
            mean_ssim_metric_train_epoch = ssim_metric_train.aggregate().item()
            mean_dice_metric_train_epoch = round(dice_metric_train.aggregate().item(), 2)
            mean_hd95_metric_train_epoch = round(hd95_metric_train.aggregate().item(), 2)
            mean_hd50_metric_train_epoch = round(hd50_metric_train.aggregate().item(), 2)
            mean_negj_metric_train_epoch = round(negj_metric_train.aggregate().item(), 2)
            
            if wandb_usage:                       
                wandb.log({f"train_{loss_name}_loss": mean_loss_train_epoch, 
                            "train_MSE_metric": mean_mse_metric_train_epoch,
                            "train_SSIM_metric": mean_ssim_metric_train_epoch,
                            "train_DSC_metric": mean_dice_metric_train_epoch,
                            "train_HD95_metric": mean_hd95_metric_train_epoch,
                            "train_HD50_metric": mean_hd50_metric_train_epoch,
                            "train_NegJ_metric": mean_negj_metric_train_epoch,
                            "forward_pass_time (ms)": tot_t_forward})    
            

        if val_loader is not None:
            losses_val = []          
            
            with torch.no_grad():
                model.eval()
                for val_batch_data in val_loader:
                    val_inputs, val_targets, \
                        val_inputs_seg, val_targets_seg = val_batch_data["moving_image"].to(device), val_batch_data["fixed_image"].to(device), \
                                                            val_batch_data["moving_seg"].to(device), val_batch_data["fixed_seg"].to(device)                    
                    # print(f'Shape of validation inputs: {val_inputs.shape}')  # (1,1,224,224)
                    # print(f'Shape of validation targets: {val_targets.shape}')  # (1,1,224,224)
                    
                    # forward pass
                    val_outputs, val_outputs_seg, val_ddf = model(torch.cat((val_inputs, val_targets),dim=1), moving_seg=val_inputs_seg, eval_warping=True)
                    
                    # loss computation
                    if loss_name == 'MSE':
                        val_loss = MSELoss(reduction="mean")(val_outputs, val_targets)
                    elif loss_name == 'SSIM':
                        val_loss = SSIMLoss(win_size=7, k1=0.01, k2=0.03, spatial_dims=2)(val_outputs, val_targets, data_range=data_range)
                    elif loss_name == 'LNCC':
                        val_loss = LocalNormalizedCrossCorrelationLoss(spatial_dims=2, kernel_size=9, reduction='mean')(val_outputs, val_targets)
                    elif loss_name == 'Dice':
                        val_loss = DiceLoss(include_background=True, sigmoid=False, reduction="mean")(val_outputs_seg, val_targets_seg)
                    elif loss_name == 'MSE-Diffusion':
                        val_loss = losses.CombinedImageDisplacementLoss(MSELoss(reduction="mean"),
                                                                        losses.DisplacementRegularizer2D(reg_type='gradient-l2'),
                                                                        image_weight=loss_weights[0], displacement_weight=loss_weights[1])(val_outputs, val_targets, val_ddf)
                    elif loss_name == 'MSE-Bending':
                        val_loss = losses.CombinedImageDisplacementLoss(MSELoss(reduction="mean"),
                                                                        losses.DisplacementRegularizer2D(reg_type='bending'),
                                                                        image_weight=loss_weights[0], displacement_weight=loss_weights[1])(val_outputs, val_targets, val_ddf) 
                    elif loss_name == 'LNCC-Diffusion':
                        val_loss = losses.CombinedImageDisplacementLoss(LocalNormalizedCrossCorrelationLoss(spatial_dims=2, kernel_size=9, reduction='mean'),
                                                                        losses.DisplacementRegularizer2D(reg_type='gradient-l2'),
                                                                        image_weight=loss_weights[0], displacement_weight=loss_weights[1])(val_outputs, val_targets, val_ddf)
                    elif loss_name == 'LNCC-Bending':
                        val_loss = losses.CombinedImageDisplacementLoss(LocalNormalizedCrossCorrelationLoss(spatial_dims=2, kernel_size=9, reduction='mean'),
                                                                        losses.DisplacementRegularizer2D(reg_type='bending'),
                                                                        image_weight=loss_weights[0], displacement_weight=loss_weights[1])(val_outputs, val_targets, val_ddf)  
                    elif loss_name == 'Dice-Diffusion':
                        val_loss = losses.CombinedImageDisplacementLoss(DiceLoss(include_background=True, sigmoid=False, reduction="mean"),
                                                                        losses.DisplacementRegularizer2D(reg_type='gradient-l2'),
                                                                        image_weight=loss_weights[0], displacement_weight=loss_weights[1])(val_outputs_seg, val_targets_seg, val_ddf)
                    elif loss_name == 'Dice-Bending':
                        val_loss = losses.CombinedImageDisplacementLoss(DiceLoss(include_background=True, sigmoid=False, reduction="mean"),
                                                                        losses.DisplacementRegularizer2D(reg_type='bending'),
                                                                        image_weight=loss_weights[0], displacement_weight=loss_weights[1])(val_outputs_seg, val_targets_seg, val_ddf)
                    elif loss_name == 'MSE-Dice-Diffusion':
                        val_loss = losses.CombinedImageSegmentationDisplacementLoss(MSELoss(reduction="mean"),
                                                                        DiceLoss(include_background=True, sigmoid=False, reduction="mean"),
                                                                        losses.DisplacementRegularizer2D(reg_type='gradient-l2'),
                                                                        image_weight=loss_weights[0], 
                                                                        segmentation_weight=loss_weights[1], 
                                                                        displacement_weight=loss_weights[2])(val_outputs, val_targets, val_outputs_seg, val_targets_seg, val_ddf)
                    elif loss_name == 'MSE-Dice-Bending':
                        val_loss = losses.CombinedImageSegmentationDisplacementLoss(MSELoss(reduction="mean"),
                                                                        DiceLoss(include_background=True, sigmoid=False, reduction="mean"),
                                                                        losses.DisplacementRegularizer2D(reg_type='bending'),
                                                                        image_weight=loss_weights[0], 
                                                                        segmentation_weight=loss_weights[1], 
                                                                        displacement_weight=loss_weights[2])(val_outputs, val_targets, val_outputs_seg, val_targets_seg, val_ddf)
                    else:
                        raise ValueError('Unkown loss_name specified!')
        
                    losses_val.append(val_loss.item())
                    
                    # compute metric
                    mse_metric_val(val_outputs, val_targets)
                    ssim_metric_val(val_outputs, val_targets)
                    dice_metric_val(val_outputs_seg, val_targets_seg)
                    hd95_metric_val(val_outputs_seg, val_targets_seg)
                    hd50_metric_val(val_outputs_seg, val_targets_seg)
                    negj_metric_val(val_ddf)

                # compute loss/metric for current epoch by averaging over batch losses and log results
                mean_loss_val_epoch = np.mean(losses_val) 
                mean_mse_metric_val_epoch = mse_metric_val.aggregate().item()
                mean_ssim_metric_val_epoch = ssim_metric_val.aggregate().item()
                mean_dice_metric_val_epoch = round(dice_metric_val.aggregate().item(), 2)
                mean_hd95_metric_val_epoch = round(hd95_metric_val.aggregate().item(), 2)
                mean_hd50_metric_val_epoch = round(hd50_metric_val.aggregate().item(), 2)
                mean_negj_metric_val_epoch = round(negj_metric_val.aggregate().item(), 2)
                
                if wandb_usage:                    
                    wandb.log({f"val_supervised_{loss_name}_loss": mean_loss_val_epoch, 
                                "val_supervised_MSE_metric": mean_mse_metric_val_epoch,
                                "val_supervised_SSIM_metric": mean_ssim_metric_val_epoch,
                                "val_supervised_DSC_metric": mean_dice_metric_val_epoch,
                                "val_supervised_HD95_metric": mean_hd95_metric_val_epoch,
                                "val_supervised_HD50_metric": mean_hd50_metric_val_epoch,
                                "val_supervised_NegJ_metric": mean_negj_metric_val_epoch,
                                "forward_pass_time (ms)": tot_t_forward})   
                 
                                       
            # save model if supervised validation HD95 improves
            if (mean_hd95_metric_val_epoch + mean_hd50_metric_val_epoch) < best_val_value:
                best_epoch = epoch
                best_val_value = mean_hd95_metric_val_epoch + mean_hd50_metric_val_epoch
                if path_saving is not None:
                    torch.save(model.state_dict(), os.path.join(
                        path_saving, f'best_model_epoch_{best_epoch:03d}_' + \
                        f'val_mtrc_{best_val_value:.6f}.pth'))
                    torch.save(optimizer.state_dict(), os.path.join(
                        path_saving, f'best_optim_epoch_{best_epoch:03d}_' + \
                        f'val_mtrc_{best_val_value:.6f}.pt'))
                    print('...saved model/optimizer based on new best validation metric.') 
                    
            if epoch % 1 == 0:
                print(f'Train {loss_name} loss: {mean_loss_train_epoch} - '
                        f'Val {loss_name} loss: {mean_loss_val_epoch} - '
                        f'Train MSE metric: {mean_mse_metric_train_epoch} - '
                        f'Val MSE metric: {mean_mse_metric_val_epoch} - '
                        f'Train SSIM metric: {mean_ssim_metric_train_epoch}'
                        f'Val SSIM metric: {mean_ssim_metric_val_epoch}'
                        f'Train NegJ metric: {mean_negj_metric_train_epoch} - \n'
                        f'Val NegJ metric: {mean_negj_metric_val_epoch} - \n'
                        f'Train DSC metric: {mean_dice_metric_train_epoch} - \n'
                        f'Val DSC metric: {mean_dice_metric_val_epoch} - \n'
                        f'Train HD50 metric: {mean_hd50_metric_train_epoch} - \n'
                        f'Val HD50 metric: {mean_hd50_metric_val_epoch} - \n'
                        f'Train HD95 metric: {mean_hd95_metric_train_epoch} - '
                        f'Val HD95 metric: {mean_hd95_metric_val_epoch} - \n'
                        f'Current val combined metric: {mean_hd50_metric_val_epoch + mean_hd95_metric_val_epoch} - '
                        f'Best val combined metric: {best_val_value}') 

            # stop the optimization if the loss didn't decrease after early_stopping nr of epochs
            if early_stopping_patience is not None:
                if (epoch - best_epoch) > early_stopping_patience:
                    print('Early stopping the optimization!')
                    break
        
        # no validation set                                               
        else:
            if epoch % 1 == 0:
                print(f'Train {loss_name} loss: {mean_loss_train_epoch} - '
                        f'Train MSE metric: {mean_mse_metric_train_epoch} - '
                        f'Train SSIM metric: {mean_ssim_metric_train_epoch}'
                        f'Train NegJ metric: {mean_negj_metric_train_epoch} - \n'
                        f'Train DSC metric: {mean_dice_metric_train_epoch} - \n'
                        f'Train HD50 metric: {mean_hd50_metric_train_epoch} - \n'
                        f'Train HD95 metric: {mean_hd95_metric_train_epoch}')
                    
    
    if plot:
        print('Plotting last validation results...')
        plotting.plot_moving_fixed_and_outputs(moving=val_inputs, fixed=val_targets, 
                                                output=val_outputs, ddf=val_ddf, 
                                                sample_nr=0, path_saving=path_saving)
        
        plotting.plot_moving_fixed_and_outputs_seg(moving=val_inputs_seg, fixed=val_targets_seg, fixed_img=val_targets,
                                                    output=val_outputs_seg,
                                                    sample_nr=-1, path_saving=path_saving)
    
      
    t1 = time.time()
    tot_t = (t1 - t0) / 60
    print(f'\n------------ Total time needed for optimization: {tot_t} min --------- ') 
    
    # clean up
    gc.collect()
    torch.cuda.empty_cache() 
    

def train_val_model_ps(model, train_loader, optimizer, loss_name, loss_weights, epoch_nr, device,
                                lr_scheduler=None, epoch_start=0, infer_loader=None,
                                path_saving=None, wandb_usage=False, plot=False):
    """Function to train and validate a Pytorch model in a patient specific fashion.

    Args:
        model (Pytorch model): Network architecture
        train_loader (Pytorch laoder): pytorch data loader with training input and output
        loss_name (str): name of loss function, eg 'MSE'
        loss_weights (list): list with weights of combined loss if used, e.g. [1, 1] where 0th element is for image and 1st for displacmements
        optimizer (Pytorch optimizer): optimizer used to update network's weights
        lr_scheduler (Pytorch scheduler): scheduler for learning rate
        epoch_nr (int): maximum number of training epochs
        device (Pytorch device): cuda device
        early_stopping_patience (int, optional): whether to stop the optimization if the loss does
                                not get better after early_stopping nr of epochs. Defaults to None.
        epoch_start: starting epoch for loop
        infer_loader (Pytorch laoder): pytorch data loader with validation/testing input and output
        path_saving (str): path to folder where models are saved. Defaults to None.
        plot (str): plot some results
    """
    
    # set data range for structural similiraty
    data_range = torch.tensor([1.0], device=device)
        
    # initialize metrics
    mse_metric_train = MSEMetric(reduction="mean", get_not_nans=False)
    mse_metric_infer = MSEMetric(reduction="mean", get_not_nans=False)
    ssim_metric_train = SSIMMetric(data_range=data_range, win_size=7, k1=0.01, k2=0.03, spatial_dims=2, reduction="mean", get_not_nans=False)
    ssim_metric_infer = SSIMMetric(data_range=data_range, win_size=7, k1=0.01, k2=0.03, spatial_dims=2, reduction="mean", get_not_nans=False)
    dice_metric_train = DiceMetric(include_background=True, reduction="mean", get_not_nans=False, ignore_empty=True)
    dice_metric_infer = DiceMetric(include_background=True, reduction="mean", get_not_nans=False, ignore_empty=True)
    hd95_metric_train = HausdorffDistanceMetric(include_background=False, distance_metric='euclidean', percentile=95, 
                                                    directed=False, reduction="mean", get_not_nans=False)
    hd95_metric_infer = HausdorffDistanceMetric(include_background=False, distance_metric='euclidean', percentile=95, 
                                                    directed=False, reduction="mean", get_not_nans=False)
    hd50_metric_train = HausdorffDistanceMetric(include_background=False, distance_metric='euclidean', percentile=50, 
                                                    directed=False, reduction="mean", get_not_nans=False)
    hd50_metric_infer = HausdorffDistanceMetric(include_background=False, distance_metric='euclidean', percentile=50, 
                                                    directed=False, reduction="mean", get_not_nans=False)
    negj_metric_train = metrics.NegativeJacobianMetric(reduction="mean", get_not_nans=False)
    negj_metric_infer = metrics.NegativeJacobianMetric(reduction="mean", get_not_nans=False)


    t0 = time.time()
    
    # loop over all epochs
    for epoch in range(epoch_start, epoch_start + epoch_nr):
        model.train()
        losses_train = [] 

        # loop over all batches of data
        with tqdm.tqdm(train_loader, unit="batch", initial=0) as tepoch:
            for train_batch_data in tepoch:
                tepoch.set_description(f"Epoch {epoch}/{epoch_start + epoch_nr - 1}")
                if wandb_usage:
                    wandb.log({"lr": optimizer.param_groups[0]['lr']})  
                
                # clear stored gradients
                optimizer.zero_grad()
                
                # get batch of data
                train_inputs, train_targets, \
                    train_inputs_seg, train_targets_seg = train_batch_data["moving_image"].to(device), train_batch_data["fixed_image"].to(device), \
                                                        train_batch_data["moving_seg"].to(device), train_batch_data["fixed_seg"].to(device)                    
                # print(f'Shape of validation inputs: {train_inputs.shape}')  # (1,1,224,224)
                # print(f'Shape of validation targets: {train_targets.shape}')  # (1,1,224,224)
                
                # forward pass
                # t0_forward = time.time()
                train_outputs, train_outputs_seg, train_ddf = model(torch.cat((train_inputs, train_targets), dim=1), moving_seg=train_inputs_seg)
                # t1_forward = time.time()
                # tot_t_forward = (t1_forward - t0_forward) * 1000
                # print(f'Time needed for forward pass: {tot_t_forward} ms')

                # loss computation
                if loss_name == 'MSE':
                    train_loss = MSELoss(reduction="mean")(train_outputs, train_targets)
                elif loss_name == 'SSIM':
                    train_loss = SSIMLoss(win_size=7, k1=0.01, k2=0.03, spatial_dims=2)(train_outputs, train_targets, data_range=data_range)
                elif loss_name == 'LNCC':
                    train_loss = LocalNormalizedCrossCorrelationLoss(spatial_dims=2, kernel_size=9, reduction='mean')(train_outputs, train_targets)
                elif loss_name == 'Dice':
                    train_loss = DiceLoss(include_background=True, sigmoid=False, reduction="mean")(train_outputs_seg, train_targets_seg)
                elif loss_name == 'MSE-Diffusion':
                    train_loss = losses.CombinedImageDisplacementLoss(MSELoss(reduction="mean"),
                                                                      losses.DisplacementRegularizer2D(reg_type='gradient-l2'),
                                                                      image_weight=loss_weights[0], displacement_weight=loss_weights[1])(train_outputs, train_targets, train_ddf)
                elif loss_name == 'MSE-Bending':
                    train_loss = losses.CombinedImageDisplacementLoss(MSELoss(reduction="mean"),
                                                                      losses.DisplacementRegularizer2D(reg_type='bending'),
                                                                      image_weight=loss_weights[0], displacement_weight=loss_weights[1])(train_outputs, train_targets, train_ddf)
                elif loss_name == 'LNCC-Diffusion':
                    train_loss = losses.CombinedImageDisplacementLoss(LocalNormalizedCrossCorrelationLoss(spatial_dims=2, kernel_size=9, reduction='mean'),
                                                                      losses.DisplacementRegularizer2D(reg_type='gradient-l2'),
                                                                      image_weight=loss_weights[0], displacement_weight=loss_weights[1])(train_outputs, train_targets, train_ddf)
                elif loss_name == 'LNCC-Bending':
                    train_loss = losses.CombinedImageDisplacementLoss(LocalNormalizedCrossCorrelationLoss(spatial_dims=2, kernel_size=9, reduction='mean'),
                                                                      losses.DisplacementRegularizer2D(reg_type='bending'),
                                                                      image_weight=loss_weights[0], displacement_weight=loss_weights[1])(train_outputs, train_targets, train_ddf)
                elif loss_name == 'Dice-Diffusion':
                    train_loss = losses.CombinedImageDisplacementLoss(DiceLoss(include_background=True, sigmoid=False, reduction="mean"),
                                                                      losses.DisplacementRegularizer2D(reg_type='gradient-l2'),
                                                                      image_weight=loss_weights[0], displacement_weight=loss_weights[1])(train_outputs_seg, train_targets_seg, train_ddf)
                elif loss_name == 'Dice-Bending':
                    train_loss = losses.CombinedImageDisplacementLoss(DiceLoss(include_background=True, sigmoid=False, reduction="mean"),
                                                                      losses.DisplacementRegularizer2D(reg_type='bending'),
                                                                      image_weight=loss_weights[0], displacement_weight=loss_weights[1])(train_outputs_seg, train_targets_seg, train_ddf)
                elif loss_name == 'MSE-Dice-Diffusion':
                    train_loss = losses.CombinedImageSegmentationDisplacementLoss(MSELoss(reduction="mean"),
                                                                      DiceLoss(include_background=True, sigmoid=False, reduction="mean"),
                                                                      losses.DisplacementRegularizer2D(reg_type='gradient-l2'),
                                                                      image_weight=loss_weights[0], 
                                                                      segmentation_weight=loss_weights[1], 
                                                                      displacement_weight=loss_weights[2])(train_outputs, train_targets, train_outputs_seg, train_targets_seg, train_ddf)
                elif loss_name == 'MSE-Dice-Bending':
                    train_loss = losses.CombinedImageSegmentationDisplacementLoss(MSELoss(reduction="mean"),
                                                                      DiceLoss(include_background=True, sigmoid=False, reduction="mean"),
                                                                      losses.DisplacementRegularizer2D(reg_type='bending'),
                                                                      image_weight=loss_weights[0], 
                                                                      segmentation_weight=loss_weights[1], 
                                                                      displacement_weight=loss_weights[2])(train_outputs, train_targets, train_outputs_seg, train_targets_seg, train_ddf)

                else:
                    raise ValueError('Unkown loss_name specified!')
                
                # back propagate the errors and update the weights within batch    
                train_loss.backward()
                optimizer.step()
                losses_train.append(train_loss.item())
                # print(model.state_dict()['conv_enc1.0.bias'])
                # print(train_loss.item())
                
                if lr_scheduler is not None:
                    lr_scheduler.step()
                
                # compute metric
                mse_metric_train(train_outputs, train_targets)
                ssim_metric_train(train_outputs, train_targets)
                dice_metric_train(train_outputs_seg, train_targets_seg)
                hd95_metric_train(train_outputs_seg, train_targets_seg)
                hd50_metric_train(train_outputs_seg, train_targets_seg)
                negj_metric_train(train_ddf)


            # compute loss/metric for current epoch by averaging over batch losses and log results
            mean_loss_train_epoch = np.mean(losses_train) 
            mean_mse_metric_train_epoch = mse_metric_train.aggregate().item()
            mean_ssim_metric_train_epoch = ssim_metric_train.aggregate().item()
            mean_dice_metric_train_epoch = round(dice_metric_train.aggregate().item(), 2)
            mean_hd95_metric_train_epoch = round(hd95_metric_train.aggregate().item(), 2)
            mean_hd50_metric_train_epoch = round(hd50_metric_train.aggregate().item(), 2)
            mean_negj_metric_train_epoch = round(negj_metric_train.aggregate().item(), 2)
            
            print(f'Train {loss_name} loss: {mean_loss_train_epoch} - '
                    f'Train MSE metric: {mean_mse_metric_train_epoch} - '
                    f'Train SSIM metric: {mean_ssim_metric_train_epoch}'
                    f'Train NegJ metric: {mean_negj_metric_train_epoch} - \n'
                    f'Train DSC metric: {mean_dice_metric_train_epoch} - \n'
                    f'Train HD50 metric: {mean_hd50_metric_train_epoch} - \n'
                    f'Train HD95 metric: {mean_hd95_metric_train_epoch}')
            
    t1 = time.time()
    tot_t = (t1 - t0) / 60
    print(f'\n------------ Total time needed for optimization only: {tot_t} min --------- ') 
    
    
    if infer_loader is not None:
        losses_infer = []          
        
        with torch.no_grad():
            model.eval()
            for infer_batch_data in infer_loader:
                infer_inputs, infer_targets, \
                    infer_inputs_seg, infer_targets_seg = infer_batch_data["moving_image"].to(device), infer_batch_data["fixed_image"].to(device), \
                                                        infer_batch_data["moving_seg"].to(device), infer_batch_data["fixed_seg"].to(device)                    
                # print(f'Shape of validation inputs: {val_inputs.shape}')  # (1,1,224,224)
                # print(f'Shape of validation targets: {val_targets.shape}')  # (1,1,224,224)
                
                # forward pass
                infer_outputs, infer_outputs_seg, infer_ddf = model(torch.cat((infer_inputs, infer_targets),dim=1), moving_seg=infer_inputs_seg, eval_warping=True)
                
                # loss computation
                if loss_name == 'MSE':
                    infer_loss = MSELoss(reduction="mean")(infer_outputs, infer_targets)
                elif loss_name == 'SSIM':
                    infer_loss = SSIMLoss(win_size=7, k1=0.01, k2=0.03, spatial_dims=2)(infer_outputs, infer_targets, data_range=data_range)
                elif loss_name == 'LNCC':
                    infer_loss = LocalNormalizedCrossCorrelationLoss(spatial_dims=2, kernel_size=9, reduction='mean')(infer_outputs, infer_targets)
                elif loss_name == 'Dice':
                    infer_loss = DiceLoss(include_background=True, sigmoid=False, reduction="mean")(infer_outputs_seg, infer_targets_seg)
                elif loss_name == 'MSE-Diffusion':
                    infer_loss = losses.CombinedImageDisplacementLoss(MSELoss(reduction="mean"),
                                                                    losses.DisplacementRegularizer2D(reg_type='gradient-l2'),
                                                                    image_weight=loss_weights[0], displacement_weight=loss_weights[1])(infer_outputs, infer_targets, infer_ddf)
                elif loss_name == 'MSE-Bending':
                    infer_loss = losses.CombinedImageDisplacementLoss(MSELoss(reduction="mean"),
                                                                    losses.DisplacementRegularizer2D(reg_type='bending'),
                                                                    image_weight=loss_weights[0], displacement_weight=loss_weights[1])(infer_outputs, infer_targets, infer_ddf) 
                elif loss_name == 'LNCC-Diffusion':
                    infer_loss = losses.CombinedImageDisplacementLoss(LocalNormalizedCrossCorrelationLoss(spatial_dims=2, kernel_size=9, reduction='mean'),
                                                                    losses.DisplacementRegularizer2D(reg_type='gradient-l2'),
                                                                    image_weight=loss_weights[0], displacement_weight=loss_weights[1])(infer_outputs, infer_targets, infer_ddf)
                elif loss_name == 'LNCC-Bending':
                    infer_loss = losses.CombinedImageDisplacementLoss(LocalNormalizedCrossCorrelationLoss(spatial_dims=2, kernel_size=9, reduction='mean'),
                                                                    losses.DisplacementRegularizer2D(reg_type='bending'),
                                                                    image_weight=loss_weights[0], displacement_weight=loss_weights[1])(infer_outputs, infer_targets, infer_ddf)  
                elif loss_name == 'Dice-Diffusion':
                    infer_loss = losses.CombinedImageDisplacementLoss(DiceLoss(include_background=True, sigmoid=False, reduction="mean"),
                                                                    losses.DisplacementRegularizer2D(reg_type='gradient-l2'),
                                                                    image_weight=loss_weights[0], displacement_weight=loss_weights[1])(infer_outputs_seg, infer_targets_seg, infer_ddf)
                elif loss_name == 'Dice-Bending':
                    infer_loss = losses.CombinedImageDisplacementLoss(DiceLoss(include_background=True, sigmoid=False, reduction="mean"),
                                                                    losses.DisplacementRegularizer2D(reg_type='bending'),
                                                                    image_weight=loss_weights[0], displacement_weight=loss_weights[1])(infer_outputs_seg, infer_targets_seg, infer_ddf)
                elif loss_name == 'MSE-Dice-Diffusion':
                    infer_loss = losses.CombinedImageSegmentationDisplacementLoss(MSELoss(reduction="mean"),
                                                                    DiceLoss(include_background=True, sigmoid=False, reduction="mean"),
                                                                    losses.DisplacementRegularizer2D(reg_type='gradient-l2'),
                                                                    image_weight=loss_weights[0], 
                                                                    segmentation_weight=loss_weights[1], 
                                                                    displacement_weight=loss_weights[2])(infer_outputs, infer_targets, infer_outputs_seg, infer_targets_seg, infer_ddf)
                elif loss_name == 'MSE-Dice-Bending':
                    infer_loss = losses.CombinedImageSegmentationDisplacementLoss(MSELoss(reduction="mean"),
                                                                    DiceLoss(include_background=True, sigmoid=False, reduction="mean"),
                                                                    losses.DisplacementRegularizer2D(reg_type='bending'),
                                                                    image_weight=loss_weights[0], 
                                                                    segmentation_weight=loss_weights[1], 
                                                                    displacement_weight=loss_weights[2])(infer_outputs, infer_targets, infer_outputs_seg, infer_targets_seg, infer_ddf)
                else:
                    raise ValueError('Unkown loss_name specified!')
    
                losses_infer.append(infer_loss.item())
                
                # compute metric
                mse_metric_infer(infer_outputs, infer_targets)
                ssim_metric_infer(infer_outputs, infer_targets)
                dice_metric_infer(infer_outputs_seg, infer_targets_seg)
                hd95_metric_infer(infer_outputs_seg, infer_targets_seg)
                hd50_metric_infer(infer_outputs_seg, infer_targets_seg)
                negj_metric_infer(infer_ddf)
            
            # compute metrics by averaging over batches and store results
            mean_mse_metric = round(np.nanmean(mse_metric_infer.get_buffer().detach().cpu()), 4)
            std_mse_metric = round(np.nanstd(mse_metric_infer.get_buffer().detach().cpu()), 4)
            mean_ssim_metric = round(np.nanmean(ssim_metric_infer.get_buffer().detach().cpu()), 4)
            std_ssim_metric = round(np.nanstd(ssim_metric_infer.get_buffer().detach().cpu()), 4)
            mean_dice_metric = round(np.nanmean(dice_metric_infer.get_buffer().detach().cpu()), 2)
            std_dice_metric = round(np.nanstd(dice_metric_infer.get_buffer().detach().cpu()), 2)
            mean_hd95_metric = round(np.nanmean(hd95_metric_infer.get_buffer().detach().cpu()), 2)
            std_hd95_metric = round(np.nanstd(hd95_metric_infer.get_buffer().detach().cpu()), 2)    
            mean_hd50_metric = round(np.nanmean(hd50_metric_infer.get_buffer().detach().cpu()), 2)
            std_hd50_metric = round(np.nanstd(hd50_metric_infer.get_buffer().detach().cpu()), 2)       
            mean_negj_metric = round(np.nanmean(negj_metric_infer.get_buffer().detach().cpu()), 2)
            std_negj_metric = round(np.nanstd(negj_metric_infer.get_buffer().detach().cpu()), 2)    
        
        
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
                        "NegJ (mean)": mean_negj_metric,
                        "NegJ (std)": std_negj_metric}
        
        print(f'Inference metrics: \n {eval_metrics_patient}')
        
        # save metrics for all frames for each patient in txt for later analysis
        np.savetxt(os.path.join(path_saving, 'mse_metric.txt'), mse_metric_infer.get_buffer().squeeze().detach().cpu().numpy())
        np.savetxt(os.path.join(path_saving, 'ssim_metric.txt'), ssim_metric_infer.get_buffer().squeeze().detach().cpu().numpy())
        np.savetxt(os.path.join(path_saving, 'dice_metric.txt'), dice_metric_infer.get_buffer().squeeze().detach().cpu().numpy())
        np.savetxt(os.path.join(path_saving, 'hd95_metric.txt'), hd95_metric_infer.get_buffer().squeeze().detach().cpu().numpy())
        np.savetxt(os.path.join(path_saving, 'hd50_metric.txt'), hd50_metric_infer.get_buffer().squeeze().detach().cpu().numpy())
        np.savetxt(os.path.join(path_saving, 'negj_metric.txt'), negj_metric_infer.get_buffer().squeeze().detach().cpu().numpy())
            
                
        # save model 
        best_epoch = epoch_nr + epoch_start
        best_infer_value = mean_hd95_metric + mean_hd50_metric
        torch.save(model.state_dict(), os.path.join(
            path_saving, f'best_model_epoch_{best_epoch:03d}_' + \
            f'inf_mtrc_{best_infer_value:.6f}.pth'))
        torch.save(optimizer.state_dict(), os.path.join(
            path_saving, f'best_optim_epoch_{best_epoch:03d}_' + \
            f'inf_mtrc_{best_infer_value:.6f}.pt'))
        print('...saved model/optimizer.') 
    
    else:
        eval_metrics_patient = None
                
    
    if plot:
        print('Plotting last inference results...')
        plotting.plot_moving_fixed_and_outputs(moving=infer_inputs, fixed=infer_targets, 
                                                output=infer_outputs, ddf=infer_ddf, 
                                                sample_nr=0, path_saving=path_saving)
        
        plotting.plot_moving_fixed_and_outputs_seg(moving=infer_inputs_seg, fixed=infer_targets_seg, fixed_img=infer_targets,
                                                    output=infer_outputs_seg,
                                                    sample_nr=-1, path_saving=path_saving)
    
    # clean up
    gc.collect()
    torch.cuda.empty_cache() 
    
    return eval_metrics_patient
    

def evaluate_model(model, device, data_loader,
                    path_saving=None, plot=False):
    """Function to evaluate a Pytorch model and score some metric (supervised).

    Args:
        model (Pytorch model): Network architecture
        device (Pytorch device): cuda device
        data_loader (Pytorch laoder): pytorch data loader with training input and output
        path_saving (str, optional): path where results are saved. Defaults to None.
        plot (bool, optional): whether to plot. Defaults to False.
    """
    
     # set data range for structural similiraty
    data_range = torch.tensor([1.0], device=device)
       
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
    
    
    with torch.no_grad():
        model.eval()
        for batch_data in data_loader:
            inputs, targets, \
                inputs_seg, targets_seg = batch_data["moving_image"].to(device), batch_data["fixed_image"].to(device), \
                                                    batch_data["moving_seg"].to(device), batch_data["fixed_seg"].to(device)
            # print(f'Shape of inference inputs: {inputs.shape}')  # (b,1,224,224)
            # print(f'Shape of inference outputs: {outputs.shape}')  # (b,1,224,224)
            
            # forward pass
            # t0 = time.time()
            outputs, outputs_seg, ddf = model(moving_img=torch.cat((inputs, targets),dim=1), moving_seg=inputs_seg, eval_warping=True)
            # t1 = time.time()
            # print(f'Time for forward pass: {(t1 - t0)*1000} ms')
            
            # get center of mass of segmentations for RMSE
            outputs_com, targets_com = utils.get_segmentation_com(outputs_seg, targets_seg)
            
            # compute and collect metrics
            mse_metric(outputs, targets)
            ssim_metric(outputs, targets)
            dice_metric(outputs_seg, targets_seg)
            hd95_metric(outputs_seg, targets_seg)
            hd50_metric(outputs_seg, targets_seg)
            negj_metric(ddf)
            rmseSI_metric(outputs_com[:,0,None], targets_com[:,0,None])
            rmseAP_metric(outputs_com[:,1,None], targets_com[:,1,None])

        # print(torch.sum(targets_seg[-2,...]))  
        # print(torch.sum(outputs_seg[-2,...]))
        # compute metrics by averaging over batches and store results
        mean_mse_metric = round(np.nanmean(mse_metric.get_buffer().detach().cpu()), 4)
        std_mse_metric = round(np.nanstd(mse_metric.get_buffer().detach().cpu()), 4)
        mean_ssim_metric = round(np.nanmean(ssim_metric.get_buffer().detach().cpu()), 4)
        std_ssim_metric = round(np.nanstd(ssim_metric.get_buffer().detach().cpu()), 4)
        mean_dice_metric = round(np.nanmean(dice_metric.get_buffer().detach().cpu()), 2)
        std_dice_metric = round(np.nanstd(dice_metric.get_buffer().detach().cpu()), 2)
        mean_hd95_metric = round(np.nanmean(hd95_metric.get_buffer().detach().cpu()), 2)
        std_hd95_metric = round(np.nanstd(hd95_metric.get_buffer().detach().cpu()), 2)    
        mean_hd50_metric = round(np.nanmean(hd50_metric.get_buffer().detach().cpu()), 2)
        std_hd50_metric = round(np.nanstd(hd50_metric.get_buffer().detach().cpu()), 2)       
        mean_negj_metric = round(np.nanmean(negj_metric.get_buffer().detach().cpu()), 2)
        std_negj_metric = round(np.nanstd(negj_metric.get_buffer().detach().cpu()), 2)    
        mean_rmseSI_metric = round(np.nanmean(rmseSI_metric.get_buffer().detach().cpu()), 2)
        std_rmseSI_metric = round(np.nanstd(rmseSI_metric.get_buffer().detach().cpu()), 2)
        mean_rmseAP_metric = round(np.nanmean(rmseAP_metric.get_buffer().detach().cpu()), 2)
        std_rmseAP_metric = round(np.nanstd(rmseAP_metric.get_buffer().detach().cpu()), 2)
        
    # store mean and std per patient in dict
    eval_metrics = {"MSE (mean)": mean_mse_metric,
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
    
       
    # save metrics for all frames for each patient in txt for later analysis
    np.savetxt(os.path.join(path_saving, 'mse_metric.txt'), mse_metric.get_buffer().squeeze().detach().cpu().numpy())
    np.savetxt(os.path.join(path_saving, 'ssim_metric.txt'), ssim_metric.get_buffer().squeeze().detach().cpu().numpy())
    np.savetxt(os.path.join(path_saving, 'dice_metric.txt'), dice_metric.get_buffer().squeeze().detach().cpu().numpy())
    np.savetxt(os.path.join(path_saving, 'hd95_metric.txt'), hd95_metric.get_buffer().squeeze().detach().cpu().numpy())
    np.savetxt(os.path.join(path_saving, 'hd50_metric.txt'), hd50_metric.get_buffer().squeeze().detach().cpu().numpy())
    np.savetxt(os.path.join(path_saving, 'negj_metric.txt'), negj_metric.get_buffer().squeeze().detach().cpu().numpy())
    np.savetxt(os.path.join(path_saving, 'rmseSI_metric.txt'), rmseSI_metric.get_buffer().squeeze().detach().cpu().numpy())
    np.savetxt(os.path.join(path_saving, 'rmseAP_metric.txt'), rmseAP_metric.get_buffer().squeeze().detach().cpu().numpy())

    if plot:
        print('Plotting some results...')
        plotting.plot_moving_fixed_and_outputs(moving=inputs, fixed=targets, 
                                                output=outputs, ddf=ddf, 
                                                sample_nr=-1, path_saving=path_saving)
        
        plotting.plot_moving_fixed_and_outputs_seg(moving=inputs_seg, fixed=targets_seg, fixed_img=targets,
                                                    output=outputs_seg,
                                                    sample_nr=-1, path_saving=path_saving)
        
    # clean up
    gc.collect()
    torch.cuda.empty_cache() 
        
    return eval_metrics

