# %%

import torch

#%%

class DisplacementRegularizer2D(torch.nn.Module):
    def __init__(self, reg_type):
        super().__init__()
        self.reg_type = reg_type

    # compute gradients in x using forward differences method
    def gradient_dx(self, fv): return (fv[:, 2:, 1:-1] - fv[:, :-2, 1:-1]) / 2
    
    # compute gradients in y using forward differences method
    def gradient_dy(self, fv): return (fv[:, 1:-1, 2:] - fv[:, 1:-1, :-2]) / 2

    def apply_gradient(self, displacement, gradient_fn):
        """Apply gradinet function in a specifed direction to displacement vector field in all directions

        Args:
            displacement (tensor): displacment vecotr field for both spatial dimensions, i.e. with shape (b, 2, h, w)
            gradient_fn (function): function to compute gradient in one direction

        Returns:
            tensor: tensor of gradients along each direction
        """
        return torch.stack([gradient_fn(displacement[:,i,...]) for i in [0, 1]], dim=1)

    def compute_gradient_norm(self, displacement, flag_l1=False):
        "diffusion regularizer, acting on first derivative of displacements"
        dTdx = self.apply_gradient(displacement, self.gradient_dx)
        dTdy = self.apply_gradient(displacement, self.gradient_dy)
        if flag_l1:
            norms = torch.abs(dTdx) + torch.abs(dTdy)
        else:
            norms = dTdx**2 + dTdy**2
        return torch.mean(norms)/2.0

    def compute_bending_energy(self, displacement):
        "bending energy regularizer, acting on second derivative of displacements"
        dTdx = self.apply_gradient(displacement, self.gradient_dx)
        dTdy = self.apply_gradient(displacement, self.gradient_dy)
        dTdxx = self.apply_gradient(dTdx, self.gradient_dx)
        dTdyy = self.apply_gradient(dTdy, self.gradient_dy)
        dTdxy = self.apply_gradient(dTdx, self.gradient_dy)
        return torch.mean(dTdxx**2 + dTdyy**2 + 2*dTdxy**2)

    def forward(self, displacement):
        if self.reg_type == 'bending':
            reg = self.compute_bending_energy(displacement)
        elif self.reg_type == 'gradient-l2':
            reg = self.compute_gradient_norm(displacement)
        elif self.reg_type == 'gradient-l1':
            reg = self.compute_gradient_norm(displacement, flag_l1=True)
        else:
            raise Exception('Unknown displacement regulariser!')
        return reg


class DisplacementRegularizer3D(torch.nn.Module):
    def __init__(self, reg_type):
        super().__init__()
        self.reg_type = reg_type

    def gradient_dx(self, fv): return (fv[:, 2:, 1:-1, 1:-1] - fv[:, :-2, 1:-1, 1:-1]) / 2

    def gradient_dy(self, fv): return (fv[:, 1:-1, 2:, 1:-1] - fv[:, 1:-1, :-2, 1:-1]) / 2

    def gradient_dz(self, fv): return (fv[:, 1:-1, 1:-1, 2:] - fv[:, 1:-1, 1:-1, :-2]) / 2

    def apply_gradient(self, displacement, gradient_fn):
        return torch.stack([gradient_fn(displacement[:,i,...]) for i in [0, 1, 2]], dim=1)

    def compute_gradient_norm(self, displacement, flag_l1=False):
        dTdx = self.apply_gradient(displacement, self.gradient_dx)
        dTdy = self.apply_gradient(displacement, self.gradient_dy)
        dTdz = self.apply_gradient(displacement, self.gradient_dz)
        if flag_l1:
            norms = torch.abs(dTdx) + torch.abs(dTdy) + torch.abs(dTdz)
        else:
            norms = dTdx**2 + dTdy**2 + dTdz**2
        return torch.mean(norms)/3.0

    def compute_bending_energy(self, displacement):
        dTdx = self.apply_gradient(displacement, self.gradient_dx)
        dTdy = self.apply_gradient(displacement, self.gradient_dy)
        dTdz = self.apply_gradient(displacement, self.gradient_dz)
        dTdxx = self.apply_gradient(dTdx, self.gradient_dx)
        dTdyy = self.apply_gradient(dTdy, self.gradient_dy)
        dTdzz = self.apply_gradient(dTdz, self.gradient_dz)
        dTdxy = self.apply_gradient(dTdx, self.gradient_dy)
        dTdyz = self.apply_gradient(dTdy, self.gradient_dz)
        dTdxz = self.apply_gradient(dTdx, self.gradient_dz)
        return torch.mean(dTdxx**2 + dTdyy**2 + dTdzz**2 + 2*dTdxy**2 + 2*dTdxz**2 + 2*dTdyz**2)

    def forward(self, displacement):
        if self.reg_type == 'bending':
            reg = self.compute_bending_energy(displacement)
        elif self.reg_type == 'gradient-l2':
            reg = self.compute_gradient_norm(displacement)
        elif self.reg_type == 'gradient-l1':
            reg = self.compute_gradient_norm(displacement, flag_l1=True)
        else:
            raise Exception('Unknown displacement regulariser!')
        return reg


class CombinedImageDisplacementLoss(torch.nn.Module):
    def __init__(self, image_loss, displacement_loss, image_weight=0.5, displacement_weight=0.5):
        super(CombinedImageDisplacementLoss, self).__init__()
        self.image_loss = image_loss
        self.displacement_loss = displacement_loss
        self.image_weight = image_weight
        self.displacement_weight = displacement_weight

    def forward(self, y_pred, y_true, displacement):
        loss = self.image_weight*self.image_loss(y_pred, y_true) + self.displacement_weight*self.displacement_loss(displacement)
        return loss
    
    
class CombinedImageSegmentationDisplacementLoss(torch.nn.Module):
    def __init__(self, image_loss, segmentation_loss, displacement_loss, image_weight=0.5, segmentation_weight=0.5, displacement_weight=0.5):
        super(CombinedImageSegmentationDisplacementLoss, self).__init__()
        self.image_loss = image_loss
        self.segmentation_loss = segmentation_loss
        self.displacement_loss = displacement_loss
        self.image_weight = image_weight
        self.segmentation_weight = segmentation_weight
        self.displacement_weight = displacement_weight

    def forward(self, y_pred, y_true, y_pred_seg, y_true_seg, displacement):
        loss = self.image_weight*self.image_loss(y_pred, y_true) + self.segmentation_weight*self.segmentation_loss(y_pred_seg, y_true_seg) + self.displacement_weight*self.displacement_loss(displacement)
        return loss

