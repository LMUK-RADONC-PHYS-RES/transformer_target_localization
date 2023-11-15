from __future__ import absolute_import
#%%
"""
author: Elia Lombardo
email: elia.lombardo@med.uni-muenchen.de
"""
import numpy as np

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


#%%
'''
BSpline Transformations

Original code retrieved from:
https://github.com/qiuhuaqi/midir

Original paper:
Qiu, H., Qin, C., Schuh, A., Hammernik, K., & Rueckert, D. (2021, February).
Learning Diffeomorphic and Modality-invariant Registration using B-splines.
In Medical Imaging with Deep Learning.

Modified and tested by:
Junyu Chen
jchen245@jhmi.edu
Johns Hopkins University
'''
import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

class _Transform(object):
    """ Transformation base class """
    def __init__(self,
                 svf=False,
                 svf_steps=7,
                 svf_scale=1):
        self.svf = svf
        self.svf_steps = svf_steps
        self.svf_scale = svf_scale

    def compute_flow(self, x):
        raise NotImplementedError

    def __call__(self, x):
        flow = self.compute_flow(x)
        if self.svf:
            disp = svf_exp(flow,
                           scale=self.svf_scale,
                           steps=self.svf_steps)
            return flow, disp
        else:
            disp = flow
            return disp


class DenseTransform(_Transform):
    """ Dense field transformation """
    def __init__(self,
                 svf=False,
                 svf_steps=7,
                 svf_scale=1):
        super(DenseTransform, self).__init__(svf=svf,
                                             svf_steps=svf_steps,
                                             svf_scale=svf_scale)

    def compute_flow(self, x):
        return x


class CubicBSplineFFDTransform(_Transform):
    def __init__(self,
                 ndim,
                 img_size=(256, 256),
                 cps=(5,5),
                 svf=False,
                 svf_steps=7,
                 svf_scale=1):
        """
        Compute dense displacement field of Cubic B-spline FFD transformation model
        from input control point parameters.
        Args:
            ndim: (int) image dimension
            img_size: (int or tuple) size of the image
            cps: (int or tuple) control point spacing in number of intervals between pixel/voxel centres
            svf: (bool) stationary velocity field formulation if True
        """
        super(CubicBSplineFFDTransform, self).__init__(svf=svf,
                                                       svf_steps=svf_steps,
                                                       svf_scale=svf_scale)
        self.ndim = ndim
        self.img_size = img_size#param_ndim_setup(img_size, self.ndim)
        self.stride = cps#param_ndim_setup(cps, self.ndim)

        self.kernels = self.set_kernel()
        self.padding = [(len(k) - 1) // 2
                        for k in self.kernels]  # the size of the kernel is always odd number

    def set_kernel(self):
        kernels = list()
        for s in self.stride:
            # 1d cubic b-spline kernels
            kernels += [cubic_bspline1d(s)]
        return kernels

    def compute_flow(self, x):
        """
        Args:
            x: (N, dim, *(sizes)) Control point parameters
        Returns:
            y: (N, dim, *(img_sizes)) The dense flow field of the transformation
        """
        # separable 1d transposed convolution
        flow = x
        for i, (k, s, p) in enumerate(zip(self.kernels[:self.ndim], self.stride[:self.ndim], self.padding[:self.ndim])):
            k = k.to(dtype=x.dtype, device=x.device)
            flow = conv1d(flow, dim=i + 2, kernel=k, stride=s, padding=p, transpose=True)

        #  crop the output to image size
        slicer = (slice(0, flow.shape[0]), slice(0, flow.shape[1])) \
                 + tuple(slice(s, s + self.img_size[i]) for i, s in enumerate(self.stride[:self.ndim]))
        flow = flow[slicer]
        return flow


def normalise_disp(disp):
    """
    Spatially normalise DVF to [-1, 1] coordinate system used by Pytorch `grid_sample()`
    Assumes disp size is the same as the corresponding image.
    Args:
        disp: (numpy.ndarray or torch.Tensor, shape (N, ndim, *size)) Displacement field
    Returns:
        disp: (normalised disp)
    """

    ndim = disp.ndim - 2

    if type(disp) is np.ndarray:
        norm_factors = 2. / np.array(disp.shape[2:])
        norm_factors = norm_factors.reshape(1, ndim, *(1,) * ndim)

    elif type(disp) is torch.Tensor:
        norm_factors = torch.tensor(2.) / torch.tensor(disp.size()[2:], dtype=disp.dtype, device=disp.device)
        norm_factors = norm_factors.view(1, ndim, *(1,)*ndim)

    else:
        raise RuntimeError("Input data type not recognised, expect numpy.ndarray or torch.Tensor")
    return disp * norm_factors


def svf_exp(flow, scale=1, steps=5, sampling='bilinear'):
    """ Exponential of velocity field by Scaling and Squaring"""
    disp = flow * (scale / (2 ** steps))
    for _ in range(steps):
        disp = disp + warp(x=disp, disp=disp,
                           interp_mode=sampling)
    return disp


def cubic_bspline_value(x: float, derivative: int = 0) -> float:
    r"""Evaluate 1-dimensional cubic B-spline."""
    t = abs(x)
    # outside local support region
    if t >= 2:
        return 0
    # 0-th order derivative
    if derivative == 0:
        if t < 1:
            return 2 / 3 + (0.5 * t - 1) * t ** 2
        return -((t - 2) ** 3) / 6
    # 1st order derivative
    if derivative == 1:
        if t < 1:
            return (1.5 * t - 2.0) * x
        if x < 0:
            return 0.5 * (t - 2) ** 2
        return -0.5 * (t - 2) ** 2
    # 2nd oder derivative
    if derivative == 2:
        if t < 1:
            return 3 * t - 2
        return -t + 2


def cubic_bspline1d(stride, derivative: int = 0, dtype=None, device= None) -> torch.Tensor:
    r"""Cubic B-spline kernel for specified control point spacing.
    Args:
        stride: Spacing between control points with respect to original (upsampled) image grid.
        derivative: Order of cubic B-spline derivative.
    Returns:
        Cubic B-spline convolution kernel.
    """
    if dtype is None:
        dtype = torch.float
    if not isinstance(stride, int):
        (stride,) = stride
    kernel = torch.ones(4 * stride - 1, dtype=dtype)
    radius = kernel.shape[0] // 2
    for i in range(kernel.shape[0]):
        kernel[i] = cubic_bspline_value((i - radius) / stride, derivative=derivative)
    if device is None:
        device = kernel.device
    return kernel.to(device)


def conv1d(
        data: Tensor,
        kernel: Tensor,
        dim: int = -1,
        stride: int = 1,
        dilation: int = 1,
        padding: int = 0,
        transpose: bool = False
) -> Tensor:
    r"""Convolve data with 1-dimensional kernel along specified dimension."""
    result = data.type(kernel.dtype)  # (n, ndim, h, w, d)
    result = result.transpose(dim, -1)  # (n, ndim, ..., shape[dim])
    shape_ = result.size()
    # use native pytorch
    groups = int(torch.prod(torch.tensor(shape_[1:-1])))
    # groups = numel(shape_[1:-1])  # (n, nidim * shape[not dim], shape[dim])
    weight = kernel.expand(groups, 1, kernel.shape[-1])  # 3*w*d, 1, kernel_size
    result = result.reshape(shape_[0], groups, shape_[-1])  # n, 3*w*d, shape[dim]
    conv_fn = F.conv_transpose1d if transpose else F.conv1d
    result = conv_fn(
        result,
        weight,
        stride=stride,
        dilation=dilation,
        padding=padding,
        groups=groups,
    )
    result = result.reshape(shape_[0:-1] + result.shape[-1:])
    result = result.transpose(-1, dim)
    return result


def warp(x, disp, interp_mode="bilinear"):
    """
    Spatially transform an image by sampling at transformed locations (2D and 3D)
    Args:
        x: (Tensor float, shape (N, ndim, *sizes)) input image
        disp: (Tensor float, shape (N, ndim, *sizes)) dense disp field in i-j-k order (NOT spatially normalised)
        interp_mode: (string) mode of interpolation in grid_sample()
    Returns:
        deformed x, Tensor of the same shape as input
    """
    ndim = x.ndim - 2
    size = x.size()[2:]
    disp = disp.type_as(x)

    # normalise disp to [-1, 1]
    disp = normalise_disp(disp)

    # generate standard mesh grid
    grid = torch.meshgrid([torch.linspace(-1, 1, size[i]).type_as(disp) for i in range(ndim)])
    grid = [grid[i].requires_grad_(False) for i in range(ndim)]

    # apply displacements to each direction (N, *size)
    warped_grid = [grid[i] + disp[:, i, ...] for i in range(ndim)]

    # swapping i-j-k order to x-y-z (k-j-i) order for grid_sample()
    warped_grid = [warped_grid[ndim - 1 - i] for i in range(ndim)]
    warped_grid = torch.stack(warped_grid, -1)  # (N, *size, dim)

    return F.grid_sample(x, warped_grid, mode=interp_mode, align_corners=False)


#%%

"""
Thie following code was obtained from: https://github.com/uncbiag/easyreg

*finite_difference.py* is the main package to compute finite differences in
1D, 2D, and 3D on numpy arrays (class FD_np) and pytorch tensors (class FD_torch).
The package supports first and second order derivatives and Neumann and linear extrapolation
boundary conditions (though the latter have not been tested extensively yet).
"""
# from builtins import object
from abc import ABCMeta, abstractmethod
import torch
from torch.autograd import Variable
import numpy as np
from future.utils import with_metaclass

MyTensor = torch.FloatTensor

class FD(with_metaclass(ABCMeta, object)):
    """
    *FD* is the abstract class for finite differences. It includes most of the actual finite difference code,
    but requires the definition (in a derived class) of the methods *get_dimension*, *create_zero_array*, and *get_size_of_array*.
    In this way the numpy and pytorch versions can easily be derived. All the method expect BxXxYxZ format (i.e., they process a batch at a time)
    """

    def __init__(self, spacing, mode='linear'):
        """
        Constructor
        :param spacing: 1D numpy array defining the spatial spacing, e.g., [0.1,0.1,0.1] for a 3D image
        :param bcNeumannZero: Defines the boundary condition. If set to *True* (default) zero Neumann boundary conditions
            are imposed. If set to *False* linear extrapolation is used (this is still experimental, but may be beneficial
            for better boundary behavior)
        """
        self.dim = spacing.size
        """spatial dimension"""
        self.spacing = np.ones(self.dim)
        """spacing"""
        assert mode in ['linear', 'neumann_zero', 'dirichlet_zero'], \
            " boundary condition {} is not supported , supported list 'linear', 'neumann_zero', 'dirichlet_zero'".format(
                mode)
        self.bcNeumannZero = mode == 'neumann_zero'  # if false then linear interpolation
        self.bclinearInterp = mode == 'linear'
        self.bcDirichletZero = mode == 'dirichlet_zero'
        """should Neumann boundary conditions be used? (otherwise linear extrapolation)"""
        if spacing.size == 1:
            self.spacing[0] = spacing[0]
        elif spacing.size == 2:
            self.spacing[0] = spacing[0]
            self.spacing[1] = spacing[1]
        elif spacing.size == 3:
            self.spacing = spacing
        else:
            raise ValueError('Finite differences are only supported in dimensions 1 to 3')

    def dXb(self, I):
        """
        Backward difference in x direction:
        :math:`\\frac{dI(i)}{dx}\\approx\\frac{I_i-I_{i-1}}{h_x}`
        :param I: Input image
        :return: Returns the first derivative in x direction using backward differences
        """
        res = (I - self.xm(I)) * (1. / self.spacing[0])
        return res

    def dXf(self, I):
        """
        Forward difference in x direction:
        :math:`\\frac{dI(i)}{dx}\\approx\\frac{I_{i+1}-I_{i}}{h_x}`

        :param I: Input image
        :return: Returns the first derivative in x direction using forward differences
        """
        res = (self.xp(I) - I) * (1. / self.spacing[0])
        return res

    def dXc(self, I):
        """
        Central difference in x direction:
        :math:`\\frac{dI(i)}{dx}\\approx\\frac{I_{i+1}-I_{i-1}}{2h_x}`

        :param I: Input image
        :return: Returns the first derivative in x direction using central differences
        """
        res = (self.xp(I, central=True) - self.xm(I, central=True)) * (0.5 / self.spacing[0])
        return res

    def ddXc(self, I):
        """
        Second deriative in x direction

        :param I: Input image
        :return: Returns the second derivative in x direction
        """
        res = (self.xp(I, central=True) - I - I + self.xm(I, central=True)) * (1 / (self.spacing[0] ** 2))
        return res

    def dYb(self, I):
        """
        Same as dXb, but for the y direction

        :param I: Input image
        :return: Returns the first derivative in y direction using backward differences
        """
        res = (I - self.ym(I)) * (1. / self.spacing[1])
        return res

    def dYf(self, I):
        """
        Same as dXf, but for the y direction

        :param I: Input image
        :return: Returns the first derivative in y direction using forward differences
        """
        res = (self.yp(I) - I) * (1. / self.spacing[1])
        return res

    def dYc(self, I):
        """
        Same as dXc, but for the y direction

        :param I: Input image
        :return: Returns the first derivative in y direction using central differences
        """
        res = (self.yp(I, central=True) - self.ym(I, central=True)) * (0.5 / self.spacing[1])
        return res

    def ddYc(self, I):
        """
        Same as ddXc, but for the y direction

        :param I: Input image
        :return: Returns the second derivative in the y direction
        """
        res = (self.yp(I, central=True) - I - I + self.ym(I, central=True)) * (1 / (self.spacing[1] ** 2))
        return res

    def dZb(self, I):
        """
        Same as dXb, but for the z direction

        :param I: Input image
        :return: Returns the first derivative in the z direction using backward differences
        """
        res = (I - self.zm(I)) * (1. / self.spacing[2])
        return res

    def dZf(self, I):
        """
        Same as dXf, but for the z direction

        :param I: Input image
        :return: Returns the first derivative in the z direction using forward differences
        """
        res = (self.zp(I) - I) * (1. / self.spacing[2])
        return res

    def dZc(self, I):
        """
        Same as dXc, but for the z direction

        :param I: Input image
        :return: Returns the first derivative in the z direction using central differences
        """
        res = (self.zp(I, central=True) - self.zm(I, central=True)) * (0.5 / self.spacing[2])
        return res

    def ddZc(self, I):
        """
        Same as ddXc, but for the z direction

        :param I: Input iamge
        :return: Returns the second derivative in the z direction
        """
        res = (self.zp(I, central=True) - I - I + self.zm(I, central=True)) * (1 / (self.spacing[2] ** 2))
        return res

    def lap(self, I):
        """
        Compute the Lapacian of an image
        !!!!!!!!!!!
        IMPORTANT:
        ALL THE FOLLOWING IMPLEMENTED CODE ADD 1 ON DIMENSION, WHICH REPRESENT BATCH DIMENSION.
        THIS IS FOR COMPUTATIONAL EFFICIENCY.

        :param I: Input image [batch, channel, X,Y,Z]
        :return: Returns the Laplacian
        """
        ndim = self.getdimension(I)
        if ndim == 1 + 1:
            return self.ddXc(I)
        elif ndim == 2 + 1:
            return (self.ddXc(I) + self.ddYc(I))
        elif ndim == 3 + 1:
            return (self.ddXc(I) + self.ddYc(I) + self.ddZc(I))
        else:
            raise ValueError('Finite differences are only supported in dimensions 1 to 3')

    def grad_norm_sqr_c(self, I):
        """
        Computes the gradient norm of an image
        !!!!!!!!!!!
        IMPORTANT:
        ALL THE FOLLOWING IMPLEMENTED CODE ADD 1 ON DIMENSION, WHICH REPRESENT BATCH DIMENSION.
        THIS IS FOR COMPUTATIONAL EFFICIENCY.
        :param I: Input image [batch, channel, X,Y,Z]
        :return: returns ||grad I||^2
        """
        ndim = self.getdimension(I)
        if ndim == 1 + 1:
            return self.dXc(I) ** 2
        elif ndim == 2 + 1:
            return (self.dXc(I) ** 2 + self.dYc(I) ** 2)
        elif ndim == 3 + 1:
            return (self.dXc(I) ** 2 + self.dYc(I) ** 2 + self.dZc(I) ** 2)
        else:
            raise ValueError('Finite differences are only supported in dimensions 1 to 3')

    def grad_norm_sqr_f(self, I):
        """
        Computes the gradient norm of an image
        !!!!!!!!!!!
        IMPORTANT:
        ALL THE FOLLOWING IMPLEMENTED CODE ADD 1 ON DIMENSION, WHICH REPRESENT BATCH DIMENSION.
        THIS IS FOR COMPUTATIONAL EFFICIENCY.
        :param I: Input image [batch, channel, X,Y,Z]
        :return: returns ||grad I||^2
        """
        ndim = self.getdimension(I)
        if ndim == 1 + 1:
            return self.dXf(I) ** 2
        elif ndim == 2 + 1:
            return (self.dXf(I) ** 2 + self.dYf(I) ** 2)
        elif ndim == 3 + 1:
            return (self.dXf(I) ** 2 + self.dYf(I) ** 2 + self.dZf(I) ** 2)
        else:
            raise ValueError('Finite differences are only supported in dimensions 1 to 3')

    def grad_norm_sqr_b(self, I):
        """
        Computes the gradient norm of an image
        !!!!!!!!!!!
        IMPORTANT:
        ALL THE FOLLOWING IMPLEMENTED CODE ADD 1 ON DIMENSION, WHICH REPRESENT BATCH DIMENSION.
        THIS IS FOR COMPUTATIONAL EFFICIENCY.
        :param I: Input image [batch, channel, X,Y,Z]
        :return: returns ||grad I||^2
        """
        ndim = self.getdimension(I)
        if ndim == 1 + 1:
            return self.dXb(I) ** 2
        elif ndim == 2 + 1:
            return (self.dXb(I) ** 2 + self.dYb(I) ** 2)
        elif ndim == 3 + 1:
            return (self.dXb(I) ** 2 + self.dYb(I) ** 2 + self.dZb(I) ** 2)
        else:
            raise ValueError('Finite differences are only supported in dimensions 1 to 3')

    @abstractmethod
    def getdimension(self, I):
        """
        Abstract method to return the dimension of an input image I

        :param I: Input image
        :return: Returns the dimension of the image I
        """
        pass

    @abstractmethod
    def create_zero_array(self, sz):
        """
        Abstract method to create a zero array of a given size, sz. E.g., sz=[10,2,5]

        :param sz: Size array
        :return: Returns a zero array of the specified size
        """
        pass

    @abstractmethod
    def get_size_of_array(self, A):
        """
        Abstract method to return the size of an array (as a vector)

        :param A: Input array
        :return: Returns its size (e.g., [5,10] or [3,4,6]
        """
        pass

    def xp(self, I, central=False):
        """
        !!!!!!!!!!!
        IMPORTANT:
        ALL THE FOLLOWING IMPLEMENTED CODE ADD 1 ON DIMENSION, WHICH REPRESENT BATCH DIMENSION.
        THIS IS FOR COMPUTATIONAL EFFICIENCY.
        Returns the values for x-index incremented by one (to the right in 1D)

        :param I: Input image [batch, channel, X, Y,Z]
        :return: Image with values at an x-index one larger
        """
        rxp = self.create_zero_array(self.get_size_of_array(I))
        ndim = self.getdimension(I)
        if ndim in [1 + 1, 2 + 1, 3 + 1]:
            rxp[:, 0:-1] = I[:, 1:]
            if self.bcNeumannZero:
                rxp[:, -1] = I[:, -1]
                if central:
                    rxp[:, 0] = I[:, 0]
            elif self.bclinearInterp:
                rxp[:, -1] = 2 * I[:, -1] - I[:, -2]
            elif self.bcDirichletZero:
                rxp[:, -1] = 0.
        else:
            raise ValueError('Finite differences are only supported in dimensions 1 to 3')
        return rxp

    def xm(self, I, central=False):
        """
        !!!!!!!!!!!
        IMPORTANT:
        ALL THE FOLLOWING IMPLEMENTED CODE ADD 1 ON DIMENSION, WHICH REPRESENT BATCH DIMENSION.
        THIS IS FOR COMPUTATIONAL EFFICIENCY.
        Returns the values for x-index decremented by one (to the left in 1D)

        :param I: Input image [batch, channel, X, Y, Z]
        :return: Image with values at an x-index one smaller
        """
        rxm = self.create_zero_array(self.get_size_of_array(I))
        ndim = self.getdimension(I)
        if ndim in [1 + 1, 2 + 1, 3 + 1]:
            rxm[:, 1:] = I[:, 0:-1]
            if self.bcNeumannZero:
                rxm[:, 0] = I[:, 0]
                if central:
                    rxm[:, -1] = I[:, -1]
            elif self.bclinearInterp:
                rxm[:, 0] = 2. * I[:, 0] - I[:, 1]
            elif self.bcDirichletZero:
                rxm[:, 0] = 0.
        else:
            raise ValueError('Finite differences are only supported in dimensions 1 to 3')
        return rxm

    def yp(self, I, central=False):
        """
        !!!!!!!!!!!
        IMPORTANT:
        ALL THE FOLLOWING IMPLEMENTED CODE ADD 1 ON DIMENSION, WHICH REPRESENT BATCH DIMENSION.
        THIS IS FOR COMPUTATIONAL EFFICIENCY.
        Same as xp, but for the y direction

        :param I: Input image
        :return: Image with values at y-index one larger
        """
        ryp = self.create_zero_array(self.get_size_of_array(I))
        ndim = self.getdimension(I)
        if ndim in [2 + 1, 3 + 1]:
            ryp[:, :, 0:-1] = I[:, :, 1:]
            if self.bcNeumannZero:
                ryp[:, :, -1] = I[:, :, -1]
                if central:
                    ryp[:, :, 0] = I[:, :, 0]
            elif self.bclinearInterp:
                ryp[:, :, -1] = 2. * I[:, :, -1] - I[:, :, -2]
            elif self.bcDirichletZero:
                ryp[:, :, -1] = 0.
        else:
            raise ValueError('Finite differences are only supported in dimensions 1 to 3')
        return ryp

    def ym(self, I, central=False):
        """
        Same as xm, but for the y direction
        !!!!!!!!!!!
        IMPORTANT:
        ALL THE FOLLOWING IMPLEMENTED CODE ADD 1 ON DIMENSION, WHICH REPRESENT BATCH DIMENSION.
        THIS IS FOR COMPUTATIONAL EFFICIENCY.
        Returns the values for x-index decremented by one (to the left in 1D)
        :param I: Input image [batch, channel, X, Y, Z]
        :return: Image with values at y-index one smaller
        """
        rym = self.create_zero_array(self.get_size_of_array(I))
        ndim = self.getdimension(I)
        if ndim in [2 + 1, 3 + 1]:
            rym[:, :, 1:] = I[:, :, 0:-1]
            if self.bcNeumannZero:
                rym[:, :, 0] = I[:, :, 0]
                if central:
                    rym[:, :, -1] = I[:, :, -1]
            elif self.bclinearInterp:
                rym[:, :, 0] = 2. * I[:, :, 0] - I[:, :, 1]
            elif self.bcDirichletZero:
                rym[:, :, 0] = 0.
        else:
            raise ValueError('Finite differences are only supported in dimensions 1 to 3')
        return rym

    def zp(self, I, central=False):
        """
        Same as xp, but for the z direction

        !!!!!!!!!!!
        IMPORTANT:
        ALL THE FOLLOWING IMPLEMENTED CODE ADD 1 ON DIMENSION, WHICH REPRESENT BATCH DIMENSION.
        THIS IS FOR COMPUTATIONAL EFFICIENCY.
        Returns the values for x-index decremented by one (to the left in 1D)
        :param I: Input image [batch, channel, X, Y, Z]
        :return: Image with values at z-index one larger
        """
        rzp = self.create_zero_array(self.get_size_of_array(I))
        ndim = self.getdimension(I)
        if ndim in [3 + 1]:
            rzp[:, :, :, 0:-1] = I[:, :, :, 1:]
            if self.bcNeumannZero:
                rzp[:, :, :, -1] = I[:, :, :, -1]
                if central:
                    rzp[:, :, :, 0] = I[:, :, :, 0]
            elif self.bclinearInterp:
                rzp[:, :, :, -1] = 2. * I[:, :, :, -1] - I[:, :, :, -2]
            elif self.bcDirichletZero:
                rzp[:, :, :, -1] = 0.
        else:
            raise ValueError('Finite differences are only supported in dimensions 1 to 3')
        return rzp

    def zm(self, I, central=False):
        """
        Same as xm, but for the z direction

        !!!!!!!!!!!
        IMPORTANT:
        ALL THE FOLLOWING IMPLEMENTED CODE ADD 1 ON DIMENSION, WHICH REPRESENT BATCH DIMENSION.
        THIS IS FOR COMPUTATIONAL EFFICIENCY.
        Returns the values for x-index decremented by one (to the left in 1D)
        :param I: Input image [batch, channel, X, Y, Z]
        :return: Image with values at z-index one smaller
        """
        rzm = self.create_zero_array(self.get_size_of_array(I))
        ndim = self.getdimension(I)
        if ndim in [3 + 1]:
            rzm[:, :, :, 1:] = I[:, :, :, 0:-1]
            if self.bcNeumannZero:
                rzm[:, :, :, 0] = I[:, :, :, 0]
                if central:
                    rzm[:, :, :, -1] = I[:, :, :, -1]
            elif self.bclinearInterp:
                rzm[:, :, :, 0] = 2. * I[:, :, :, 0] - I[:, :, :, 1]
            elif self.bcDirichletZero:
                rzm[:, :, :, 0] = 0.
        else:
            raise ValueError('Finite differences are only supported in dimensions 1 to 3')
        return rzm


class FD_np(FD):
    """
    Defnitions of the abstract methods for numpy
    """

    def __init__(self, dim, mode='linear'):
        """
        Constructor for numpy finite differences
        :param spacing: spatial spacing (array with as many entries as there are spatial dimensions)
        :param bcNeumannZero: Specifies if zero Neumann conditions should be used (if not, uses linear extrapolation)
        """
        super(FD_np, self).__init__(dim, mode)

    def getdimension(self, I):
        """
        Returns the dimension of an image
        :param I: input image
        :return: dimension of the input image
        """
        return I.ndim

    def create_zero_array(self, sz):
        """
        Creates a zero array
        :param sz: size of the zero array, e.g., [3,4,2]
        :return: the zero array
        """
        return np.zeros(sz)

    def get_size_of_array(self, A):
        """
        Returns the size (shape in numpy) of an array
        :param A: input array
        :return: shape/size
        """
        return A.shape


class FD_torch(FD):
    """
    Defnitions of the abstract methods for torch
    """

    def __init__(self, dim, mode='linear'):
        """
          Constructor for torch finite differences
          :param spacing: spatial spacing (array with as many entries as there are spatial dimensions)
          :param bcNeumannZero: Specifies if zero Neumann conditions should be used (if not, uses linear extrapolation)
          """
        super(FD_torch, self).__init__(dim, mode)

    def getdimension(self, I):
        """
        Returns the dimension of an image
        :param I: input image
        :return: dimension of the input image
        """
        return I.dim()

    def create_zero_array(self, sz):
        """
        Creats a zero array
        :param sz: size of the array, e.g., [3,4,2]
        :return: the zero array
        """
        return MyTensor(sz).zero_()

    def get_size_of_array(self, A):
        """
        Returns the size (size()) of an array
        :param A: input array
        :return: shape/size
        """
        return A.size()
    
    
#%%

""" Python wrapper for Plastimatch
original version of code by Paolo Zaffino (p.zaffino@unicz.it)
    https://github.com/pzaffino/
edited and adapted for Python3 by Moritz Rabe (moritz.rabe@med.uni-muenchen.de)
"""

import os
import subprocess

####################### PUBLIC CLASSES - START - #######################


class add:
	
	def __init__ (self, log_file=""):
		self.log_file=log_file
	
	input_files=[]
	output_file=""
	
	def run_add(self):
		
		if len(self.input_files) < 2:
			raise NameError("You must define at least two input images!")
		
		if self.log_file == "":
			self.log=open(os.devnull, "w")
		else:
			self.log=open(self.log_file, "w")
		
		input_parms=""
		
		for file_name in self.input_files:
			input_parms+=str(file_name) + " "
		
		subprocess.call("plastimatch add " + input_parms + str(self.output_file),\
		shell=True, stdout=self.log, stderr=self.log)
		
		self.log.close()




class adjust:
	
	def __init__ (self, log_file=""):
		self.log_file=log_file
	
	option={}
	
	_adjust_keys=("input", "output", "output-type", "scale", "ab-scale",\
	"stretch", "truncate-above", "truncate-below")
	
	def run_adjust(self):
		_run_plm_command("adjust", self.option, self._adjust_keys, self.log_file)




class convert:
	
	def __init__ (self, log_file=""):
		self.log_file=log_file
	
	option={}
	
	_convert_keys=("input","default-val","dif","dim","fixed",\
	"input-cxt","input-dose-ast","input-dose-img","input-dose-mc",\
	"input-dose-xio","input-ss-img","input-ss-list","interpolation",\
	"metadata","origin","output-color_map","output-ctx","output-dicom",\
	"output-dij","output-dose_img","output-img","output-labelmap",\
	"output-pointset","output-prefix","output-prefix_fcsv","output-ss_img",\
	"output-ss_list","output-type","output-vf","output-xio","patient-id",\
	"patient-name","patient-pos","prune-empty","referenced-ct","simplify-perc",\
	"spacing","xf","xor-contours")
	
	def run_convert(self):
		_run_plm_command("convert", self.option, self._convert_keys, self.log_file)




class crop:
	
	def __init__ (self, log_file=""):
		self.log_file=log_file
		
	option={}
	
	_crop_keys=("input", "output", "voxels")
	
	def run_crop(self):
		_run_plm_command("crop", self.option, self._crop_keys, self.log_file)




class dice:
	
	def __init__ (self, log_file=""):
		self.log_file=log_file
	
	input_files=[]
	
	def run_dice(self):
		
		if len(self.input_files) != 2:
			raise NameError("You must define two input structures!")
		
		if self.log_file == "":
			raise NameError("You must define a log file!")
		else:
			self.log=open(self.log_file, "w")
		
		subprocess.call("plastimatch dice " + str(self.input_files[0]) + " " + str(self.input_files[1]), \
		shell=True, stdout=self.log, stderr=self.log)
		
		self.log.close()




class diff:
	
	def __init__ (self, log_file=""):
		self.log_file=log_file
	
	input_files=[]
	output_file=""
	
	def run_diff(self):
		
		if len(self.input_files) != 2:
			raise NameError("You must define two input images!")
		
		if self.log_file == "":
			self.log=open(os.devnull, "w")
		else:
			self.log=open(self.log_file, "w")
		
		subprocess.call("plastimatch diff " + str(self.input_files[0]) + " " + str(self.input_files[1])\
		+ " " + str(self.output_file), shell=True, stdout=self.log, stderr=self.log)
		
		self.log.close()




class fill:
	
	def __init__ (self, log_file=""):
		self.log_file=log_file
	
	option={}
	
	_fill_keys=("input", "mask", "mask-value", "output", "output-format", "output-type")
	
	def run_fill(self):
		_run_plm_command("fill", self.option, self._fill_keys, self.log_file)




class mask:
	
	def __init__ (self, log_file=""):
		self.log_file=log_file
	
	option={}
	
	_mask_keys=("input", "mask", "mask-value", "output", "output-format", "output-type")
	
	def run_mask(self):
		_run_plm_command("mask", self.option, self._mask_keys, self.log_file)




class register:
	
	def __init__ (self, par_file="", log_file=""):
		self.par_file=par_file
		self.log_file=log_file
	
	_stage_keys=("xform","optim","impl","background_max","convergence_tol",\
	"demons_acceleration","demons_filter_width","demons_homogenization","demons_std",\
	"histoeq","grad_tol","grid_spac","max_its","max_step","metric","mi_histogram_bins",\
	"min_its","min_step","num_saples","regularization_lambda","res","ss","ss_fixed",\
	"ss_moving","threading","xform_in","xform_out","vf_out","img_out","img_out_fmt","img_out_type",\
    "fixed_roi","moving_roi","translation_scale_factor")
	
	_global_keys=("fixed","moving","xform_in","xform_out","vf_out","img_out",\
	"img_out_fmt","img_out_type","fixed_roi","moving_roi","resample_when_linear","default_value")
	
	stages=[]
	_global_stage_added=False
	
	def add_global_stage(self):
		if self.par_file=="" or os.path.exists(self.par_file):
			raise NameError("GLOBAL STAGE NOT ADDED! You have to define a new parameters file name")
		else:
			if self._global_stage_added==False:
				self.stages=[{}]+self.stages
				self._global_stage_added=True
			else:
				raise NameError("The global stage already exists!")
	
	def add_stage(self):
		if self.par_file=="" or os.path.exists(self.par_file):
			raise NameError("STAGE NOT ADDED! You have to define a new parameters file name")
		else:
			self.stages+=[{}]
	
	def delete_stage(self, stage_number):
		if self.par_file=="" or os.path.exists(self.par_file):
			raise NameError("STAGE NOT DELETED! You have to define a new parameters file name")
		else:
			if stage_number != 0:
				del self.stages[stage_number]
			else:
				raise NameError("GLOBAL STAGE NOT DELETED! You can not delete the global stage")
	
	def run_registration(self):
		if not os.path.exists(self.par_file) and self.par_file!="":
			f=open(self.par_file, "w")			
			for stage_index, stage in enumerate(self.stages):
				if stage_index==0:
					stage=_clean_parms(stage, self._global_keys)
					f.write("[GLOBAL]\n")
				else:
					stage=_clean_parms(stage, self._stage_keys)
					f.write("\n[STAGE]\n")
				
				for key, value in dict.items(stage):
						f.write(key+"="+value+"\n")	
			f.close()
		
		if self.log_file == "":
			self.log=open(os.devnull, "w")
		else:
			self.log=open(self.log_file, "w")
		
		if self.par_file!="" and os.path.exists(self.par_file):
			print("Registering...")
			subprocess.call("plastimatch register " + self.par_file, shell=True, stdout=self.log, stderr=self.log)
			self.log.close()
			print("...done!")
		else:
			raise NameError("REGISTRATION NOT EXECUTED! You have to define a new parameters file name")




class resample:
	
	def __init__ (self, log_file=""):
		self.log_file=log_file
		
	option={}
	
	_resample_keys=("default-value", "dim", "fixed", "input", "interpolation",\
	"origin", "output", "output-type", "spacing", "subsample")
	
	def run_resample(self):
		_run_plm_command("resample", self.option, self._resample_keys, self.log_file)




class segment:
	
	def __init__ (self, log_file=""):
		self.log_file=log_file
		
	option={}
	
	_segment_keys=("bottom", "debug", "fast", "input", "lower-treshold", "output-img")
	
	def run_segment(self):
		_run_plm_command("segment", self.option, self._segment_keys, self.log_file)




class warp:
	
	def __init__ (self, log_file=""):
		self.log_file=log_file
		
	option={}
	
	_warp_keys=("input","default-val","dif","dim","fixed",\
	"input-cxt","input-dose-ast","input-dose-img","input-dose-mc",\
	"input-dose-xio","input-ss-img","input-ss-list","interpolation",\
	"metadata","origin","output-color_map","output-ctx","output-dicom",\
	"output-dij","output-dose_img","output-img","output-labelmap",\
	"output-pointset","output-prefix","output-prefix_fcsv","output-ss_img",\
	"output-ss_list","output-type","output-vf","output-xio","patient-id",\
	"patient-name","patient-pos","prune-empty","referenced-ct","simplify-perc",\
	"spacing","xf","xor-contours")
	
	def run_warp(self):
		print('Warping...')
		_run_plm_command("warp", self.option, self._warp_keys, self.log_file)
		print('...done!')





class xfconvert:
	
	def __init__ (self, log_file=""):
		self.log_file=log_file
	
	option={}
	
	_xfconvert_keys=("dim", "grid-spacing", "input", "nobulk", "origin",\
	"output", "output-type", "spacing")
	
	def run_xfconvert(self):
		_run_plm_command("xf-convert", self.option, self._xfconvert_keys, self.log_file)



class compose:
	
	def __init__ (self, log_file=""):
		self.log_file=log_file
	
	input_files=[]
	
	def run_compose(self):
		
		if len(self.input_files) != 2:
			raise NameError("You must define two input structures!")
		
		if self.log_file == "":
			raise NameError("You must define a log file!")
		else:
			self.log=open(self.log_file, "w")
		
		subprocess.call("plastimatch compose " + str(self.input_files[0]) + " " + str(self.input_files[1]) + " " +  str(self.outfile),
		shell=True, stdout=self.log, stderr=self.log)
        
		
		self.log.close()



####################### PUBLIC CLASSES - END - #########################




#################### UTILITY FUNCTION - START - ########################
############ PRIVATE UTILITY FUNCTION, NOT FOR PUBLIC USE ##############

def _clean_parms (d, t):
	
	return dict((k, v) for k, v in d.items() if k in t)


def _run_plm_command(command_type, command_options, command_keys, command_log_file):
	
	if command_log_file == "":
		log=open(os.devnull, "w")
	else:
		log=open(command_log_file, "w")
	
	subprocess.call("plastimatch "+ command_type + _scan_options(command_options, command_keys),\
	shell=True, stdout=log, stderr=log)
	
	log.close()


def _scan_options (d, t):
	
		d=_clean_parms(d, t)
		
		special_keys=("voxels", "scale", "ab-scale", "stretch", "dim",\
		"grid-spacing", "origin", "spacing")
		
		opt_str=""
		
		for key, value in dict.items(d):
			if value!="Enabled" and value!="Disabled" and key not in special_keys:
				opt_str+=" --"+key+"="+value
			elif key in special_keys:
				opt_str+=" --"+key+"="+'"'+value+'"'
			elif value=="Enabled":
				opt_str+=" --"+key
			elif value == "Disabled":
				pass				
		
		return opt_str


############ PRIVATE UTILITY FUNCTION, NOT FOR PUBLIC USE ##############
##################### UTILITY FUNCTION - END - #########################
