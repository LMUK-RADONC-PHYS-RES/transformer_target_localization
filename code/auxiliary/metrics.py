#%%
import pystrum.pynd.ndutils as nd

from typing import Union, Optional
import numpy as np
import torch
from monai.metrics.utils import (
    do_metric_reduction,
)
from monai.utils import MetricReduction, convert_data_type

from monai.metrics.metric import CumulativeIterationMetric
#%%

class NegativeJacobianMetric(CumulativeIterationMetric):
    """
    Compute percentage of negative jacobi determinant of dense diplacement field in 2D or 3D. 
    Coded in MONAI metric style.
    """

    def __init__(
        self,
        reduction: Union[MetricReduction, str] = MetricReduction.MEAN,
        get_not_nans: bool = False,
    ) -> None:
        super().__init__()
        self.reduction = reduction
        self.get_not_nans = get_not_nans

    def _compute_tensor(self, y_pred: torch.Tensor,  y: Optional[torch.Tensor] = None):  # type: ignore
        """
        Args:
            y_pred: input displacemment vector fields (2D or 3D)
            y: has to be specified as optional input even though not needed, not sure why (EL)

        Raises:
            ValueError: when `disp` has less than three dimensions.
        """

        if y_pred.dim() < 3:
            raise ValueError("y_pred should have at least three dimensions.")

        return compute_percentage_neg_jacobi_determinant(disp=y_pred)
    

    def aggregate(self, reduction: Union[MetricReduction, str, None] = None):
        """
        Execute reduction logic for the output of `compute_percentage_neg_jacobi_determinant`.

        Args:
            reduction: define mode of reduction to the metrics, will only apply reduction on `not-nan` values,
                available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
                ``"mean_channel"``, ``"sum_channel"``}, default to `self.reduction`. if "none", will not do reduction.

        """
        data = self.get_buffer()
        if not isinstance(data, torch.Tensor):
            raise ValueError("the data to aggregate must be PyTorch Tensor.")

        # do metric reduction
        f, not_nans = do_metric_reduction(data, reduction or self.reduction)
        return (f, not_nans) if self.get_not_nans else f


def compute_percentage_neg_jacobi_determinant(
    disp: Union[np.ndarray, torch.Tensor],
):
    """
    Compute jacobian determinant of a displacement field.
    NB: to compute the spatial gradients, we use np.gradient.
    Parameters:
        disp: 2D or 3D displacement field of size [*vol_shape, nb_dims],
              where vol_shape is of len nb_dims
    Returns:
        jacobian determinant (scalar, torch.Tensor)
    """

    # convert tensor to numpy array on cpu if needed
    if torch.is_tensor(disp):
        disp = disp.detach().cpu().numpy()
    elif isinstance(disp, np.ndarray):
        pass
    else:
        raise Exception('Attention: unknown type!')   
    
    
    batch_size = disp.shape[0]
    percentage_neg_Jdet = np.empty((batch_size, 1)) # as many values as elements in batch
    # loop over all displacements in a batch
    for b in range(batch_size):
        single_disp = disp[b]
        
        # check inputs
        single_disp = single_disp.transpose(1, 2, 0)
        volshape = single_disp.shape[:-1]
        nb_dims = len(volshape)
        assert len(volshape) in (2, 3), 'Displacement field has to be 2D or 3D'

        # compute grid
        grid_lst = nd.volsize2ndgrid(volshape)
        grid = np.stack(grid_lst, len(volshape))

        # compute gradients
        J = np.gradient(single_disp + grid)

        # 3D displacement field
        if nb_dims == 3:
            dx = J[0]
            dy = J[1]
            dz = J[2]

            # compute jacobian components
            Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
            Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
            Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

            Jdet = Jdet0 - Jdet1 + Jdet2
            
        # must be 2D displacement field
        else:  
            dfdx = J[0]
            dfdy = J[1]

            Jdet = dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]
        
        percentage_neg_Jdet[b] = np.nan if np.prod(volshape) == 0 else np.sum(Jdet <= 0)*100 / np.prod(volshape)
    
    return convert_data_type(percentage_neg_Jdet, torch.Tensor)[0]
