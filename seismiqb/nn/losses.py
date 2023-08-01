""" Loss functions for seismic interpretation tasks. """
import torch
from torch import nn
import torch.nn.functional as F



class DepthSoftmax(nn.Module):
    """ Softmax activation for depth dimension.

    Parameters
    ----------
    width : int
        The predicted horizon width. Default is 3.
    """
    def __init__(self, width=3):
        super().__init__()
        self.width_weights = torch.ones((1, 1, 1, width))

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x):
        """ Forward pass. """
        x = torch.nn.functional.softmax(x, dim=-1)
        width_weights = self.width_weights.to(device=x.device, dtype=x.dtype)
        x = F.conv2d(x, width_weights, padding=(0, 1))
        return x.float()
