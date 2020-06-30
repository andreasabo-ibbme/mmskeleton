import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from mmskeleton.ops.st_gcn import ConvTemporalGraphical, Graph
from .st_gcn_aaai18_ordinal_smaller_2_encoder import ST_GCN_18_ordinal_smaller_2_encoder

class ST_GCN_18_ordinal_smaller_2_supcon(nn.Module):
    r"""Spatial temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_cfg (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self,
                 in_channels,
                 num_class,
                 graph_cfg,
                 data_bn=True,
                 head='linear',
                 feat_dim=32,
                 **kwargs):
        super().__init__()
        print('In ST_GCN_18 ordinal supcon: ', graph_cfg)
        print(**kwargs)
        self.encoder = ST_GCN_18_ordinal_smaller_2_encoder(self,
                 in_channels,
                 num_class,
                 graph_cfg,
                 data_bn=True,
                 **kwargs)
      
        # fcn for prediction
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )

    def forward(self, x):
        feat = self.encoder(x)
        feat = F.normalize(self.head(feat), dim=1)
        return feat