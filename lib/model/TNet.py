# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import os
import math
import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from .MLP import MLP

from ..net_util import init_net, load_network, get_embedder, init_mlp_geometric, init_mlp_siren
from torch.nn import init
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

import numpy as np
from ..mesh_util import save_obj_mesh_with_color, save_obj_mesh, scalar_to_color
from ..geometry import index3d, index3d_custom

from ..net_util import get_embedder

class TNet(nn.Module):
    def __init__(self,
                opt = None
                ):
        super(TNet, self).__init__()

        self.name = 'color_net'

        if opt is None:
            opt = {
                    'use_embed': True,
                    'd_size': 5,
                    'mlp':{ 
                            'ch_dim': [3 , 256, 256, 256, 256, 3],
                            'res_layers': [2],
                            'last_op': 'softplus',
                            'nlactiv': 'softplus',
                            'norm': 'weight',
                            'last_op': 'none'
                            },
                    'feature_dim': 512,
                    'pose_dim': 4,
                    'g_dim': 64
                }
        else:
            opt['feature_dim'] = 512
            opt['mlp']['ch_dim'][-1] = 3


        self.opt = opt
        if self.opt['use_embed']:
            _, self.opt['mlp']['ch_dim'][0] = get_embedder(opt['d_size'], input_dims=self.opt['mlp']['ch_dim'][0])

        if 'g_dim' in self.opt:
            self.opt['mlp']['ch_dim'][0] += self.opt['g_dim']

        if 'pose_dim' in self.opt:
            self.opt['mlp']['ch_dim'][0] += self.opt['pose_dim'] * 23

        self.opt['mlp']['ch_dim'][0] += self.opt['feature_dim']

        self.mlp = MLP(
            filter_channels=self.opt['mlp']['ch_dim'],
            res_layers=self.opt['mlp']['res_layers'],
            last_op=self.opt['mlp']['last_op'],
            nlactiv=self.opt['mlp']['nlactiv'],
            norm=self.opt['mlp']['norm'])

        init_net(self)

    def query(self, points_discription, last_layer_feature):
        input_data = torch.cat([points_discription, last_layer_feature], 1)

        return self.mlp(input_data)

    def forward(self, points_discription, last_layer_feature, target_color):
        input_data = torch.cat([points_discription, last_layer_feature], 1)

        pred_color = self.mlp(input_data)

        err_dict = {}

        error_color = nn.L1Loss()(pred_color, target_color)
            
        err_dict['CLR'] = error_color.item()

        return error_color, err_dict
