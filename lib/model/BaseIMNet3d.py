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
import copy
import math
import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from .MLP import MLP

from ..net_util import init_net, load_network, get_embedder, init_mlp_siren
from torch.nn import init
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
# import functools

import numpy as np
from ..mesh_util import save_obj_mesh_with_color, save_obj_mesh
from ..geometry import index3d, index3d_custom

class BaseIMNet3d(nn.Module):
    def __init__(self,
                 opt,
                 bbox_min=[-1.0,-1.0,-1.0],
                 bbox_max=[1.0,1.0,1.0]
                 ):
        super(BaseIMNet3d, self).__init__()

        self.body_centric_encoding = False if opt['mlp']['ch_dim'][0] == 3 else True

        self.name = 'base_imnet3d'
        self.opt = copy.deepcopy(opt)

        if opt['use_embed']:
            self.embedder, self.opt['mlp']['ch_dim'][0] = get_embedder(opt['d_size'], input_dims=opt['mlp']['ch_dim'][0])
        else:
            self.embedder = None

        if 'g_dim' in self.opt:
            self.opt['mlp']['ch_dim'][0] += self.opt['g_dim']
        if 'pose_dim' in self.opt:
            self.opt['mlp']['ch_dim'][0] += self.opt['pose_dim'] * 23

        self.mlp = MLP(
            filter_channels=self.opt['mlp']['ch_dim'],
            res_layers=self.opt['mlp']['res_layers'],
            last_op=self.opt['mlp']['last_op'],
            nlactiv=self.opt['mlp']['nlactiv'],
            norm=self.opt['mlp']['norm'])

        init_net(self)

        if self.opt['mlp']['nlactiv'] == 'sin': # SIREN
            self.mlp.apply(init_mlp_siren)

        self.register_buffer('bbox_min', torch.Tensor(bbox_min)[None,:,None])
        self.register_buffer('bbox_max', torch.Tensor(bbox_max)[None,:,None])

        self.feat3d = None
        self.global_feat = None

    def filter(self, feat):
        '''
        Store 3d feature 
        args:
            feat: (B, C, D, H, W)
        '''
        self.feat3d = feat

    def set_global_feat(self, feat):
        self.global_feat = feat

    def query(self, points, calib_tensor=None, bmin=None, bmax=None):
        '''
        Given 3D points, query the network predictions for each point.
        args:
            points: (B, 3, N)
        return:
            (B, C, N)
        '''
        N = points.size(2)

        if bmin is None:
            bmin = self.bbox_min
        if bmax is None:
            bmax = self.bbox_max
        points_nc3d = 2.0 * (points - bmin) / (bmax - bmin) - 1.0 # normalized coordiante
        # points_nc3d = 1.0*points
        if self.feat3d is not None and self.body_centric_encoding:
            point_local_feat = index3d_custom(self.feat3d, points_nc3d)
        else: # not body_centric_encoding
            point_local_feat = points_nc3d

        if self.embedder is not None:
            point_local_feat = self.embedder(point_local_feat.permute(0,2,1)).permute(0,2,1)

        if self.global_feat is not None:
            point_local_feat = torch.cat([point_local_feat, self.global_feat[:,:,None].expand(-1,-1,N)], 1)

        w0 = 30.0 if self.opt['mlp']['nlactiv'] == 'sin' else 1.0

        return self.mlp(w0*point_local_feat)

    # for debug
    def get_point_feat(self, feat, points, custom_index=True, bmin=None, bmax=None):
        if bmin is None:
            bmin = self.bbox_min
        if bmax is None:
            bmax = self.bbox_max

        points_nc3d = 2.0 * (points - bmin) / (bmax - bmin) - 1.0 # normalized coordiante/
        # points_nc3d = 1.0*points/
        if custom_index:
            return index3d_custom(feat, points_nc3d)
        else:
            return index3d(feat, points_nc3d)

    def forward(self, feat, points):
        # Set 3d feature
        self.filter(feat)

        return self.query(points)
