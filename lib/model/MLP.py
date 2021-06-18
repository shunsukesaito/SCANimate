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

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..net_util import Mish, Sin

class MLP(nn.Module):
    def __init__(self, filter_channels, res_layers=[], last_op=None, nlactiv='leakyrelu', norm='none'):
        super(MLP, self).__init__()

        self.filters = nn.ModuleList()
        
        if last_op == 'sigmoid':
            self.last_op = nn.Sigmoid()
        elif last_op == 'tanh':
            self.last_op = nn.Tanh()
        elif last_op == 'softmax':
            self.last_op = nn.Softmax(dim=1)
        else:
            self.last_op = None

        self.res_layers = res_layers
        for l in range(0, len(filter_channels) - 1):
            if l in res_layers:
                if norm == 'weight' and l != len(filter_channels) - 2:
                    self.filters.append(
                        nn.utils.weight_norm(nn.Conv1d(
                            filter_channels[l] + filter_channels[0],
                            filter_channels[l + 1],
                            1)))
                else:
                    self.filters.append(
                        nn.Conv1d(
                            filter_channels[l] + filter_channels[0],
                            filter_channels[l + 1],
                            1))
            else:
                if norm == 'weight' and l != len(filter_channels) - 2:
                    self.filters.append(nn.utils.weight_norm(nn.Conv1d(
                        filter_channels[l],
                        filter_channels[l + 1],
                        1)))
                else:
                    self.filters.append(nn.Conv1d(
                        filter_channels[l],
                        filter_channels[l + 1],
                        1))
        
        self.nlactiv = None
        if nlactiv == 'leakyrelu':
            self.nlactiv = nn.LeakyReLU()
        elif nlactiv == 'softplus':
            self.nlactiv = nn.Softplus(beta=100, threshold=20)
        elif nlactiv == 'relu':
            self.nlactiv = nn.ReLU()
        elif nlactiv == 'mish':
            self.nlactiv = Mish()
        elif nlactiv == 'elu':
            self.nlactiv = nn.ELU(0.1)
        elif nlactiv == 'sin':
            self.nlactiv = Sin()

    def forward(self, feature, return_last_layer_feature = False):
        '''
        :param feature: list of [BxC_inxN] tensors of image features
        :param xy: [Bx3xN] tensor of (x,y) coodinates in the image plane
        :return: [BxC_outxN] tensor of features extracted at the coordinates
        '''

        y = feature
        y0 = feature
        last_layer_feature = None
        for i, f in enumerate(self.filters):
            if i in self.res_layers:
                y = f(torch.cat([y, y0], 1))
            else:
                y = f(y)

            if i != len(self.filters) - 1 and self.nlactiv is not None:
                y = self.nlactiv(y)

            if i == len(self.filters) - 2 and return_last_layer_feature:
                last_layer_feature = y.clone()
                last_layer_feature = last_layer_feature.detach()

        if self.last_op:
            y = self.last_op(y)

        if not return_last_layer_feature:
            return y
        else:
            return y, last_layer_feature
