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
# import functools

import numpy as np
from ..mesh_util import save_obj_mesh_with_color, save_obj_mesh, scalar_to_color
from ..geometry import index3d, index3d_custom
from .BaseIMNet3d import BaseIMNet3d

from ..net_util import get_embedder

class IGRSDFNet(BaseIMNet3d):
    def __init__(self,
                 opt,
                 bbox_min,
                 bbox_max,
                 pose_map
                 ):
        super(IGRSDFNet, self).__init__(opt, bbox_min, bbox_max)

        self.name = 'neural_sdf_bpsigr'
        
        if opt['mlp']['nlactiv'] != 'sin': # SIREN
            self.mlp.apply(init_mlp_geometric)

        self.body_centric_encoding = True if opt['mlp']['ch_dim'][0] == 7 else False

        self.lbs_net = None
        self.pose_feat = None

        if opt['learn_posemap']:
            self.register_buffer('pose_map_init', pose_map)
            self.register_parameter('pose_map', nn.Parameter(pose_map))
        else:
            self.register_buffer('pose_map', pose_map)

        self.bbox_regularization = True if not opt['lambda_bbox'] == 0 else False
        self.space_non_zero_regu = True if not opt['lambda_non_zero'] == 0 else False

    def set_lbsnet(self, net):
        self.lbs_net = net

    def set_pose_feat(self, pose_feat):
        self.pose_feat = pose_feat

    def filter(self, feat):
        self.feat3d = feat
        if self.lbs_net is not None:
            self.lbs_net.filter(feat)

    def query(self, points, calib_tensor=None, return_negative=True, update_lbs=False, bmin=None, bmax=None, return_last_layer_feature = False):
        '''
        Given 3D points, query the network predictions for each point.
        args:
            points: (B, 3, N)
        return:
            (B, C, N)
        '''
        N = points.size(2)

        if self.lbs_net is not None:
            self.lbs_net.filter(self.feat3d)
            # NOTE: the first value belongs to root
            lbs = self.lbs_net.query(points, bmin=bmin, bmax=bmax)
            if self.opt['learn_posemap']:
                lbs = torch.einsum('bjv,jl->blv', lbs, F.softmax(self.pose_map,dim=0))
            else:
                lbs = torch.einsum('bjv,jl->blv', lbs, self.pose_map)
            if not update_lbs:
                lbs = lbs.detach()

        if bmin is None:
            bmin = self.bbox_min
        if bmax is None:
            bmax = self.bbox_max

        points_nc3d = 2.0 * (points - bmin) / (bmax - bmin) - 1.0 # normalized coordiante
        points_nc3d = points_nc3d.clamp(min=-1.0, max=1.0)
        # points_nc3d = 1.0 * points

        in_bbox = (points_nc3d[:, 0] >= -1.0) & (points_nc3d[:, 0] <= 1.0) &\
                  (points_nc3d[:, 1] >= -1.0) & (points_nc3d[:, 1] <= 1.0) &\
                  (points_nc3d[:, 2] >= -1.0) & (points_nc3d[:, 2] <= 1.0)
        in_bbox = in_bbox[:,None].float()

        if self.feat3d is not None and self.body_centric_encoding:
            point_local_feat = index3d(self.feat3d, points_nc3d)
        else:
            point_local_feat = points_nc3d
        
        if self.embedder is not None:
            point_local_feat = self.embedder(point_local_feat.permute(0,2,1)).permute(0,2,1)

        if self.global_feat is not None:
            global_feat = self.global_feat[:,:,None].expand(-1,-1,N)
            point_local_feat = torch.cat([point_local_feat, global_feat], 1)

        if self.pose_feat is not None:
            if self.lbs_net is not None:
                pose_feat = self.pose_feat.view(self.pose_feat.size(0),-1,self.opt['pose_dim'],1) * lbs[:,:,None]
                # Use entire feature
                if 'full_pose' in self.opt.keys():
                    if self.opt['full_pose']:
                        pose_feat = self.pose_feat.view(self.pose_feat.size(0),-1,self.opt['pose_dim'],1) * torch.ones_like(lbs[:,:,None])
                pose_feat = pose_feat.reshape(pose_feat.size(0),-1,N)
            else:
                pose_feat = self.pose_feat[:,:,None].expand(-1,-1,N)
            point_local_feat = torch.cat([point_local_feat, pose_feat], 1)

        w0 = 30.0 if self.opt['mlp']['nlactiv'] == 'sin' else 1.0

        if not return_last_layer_feature:
            if return_negative:
                return -in_bbox*self.mlp(w0*point_local_feat)-(1.0-in_bbox)
            else:
                return in_bbox*self.mlp(w0*point_local_feat)+(1.0-in_bbox)
        else:
            if return_negative:
                sdf, last_layer_feature = self.mlp(w0*point_local_feat, return_last_layer_feature = True)
                sdf = -in_bbox*sdf-(1.0-in_bbox)
            else:
                sdf, last_layer_feature = self.mlp(w0*point_local_feat, return_last_layer_feature = True)
                sdf = in_bbox*sdf+(1.0-in_bbox)
            return sdf, last_layer_feature, point_local_feat

    def compute_normal(self, points, normalize=False, return_pred=False, custom_index=False, update_lbs=False, bmin=None, bmax=None):
        '''
        since image sampling operation does not have second order derivative,
        normal can be computed only via finite difference (forward differentiation)
        '''
        N = points.size(2)

        with torch.enable_grad():
            points.requires_grad_()

            if self.lbs_net is not None:
                self.lbs_net.filter(self.feat3d)
                # NOTE: the first value belongs to root
                lbs = self.lbs_net.query(points, bmin=bmin, bmax=bmax)
                if self.opt['learn_posemap']:
                    lbs = torch.einsum('bjv,jl->blv', lbs, F.softmax(self.pose_map,dim=0))
                else:
                    lbs = torch.einsum('bjv,jl->blv', lbs, self.pose_map)                
                if not update_lbs:
                    lbs = lbs.detach()

            if bmin is None:
                bmin = self.bbox_min
            if bmax is None:
                bmax = self.bbox_max
            points_nc3d = 2.0 * (points - bmin) / (bmax - bmin) - 1.0 # normalized coordiante
            points_nc3d = points_nc3d.clamp(min=-1.0, max=1.0)
            # points_nc3d = 1.0 * points

            if self.feat3d is None:
                point_local_feat = points_nc3d
            else:
                if custom_index:
                    point_local_feat = index3d_custom(self.feat3d, points_nc3d) 
                else:
                    point_local_feat = index3d(self.feat3d, points_nc3d)
            
            if not self.body_centric_encoding:
                point_local_feat = points_nc3d

            if self.embedder is not None:
                point_local_feat = self.embedder(point_local_feat.permute(0,2,1)).permute(0,2,1)

            if self.global_feat is not None:
                global_feat = self.global_feat[:,:,None].expand(-1,-1,N)
                point_local_feat = torch.cat([point_local_feat, global_feat], 1)

            if self.pose_feat is not None:
                if self.lbs_net is not None:
                    pose_feat = self.pose_feat.view(self.pose_feat.size(0),-1,self.opt['pose_dim'],1) * lbs[:,:,None]
                    # Use entire feature
                    if 'full_pose' in self.opt.keys():
                        if self.opt['full_pose']:
                            pose_feat = self.pose_feat.view(self.pose_feat.size(0),-1,self.opt['pose_dim'],1) * torch.ones_like(lbs[:,:,None])

                    pose_feat = pose_feat.reshape(pose_feat.size(0),-1,N)
                else:
                    pose_feat = self.pose_feat[:,:,None].expand(-1,-1,N)
                point_local_feat = torch.cat([point_local_feat, pose_feat], 1)

            w0 = 30.0 if self.opt['mlp']['nlactiv'] == 'sin' else 1.0

            pred = self.mlp(w0*point_local_feat)
            normal = autograd.grad(
                    [pred.sum()], [points], 
                    create_graph=True, retain_graph=True, only_inputs=True)[0]
            
            if normalize:
                normal = F.normalize(normal, dim=1, eps=1e-6)

            if return_pred:
                return normal, pred
            else:
                return normal

    def get_error(self, res):
        '''
        based on https://arxiv.org/pdf/2002.10099.pdf
        '''
        err_dict = {}

        error_ls = self.opt['lambda_sdf'] * nn.L1Loss()(res['sdf_surface'], torch.zeros_like(res['sdf_surface']))
        error_nml = self.opt['lambda_nml'] * torch.norm(res['nml_surface'] - res['nml_gt'], p=2, dim=1).mean()
        
        nml_reg = torch.cat((res['nml_surface'], res['nml_igr']), dim=2)
        # error_reg = self.opt['lambda_reg'] * (torch.norm(res['nml_igr'], p=2, dim=1) - 1).mean().pow(2)
        error_reg = self.opt['lambda_reg'] * (torch.norm(nml_reg, p=2, dim=1) - 1).pow(2).mean()
            
        err_dict['LS'] = error_ls.item()
        err_dict['N'] = error_nml.item()
        err_dict['R'] = error_reg.item()
        error = error_ls + error_nml + error_reg

        if self.bbox_regularization:
            error_bbox = self.opt['lambda_bbox'] * F.leaky_relu(res['sdf_bound'], 1e-6, inplace=True).mean()
            err_dict['BB'] = error_bbox.item()
            error += error_bbox

        if self.space_non_zero_regu:
            error_non_zero  = self.opt['lambda_non_zero'] * torch.exp(-100.0*torch.abs(res['sdf_igr'])).mean()
            err_dict['NZ'] = error_non_zero.item()
            error += error_non_zero

        if self.pose_map.requires_grad:
            error_pose_map = self.opt['lambda_pmap'] * (F.softmax(self.pose_map,dim=0)-self.pose_map_init).abs().mean()
            err_dict['PMap'] = error_pose_map.item()
            error += error_pose_map

        if self.global_feat is not None:
            error_lat = self.opt['lambda_lat'] * torch.norm(self.global_feat, dim=1).mean()
            err_dict['z-sdf'] = error_lat.item()
            error += error_lat

        return error, err_dict


    def forward(self, feat, pts_surface, pts_body, pts_bbox, normals, bmin=None, bmax=None):
        '''
        args:
            feat: (B, C, D, H, W)
            pts_surface: (B, 3, N)
            pts_body: (B, 3, N*)
            pts_bbox: (B, 3, N**)
            normals: (B, 3, N)
        '''
        # set volumetric feature
        self.filter(feat)
        nml_surface, sdf_surface = self.compute_normal(points=pts_surface, 
                                                        normalize=False,
                                                        return_pred=True,
                                                        custom_index=True,
                                                        update_lbs=True,
                                                        bmin=bmin, bmax=bmax)
        if self.bbox_regularization:
            with torch.no_grad():
                bbox_xmin = pts_bbox[:,:,:self.opt['n_bound']].clone()
                bbox_xmin[:, 0] = self.bbox_min[0,0,0]
                bbox_ymin = pts_bbox[:,:,:self.opt['n_bound']].clone()
                bbox_ymin[:, 1] = self.bbox_min[0,1,0]
                bbox_zmin = pts_bbox[:,:,:self.opt['n_bound']].clone()
                bbox_zmin[:, 2] = self.bbox_min[0,2,0]
                bbox_xmax = pts_bbox[:,:,:self.opt['n_bound']].clone()
                bbox_xmax[:, 0] = self.bbox_max[0,0,0]
                bbox_ymax = pts_bbox[:,:,:self.opt['n_bound']].clone()
                bbox_ymax[:, 1] = self.bbox_max[0,1,0]
                bbox_zmax = pts_bbox[:,:,:self.opt['n_bound']].clone()
                bbox_zmax[:, 2] = self.bbox_max[0,2,0]

                pts_bound = torch.cat([bbox_xmin, bbox_ymin, bbox_zmin, bbox_xmax, bbox_ymax, bbox_zmax],-1)

            sdf_bound = self.query(pts_bound, bmin=bmin, bmax=bmax)

        pts_igr = torch.cat([pts_body, pts_bbox], 2)
        nml_igr, sdf_igr = self.compute_normal(points=pts_igr,
                                               normalize=False,
                                               return_pred=True,
                                               custom_index=True,
                                               bmin=bmin, bmax=bmax)

        res = {'sdf_surface': sdf_surface, 'sdf_igr': sdf_igr[:,:,pts_body.shape[2]:], 'nml_surface': nml_surface, 
                'nml_igr': nml_igr, 'nml_gt': normals}

        if self.bbox_regularization:
            res['sdf_bound'] = sdf_bound
        
        # get the error
        error, err_dict = self.get_error(res)

        return res, error, err_dict
