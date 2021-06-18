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
import torch.nn.functional as F

import numpy as np
from .BaseIMNet3d import BaseIMNet3d
from ..net_util import homogenize

class LBSNet(BaseIMNet3d):
    def __init__(self,
                 opt,
                 bbox_min,
                 bbox_max,
                 posed=False
                 ):
        super(LBSNet, self).__init__(opt, bbox_min, bbox_max)
        self.name = 'lbs_net'
        self.source_space = 'posed' if posed else 'cano'

    def get_error(self, res):
        err_dict = {}
        
        errLBS_SMPL = self.opt['lambda_smpl'] * (res['pred_lbs_smpl_%s' % self.source_space]-res['gt_lbs_smpl']).pow(2).mean()
        err_dict['SW-SMPL/%s' % self.source_space[0]] = errLBS_SMPL.item()
        error = errLBS_SMPL

        if ('reference_lbs_scan_%s' % self.source_space) in res:
            errLBS_SCAN = self.opt['lambda_scan'] * (res['pred_lbs_scan_%s' % self.source_space]-res['reference_lbs_scan_%s' % self.source_space]).pow(2).mean()
            err_dict['SW-SCAN/%s' % self.source_space[0]] = errLBS_SCAN.item()
            error += errLBS_SCAN
        if 'pred_smpl_posed' in res and 'gt_smpl_posed' in res:
            errCyc_SMPL = self.opt['lambda_cyc_smpl'] * (res['pred_smpl_posed'] - res['gt_smpl_posed']).abs().mean()
            err_dict['Cy-SMPL'] = errCyc_SMPL.item()
            error += errCyc_SMPL
        if ('tar_edge_%s' % self.source_space) in res and ('src_edge_%s' % self.source_space) in res:
            errEdge = self.opt['lambda_l_edge'] * (res['w_tri'][:,None]*(1.0 - res['src_edge_%s' % self.source_space] / (res['tar_edge_%s' % self.source_space]+1e-8)).abs()).mean()
            err_dict['L-Edge'] = errEdge.item()
            error += errEdge
        if ('pred_lbs_tri_%s' % self.source_space) in res:
            pred_lbs_tri = res['pred_lbs_tri_%s' % self.source_space]
            le1 = (pred_lbs_tri[:,:,:,0] - pred_lbs_tri[:,:,:,1]).abs().sum(1)
            le2 = (pred_lbs_tri[:,:,:,1] - pred_lbs_tri[:,:,:,2]).abs().sum(1)
            le3 = (pred_lbs_tri[:,:,:,2] - pred_lbs_tri[:,:,:,0]).abs().sum(1)
            errEdge = self.opt['lambda_w_edge'] * (res['w_tri'] * (le1 + le2 + le3)).mean()
            err_dict['SW-Edge'] = errEdge.item()
            error += errEdge
        if 'pred_smpl_cano' in res and 'gt_smpl_cano' in res:
            errCyc_SMPL = self.opt['lambda_cyc_smpl'] * (res['pred_smpl_cano'] - res['gt_smpl_cano']).abs().mean()
            if 'Cy(SMPL)' in err_dict:
                err_dict['Cy-SMPL'] += errCyc_SMPL.item()
            else:
                err_dict['Cy-SMPL'] = errCyc_SMPL.item()
            error += errCyc_SMPL
        if 'pred_scan_posed' in res and 'gt_scan_posed' in res:
            errCyc_SCAN = self.opt['lambda_cyc_scan'] * (res['pred_scan_posed'] - res['gt_scan_posed']).abs().mean()
            err_dict['Cy-SCAN'] = errCyc_SCAN.item()
            error += errCyc_SCAN
        if 'pred_lbs_scan_cano' in res and 'pred_lbs_scan_posed' in res:
            errLBS_SCAN = self.opt['lambda_scan'] * (res['pred_lbs_scan_cano']-res['pred_lbs_scan_posed']).pow(2).sum(1).mean()
            err_dict['SW-SCAN-Cy'] = errLBS_SCAN.item()
            error += errLBS_SCAN

        if self.source_space == 'posed' and 'pred_lbs_scan_posed' in res:
            errSparse = self.opt['lambda_sparse'] * (res['pred_lbs_scan_posed'].abs()+1e-12).pow(self.opt['p_val']).sum(1).mean()
            err_dict['Sprs'] = errSparse.item()
            error += errSparse

        if self.global_feat is not None:
            error_lat = self.opt['lambda_lat'] * torch.norm(self.global_feat, dim=1).mean()
            err_dict['z-lbs/%s' % self.source_space[0]] = error_lat.item()
            error += error_lat

        return error, err_dict

    def forward(self, feat, smpl, gt_lbs_smpl=None, scan=None, reference_lbs_scan=None, jT=None, res_posed=None, nml_scan=None, v_tri=None, w_tri=None, bmin=None, bmax=None):
        B = smpl.shape[0]

        if self.body_centric_encoding:
            # set volumetric feature
            self.filter(feat) # In case it is body centric encoding

        pred_lbs_smpl = self.query(smpl, bmin=bmin, bmax=bmax)

        res = {}
        if res_posed is not None:
            res = res_posed
        res['pred_lbs_smpl_%s' % self.source_space] = pred_lbs_smpl

        space_transformed_to = 'cano' if self.source_space == 'posed' else 'posed'
        if jT is not None:
            pred_vT = torch.einsum('bjst,bjv->bvst', jT, pred_lbs_smpl)
            pred_vT[:,:,3,3] = 1.0
            if self.source_space == 'posed':
                pred_vT = torch.inverse(pred_vT.reshape(-1,4,4)).view(B,-1,4,4)
            smpl_transformed = torch.einsum('bvst,btv->bsv', pred_vT, homogenize(smpl,1))[:,:3,:]
            res['pred_smpl_%s' % space_transformed_to] = smpl_transformed

        if scan is None and gt_lbs_smpl is None:
            return res
        
        res['gt_smpl_%s' % self.source_space] = smpl
        if gt_lbs_smpl is not None:
            res['gt_lbs_smpl'] = gt_lbs_smpl
        if reference_lbs_scan is not None:
            res['reference_lbs_scan_%s' % self.source_space] = reference_lbs_scan
        if scan is not None:
            pred_lbs_scan = self.query(scan, bmin=bmin, bmax=bmax)
            res['pred_lbs_scan_%s' % self.source_space] = pred_lbs_scan
            if jT is not None:
                pred_vT = torch.einsum('bjst,bjv->bvst', jT, pred_lbs_scan)
                pred_vT[:,:,3,3] = 1.0
                if space_transformed_to == 'cano':
                    pred_vT = torch.inverse(pred_vT.reshape(-1,4,4)).view(B,-1,4,4)
                    res['gt_scan_posed'] = scan
                    if nml_scan is not None:
                        nml_T = torch.einsum('bvst,btv->bsv', pred_vT[:,:,:3,:3], nml_scan)
                        nml_T = F.normalize(nml_T, dim=1)
                        res['normal_scan_cano'] = nml_T
                res['pred_scan_%s' % space_transformed_to] = torch.einsum('bvst,btv->bsv', pred_vT, homogenize(scan,1))[:,:3,:]
        if v_tri is not None and jT is not None:
            v_tri_reshape = v_tri.view(B,3,-1,3)
            e1 = torch.norm(v_tri_reshape[:,:,:,0] - v_tri_reshape[:,:,:,1], p=2, dim=1, keepdim=True)
            e2 = torch.norm(v_tri_reshape[:,:,:,1] - v_tri_reshape[:,:,:,2], p=2, dim=1, keepdim=True)
            e3 = torch.norm(v_tri_reshape[:,:,:,2] - v_tri_reshape[:,:,:,0], p=2, dim=1, keepdim=True)
            e = torch.cat([e1,e2,e3], 1)
            res['tar_edge_%s' % self.source_space] = e
            pred_lbs_tri = self.query(v_tri, bmin=bmin, bmax=bmax)
            pred_vT = torch.einsum('bjst,bjv->bvst', jT, pred_lbs_tri)
            pred_vT[:,:,3,3] = 1.0
            if space_transformed_to == 'cano':
                pred_vT = torch.inverse(pred_vT.reshape(-1,4,4)).view(B,-1,4,4)
            pred_tri = torch.einsum('bvst,btv->bsv', pred_vT, homogenize(v_tri,1))[:,:3,:].view(B,3,-1,3)
            E1 = torch.norm(pred_tri[:,:,:,0] - pred_tri[:,:,:,1], p=2, dim=1, keepdim=True)
            E2 = torch.norm(pred_tri[:,:,:,1] - pred_tri[:,:,:,2], p=2, dim=1, keepdim=True)
            E3 = torch.norm(pred_tri[:,:,:,2] - pred_tri[:,:,:,0], p=2, dim=1, keepdim=True)
            E = torch.cat([E1,E2,E3], 1)
            res['src_edge_%s' % self.source_space] = E
            res['pred_lbs_tri_%s' % self.source_space] = pred_lbs_tri.view(B,pred_lbs_tri.shape[1],-1,3)
            if w_tri is not None:
                res['w_tri'] = w_tri

        # get the error
        error, err_dict = self.get_error(res)

        return res, error, err_dict
