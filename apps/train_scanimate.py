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
import argparse
import numpy as np
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import trimesh
import json

import smpl
from lib.config import load_config
from lib.geo_util import compute_normal_v
from lib.mesh_util import reconstruction, save_obj_mesh, save_obj_mesh_with_color, scalar_to_color, replace_hands_feet, replace_hands_feet_mesh
from lib.net_util import batch_rod2quat, homogenize, load_network, get_posemap, compute_knn_feat
from lib.model.IGRSDFNet import IGRSDFNet
from lib.model.LBSNet import LBSNet
from lib.data.CapeDataset import CapeDataset_scan

import logging
logging.basicConfig(level=logging.DEBUG)

def gen_mesh1(opt, result_dir, fwd_skin_net, inv_skin_net, lat_vecs_inv_skin, 
            model, smpl_vitruvian, train_data_loader, 
            cuda, name='', reference_body_v=None, every_n_frame=10):

    dataset = train_data_loader.dataset
    smpl_face = torch.LongTensor(model.faces[:,[0,2,1]].astype(np.int32))[None].to(cuda)

    def process(data, idx=0):
        frame_names = data['frame_name']
        betas = data['betas'][None].to(device=cuda)
        body_pose = data['body_pose'][None].to(device=cuda)
        scan_posed = data['scan_posed'][None].to(device=cuda)
        transl = data['transl'][None].to(device=cuda)
        f_ids = torch.LongTensor([data['f_id']]).to(device=cuda)
        faces = data['faces'].numpy()
        global_orient = body_pose[:,:3]
        body_pose = body_pose[:,3:]

        if not reference_body_v == None:
            output = model(betas=betas, body_pose=body_pose, global_orient=0*global_orient, transl=0*transl, return_verts=True, custom_out=True,
                            body_neutral_v = reference_body_v.expand(body_pose.shape[0], -1, -1))
        else:
            output = model(betas=betas, body_pose=body_pose, global_orient=0*global_orient, transl=0*transl, return_verts=True, custom_out=True)
        smpl_posed_joints = output.joints
        rootT = model.get_root_T(global_orient, transl, smpl_posed_joints[:,0:1,:])

        smpl_neutral = output.v_shaped
        smpl_cano = output.v_posed
        smpl_posed = output.vertices.contiguous()
        bmax = smpl_posed.max(1)[0]
        bmin = smpl_posed.min(1)[0]
        offset = 0.2*(bmax - bmin)
        bmax += offset
        bmin -= offset
        jT = output.joint_transform[:,:24]
        smpl_n_posed = compute_normal_v(smpl_posed, smpl_face.expand(smpl_posed.shape[0],-1,-1))

        scan_posed = torch.einsum('bst,bvt->bsv', torch.inverse(rootT), homogenize(scan_posed))[:,:3,:] # remove root transform

        if name == '_pt2':
            # save_obj_mesh('%s/%ssmpl_posed%s%s.obj' % (result_dir, frame_names, str(idx).zfill(4), name), smpl_posed[0].cpu().numpy(), model.faces[:,[0,2,1]])
            # save_obj_mesh('%s/%ssmpl_cano%d%s.obj' % (result_dir, frame_names, idx, name), smpl_cano[0].cpu().numpy(), model.faces[:,[0,2,1]])
            save_obj_mesh('%s/%sscan_posed_gth%s%s.obj' % (result_dir, frame_names, str(idx).zfill(4), name), scan_posed[0].t().cpu().numpy(), faces)


        if inv_skin_net.opt['g_dim'] > 0:
            lat = lat_vecs_inv_skin(f_ids) # (B, Z)
            inv_skin_net.set_global_feat(lat)
        feat3d_posed = None
        res_scan_p = inv_skin_net(feat3d_posed, scan_posed, jT=jT, bmin=bmin[:,:,None], bmax=bmax[:,:,None])
        pred_scan_cano = res_scan_p['pred_smpl_cano'].permute(0,2,1)

        # for i in range(24):
        #     c_lbs = scalar_to_color(res_scan_p['pred_lbs_smpl_posed'][0,i,:].cpu().numpy(),min=0,max=1)
            # save_obj_mesh_with_color('%s/scan_lbs%d-%d%s.obj'%(result_dir, idx, i, name), scan_posed[0].t().cpu().numpy(), faces, c_lbs)

        res_smpl_p = inv_skin_net(feat3d_posed, smpl_posed.permute(0,2,1), jT=jT, bmin=bmin[:,:,None], bmax=bmax[:,:,None])
        pred_smpl_cano = res_smpl_p['pred_smpl_cano'].permute(0,2,1)

        # save_obj_mesh('%s/%spred_smpl_cano%s%s.obj' % (result_dir, frame_names, str(idx).zfill(4), name), pred_smpl_cano[0].cpu().numpy(), model.faces[:,[0,2,1]])

        if name=='_pt3':
            scan_faces, scan_mask = dataset.get_raw_scan_face_and_mask(frame_id = f_ids[0].cpu().numpy())
            valid_scan_faces = scan_faces[scan_mask,:]
            pred_scan_cano_mesh = trimesh.Trimesh(vertices = pred_scan_cano[0].cpu().numpy(), faces = valid_scan_faces[:,[0,2,1]], process=False)
            pred_body_neutral_mesh = trimesh.Trimesh(vertices = smpl_neutral[0].cpu().numpy(), faces = model.faces[:,[0,2,1]], process=False)
            output_mesh = replace_hands_feet_mesh(pred_scan_cano_mesh, pred_body_neutral_mesh, vitruvian_angle = model.vitruvian_angle)
            save_obj_mesh('%s/%s_scan_cano%s%s.obj' % (result_dir, frame_names, str(idx).zfill(4), name), pred_scan_cano_mesh.vertices, pred_scan_cano_mesh.faces)
            
        # for i in range(24):
        #     c_lbs = scalar_to_color(res_smpl_p['pred_lbs_smpl_posed'][0,i,:].cpu().numpy(),min=0,max=1)
            # save_obj_mesh_with_color('%s/smpl_lbs%d-%d%s.obj'%(result_dir, idx, i, name), smpl_posed[0].cpu().numpy(), model.faces[:,[0,2,1]], c_lbs)
        
        feat3d_cano = None
        pred_scan_reposed = fwd_skin_net(feat3d_cano, pred_scan_cano.permute(0,2,1), jT=jT)['pred_smpl_posed'].permute(0,2,1)

        pred_scan_reposed = torch.einsum('bst,bvt->bvs', rootT, homogenize(pred_scan_reposed))[0,:,:3]
        
        save_obj_mesh('%s/%spred_scan_reposed%s%s.obj' % (result_dir, frame_names, str(idx).zfill(4), name), pred_scan_reposed.cpu().numpy(), faces)

    if name=='_pt3':
        logging.info("Outputing samples of canonicalization results...")
        with torch.no_grad():
            for i in tqdm(range(len(dataset))):
                if not i % every_n_frame == 0:
                    continue
                data = dataset[i]
                process(data, i)
   
def gen_mesh2(opt, result_dir, igr_net, fwd_skin_net, lat_vecs_igr,
                model, smpl_vitruvian, test_data_loader, cuda, 
                reference_body_v=None, every_n_frame = 1):
    bbox_min = igr_net.bbox_min.squeeze().cpu().numpy()
    bbox_max = igr_net.bbox_max.squeeze().cpu().numpy()

    logging.info("Outputing predicted reconstructions...")
    with torch.no_grad():
        torch.cuda.empty_cache()
        for test_idx, test_data in enumerate(tqdm(test_data_loader)):
            if not test_idx % every_n_frame == 0:
                continue
            frame_names = test_data['frame_name']
            betas = test_data['betas'].to(device=cuda)
            body_pose = test_data['body_pose'].to(device=cuda)
            sub_ids = test_data['sub_id'].to(device=cuda)
            transl = test_data['transl'].to(device=cuda)
            global_orient = body_pose[:,:3]
            body_pose = body_pose[:,3:]
            if not reference_body_v == None:
                output = model(betas=betas, body_pose=body_pose, global_orient=0*global_orient, transl=0*transl, return_verts=True, custom_out=True,
                    body_neutral_v = reference_body_v.expand(body_pose.shape[0], -1, -1))
            else:
                output = model(betas=betas, body_pose=body_pose, global_orient=0*global_orient, transl=0*transl, return_verts=True, custom_out=True)

            smpl_neutral = output.v_shaped
            jT = output.joint_transform[:,:24]
            
            if igr_net.opt['g_dim'] > 0:
                lat = lat_vecs_igr(sub_ids) # (B, Z)
                igr_net.set_global_feat(lat)

            pose_feat = batch_rod2quat(body_pose.reshape(-1, 3)).view(betas.shape[0], -1, igr_net.opt['pose_dim'])
            igr_net.set_pose_feat(pose_feat)

            # try:
            verts, faces, _, _ = reconstruction(igr_net, cuda, torch.eye(4)[None].to(cuda), opt['experiment']['vol_res'], bbox_min, bbox_max, use_octree=True, thresh=0.0)
            # save_obj_mesh('%s/%s_cano%s.obj' % (result_dir, frame_names[0], str(test_idx).zfill(4)), verts, faces)

            verts_torch = torch.Tensor(verts)[None].to(cuda)
            feat3d = None
            res = fwd_skin_net(feat3d, verts_torch.permute(0,2,1), jT=jT)
            pred_lbs = res['pred_lbs_smpl_cano'].permute(0,2,1)
            pred_scan_posed = res['pred_smpl_posed'].permute(0,2,1)

            rootT = test_data['rootT'].cuda()
            pred_scan_posed = torch.einsum('bst,bvt->bvs', rootT, homogenize(pred_scan_posed))[0,:,:3]
            pred_scan_posed = pred_scan_posed.cpu().numpy()
            save_obj_mesh('%s/%s_posed%s.obj' % (result_dir, frame_names[0], str(test_idx).zfill(4)), pred_scan_posed, faces)

def pretrain_skinning_net(opt, result_dir, fwd_skin_net, inv_skin_net, lat_vecs_inv_skin, 
                        model, smpl_vitruvian, gt_lbs_smpl, 
                        train_data_loader, test_data_loader, 
                        cuda, reference_body_v=None):
    optimizer_lbs_c = torch.optim.Adam(fwd_skin_net.parameters(), lr=opt['training']['lr_pt1'])
    optimizer_lbs_p = torch.optim.Adam([
            {
                "params": inv_skin_net.parameters(),
                "lr": opt['training']['lr_pt1'],
            },
            {
                "params": lat_vecs_inv_skin.parameters(),
                "lr": opt['training']['lr_pt1'],
            },
            ])
    smpl_face = torch.LongTensor(model.faces[:,[0,2,1]].astype(np.int32))[None].to(cuda)

    n_iter = 0
    for epoch in range(opt['training']['num_epoch_pt1']):
        fwd_skin_net.train()
        inv_skin_net.train()

        if epoch%opt['training']['resample_every_n_epoch'] == 0:
            train_data_loader.dataset.resample_flag = True
        else:
            train_data_loader.dataset.resample_flag = False

        if epoch == opt['training']['num_epoch_pt1']//2 or epoch == 3*(opt['training']['num_epoch_pt1']//4):
            for j, _ in enumerate(optimizer_lbs_c.param_groups):
                optimizer_lbs_c.param_groups[j]['lr'] *= 0.1
            for j, _ in enumerate(optimizer_lbs_p.param_groups):
                optimizer_lbs_p.param_groups[j]['lr'] *= 0.1
        for train_idx, train_data in enumerate(train_data_loader):
            betas = train_data['betas'].to(device=cuda)
            body_pose = train_data['body_pose'].to(device=cuda)
            scan_posed = train_data['scan_cano_uni'].to(device=cuda)
            scan_tri = train_data['scan_tri_posed'].to(device=cuda)
            transl = train_data['transl'].to(device=cuda)
            f_ids = train_data['f_id'].to(device=cuda)
            smpl_data = train_data['smpl_data']
            global_orient = body_pose[:,:3]
            body_pose = body_pose[:,3:]

            smpl_neutral = smpl_data['smpl_neutral'].cuda()
            smpl_cano = smpl_data['smpl_cano'].cuda()
            smpl_posed = smpl_data['smpl_posed'].cuda()
            smpl_n_posed = smpl_data['smpl_n_posed'].cuda()
            bmax = smpl_data['bmax'].cuda()
            bmin = smpl_data['bmin'].cuda()
            jT = smpl_data['jT'].cuda()
            inv_rootT = smpl_data['inv_rootT'].cuda()

            # Get rid of global rotation from posed scans
            scan_posed = torch.einsum('bst,bvt->bvs', inv_rootT, homogenize(scan_posed))[:,:,:3]
            
            reference_lbs_scan = compute_knn_feat(scan_posed, smpl_posed, gt_lbs_smpl.expand(scan_posed.shape[0],-1,-1).permute(0,2,1))[:,:,0].permute(0,2,1)
            scan_posed = scan_posed.permute(0,2,1)

            if opt['model']['inv_skin_net']['g_dim'] > 0:
                lat = lat_vecs_inv_skin(f_ids) # (B, Z)
                inv_skin_net.set_global_feat(lat)

            feat3d_posed = None
            res_lbs_p, err_lbs_p, err_dict = inv_skin_net(feat3d_posed, smpl_posed.permute(0,2,1), gt_lbs_smpl, scan=scan_posed, reference_lbs_scan=None, jT=jT, bmin=bmin[:,:,None], bmax=bmax[:,:,None])#jT=jT, v_tri=scan_tri, 
            
            feat3d_cano = None
            res_lbs_c, err_lbs_c, err_dict_lbs_c = fwd_skin_net(feat3d_cano, smpl_cano, gt_lbs_smpl, scan=res_lbs_p['pred_scan_cano'].detach(), reference_lbs_scan=reference_lbs_scan)#, jT=jT, res_posed=res_lbs_p)
            
            # Back propagation
            err_dict.update(err_dict_lbs_c)
            err_dict['All-inv'] = err_lbs_p.item()
            err_dict['All-lbs'] = err_lbs_c.item()

            optimizer_lbs_p.zero_grad()
            err_lbs_p.backward()
            optimizer_lbs_p.step()

            optimizer_lbs_c.zero_grad()
            err_lbs_c.backward()
            optimizer_lbs_c.step()

            if n_iter % opt['training']['freq_plot'] == 0:
                err_txt = ''.join(['{}: {:.3f} '.format(k, v) for k,v in err_dict.items()])
                print('[%03d/%03d]:[%04d/%04d] %s' % (epoch, opt['training']['num_epoch_pt1'], train_idx, len(train_data_loader), err_txt))
            n_iter += 1

    train_data_loader.dataset.is_train = False
    gen_mesh1(opt, result_dir, fwd_skin_net, inv_skin_net, lat_vecs_inv_skin, model, smpl_vitruvian, train_data_loader, cuda, '_pt1', reference_body_v=reference_body_v)
    train_data_loader.dataset.is_train = True

    return optimizer_lbs_c, optimizer_lbs_p

def train_skinning_net(opt, result_dir, fwd_skin_net, inv_skin_net, lat_vecs_inv_skin, 
                        model, smpl_vitruvian, gt_lbs_smpl, 
                        train_data_loader, test_data_loader, 
                        cuda, reference_body_v=None, optimizers=None):
    
    if not optimizers == None:
        optimizer_lbs_c = optimizers[0]
        optimizer_lbs_p = optimizers[1]
    else:
        optimizer = torch.optim.Adam(list(fwd_skin_net.parameters())+list(inv_skin_net.parameters()), lr=opt['training']['lr_pt2'])

    smpl_face = torch.LongTensor(model.faces[:,[0,2,1]].astype(np.int32))[None].to(cuda)
    
    o_cyc_smpl = fwd_skin_net.opt['lambda_cyc_smpl']
    o_cyc_scan = fwd_skin_net.opt['lambda_cyc_scan']
    n_iter = 0
    for epoch in range(opt['training']['num_epoch_pt2']):
        fwd_skin_net.train()
        inv_skin_net.train()
        if epoch%opt['training']['resample_every_n_epoch'] == 0:
            train_data_loader.dataset.resample_flag = True
        else:
            train_data_loader.dataset.resample_flag = False
        if epoch == opt['training']['num_epoch_pt2']//2 or epoch == 3*(opt['training']['num_epoch_pt2']//4):
            fwd_skin_net.opt['lambda_cyc_smpl'] *= 10.0
            fwd_skin_net.opt['lambda_cyc_scan'] *= 10.0
            if not optimizers == None:
                for j, _ in enumerate(optimizer_lbs_c.param_groups):
                    optimizer_lbs_c.param_groups[j]['lr'] *= 0.1
                for j, _ in enumerate(optimizer_lbs_p.param_groups):
                    optimizer_lbs_p.param_groups[j]['lr'] *= 0.1
            else:
                for j, _ in enumerate(optimizer.param_groups):
                    optimizer.param_groups[j]['lr'] *= 0.1
        for train_idx, train_data in enumerate(train_data_loader):
            betas = train_data['betas'].to(device=cuda)
            body_pose = train_data['body_pose'].to(device=cuda)
            # scan_cano = train_data['scan_cano'].to(device=cuda).permute(0,2,1)
            scan_posed = train_data['scan_cano_uni'].to(device=cuda)
            scan_tri = train_data['scan_tri_posed'].to(device=cuda)
            w_tri = train_data['w_tri'].to(device=cuda)
            transl = train_data['transl'].to(device=cuda)
            f_ids = train_data['f_id'].to(device=cuda)
            smpl_data = train_data['smpl_data']
            global_orient = body_pose[:,:3]
            body_pose = body_pose[:,3:]

            smpl_neutral = smpl_data['smpl_neutral'].cuda()
            smpl_cano = smpl_data['smpl_cano'].cuda()
            smpl_posed = smpl_data['smpl_posed'].cuda()
            smpl_n_posed = smpl_data['smpl_n_posed'].cuda()
            bmax = smpl_data['bmax'].cuda()
            bmin = smpl_data['bmin'].cuda()
            jT = smpl_data['jT'].cuda()
            inv_rootT = smpl_data['inv_rootT'].cuda()
            
            scan_posed = torch.einsum('bst,bvt->bsv', inv_rootT, homogenize(scan_posed))[:,:3,:] # remove root transform
            scan_tri = torch.einsum('bst,btv->bsv', inv_rootT, homogenize(scan_tri,1))[:,:3,:] 

            reference_lbs_scan = compute_knn_feat(scan_posed.permute(0,2,1), smpl_posed, gt_lbs_smpl.expand(scan_posed.shape[0],-1,-1).permute(0,2,1))[:,:,0].permute(0,2,1)

            if opt['model']['inv_skin_net']['g_dim'] > 0:
                lat = lat_vecs_inv_skin(f_ids) # (B, Z)
                inv_skin_net.set_global_feat(lat)

            feat3d_posed = None
            res_lbs_p, err_lbs_p, err_dict = inv_skin_net(feat3d_posed, smpl_posed.permute(0,2,1), gt_lbs_smpl, scan_posed, reference_lbs_scan=reference_lbs_scan, jT=jT, v_tri=scan_tri, w_tri=w_tri, bmin=bmin[:,:,None], bmax=bmax[:,:,None])

            feat3d_cano = None
            res_lbs_c, err_lbs_c, err_dict_lbs_c = fwd_skin_net(feat3d_cano, smpl_cano, gt_lbs_smpl, res_lbs_p['pred_scan_cano'], jT=jT, res_posed=res_lbs_p)
            
            # Back propagation
            err_dict.update(err_dict_lbs_c)
            err = err_lbs_p + err_lbs_c
            err_dict['All'] = err.item()

            if not optimizers == None:
                optimizer_lbs_c.zero_grad()
                optimizer_lbs_p.zero_grad()
            else:
                optimizer.zero_grad()
            err.backward()
            if not optimizers == None:
                optimizer_lbs_c.step()
                optimizer_lbs_p.step()
            else:
                optimizer.step()

            if n_iter % opt['training']['freq_plot'] == 0:
                err_txt = ''.join(['{}: {:.3f} '.format(k, v) for k,v in err_dict.items()])
                print('[%03d/%03d]:[%04d/%04d] %s' % (epoch, opt['training']['num_epoch_pt2'], train_idx, len(train_data_loader), err_txt))
            n_iter += 1

    fwd_skin_net.opt['lambda_cyc_smpl'] = o_cyc_smpl
    fwd_skin_net.opt['lambda_cyc_scan'] = o_cyc_scan

    train_data_loader.dataset.is_train = False
    gen_mesh1(opt, result_dir, fwd_skin_net, inv_skin_net, lat_vecs_inv_skin, model, smpl_vitruvian, train_data_loader, cuda, '_pt2', reference_body_v=reference_body_v)
    train_data_loader.dataset.is_train = True

def train_igr(opt, ckpt_dir, result_dir, igr_net, fwd_skin_net, inv_skin_net, 
                lat_vecs_igr, lat_vecs_inv_skin, 
                model, smpl_vitruvian, gt_lbs_smpl, 
                train_data_loader, test_data_loader, 
                cuda, reference_body_v=None, optimizer_state = None, start_epoch = 0):
    
    def set_train():
        fwd_skin_net.train()
        igr_net.train()
        inv_skin_net.train()

    def set_eval():
        fwd_skin_net.eval()
        igr_net.eval()
        inv_skin_net.eval()

    smpl_face = torch.LongTensor(model.faces[:,[0,2,1]].astype(np.int32))[None].to(cuda)

    optimizer = torch.optim.Adam([{
            "params": igr_net.parameters(),
            "lr": opt['training']['lr_sdf']},{
            "params": lat_vecs_igr.parameters(),
            "lr": 0.1*opt['training']['lr_sdf']}])
    if opt['training']['continue_train'] and not optimizer_state == None:
        optimizer.load_state_dict(optimizer_state)
        # for j, _ in enumerate(optimizer.param_groups):
        #         optimizer.param_groups[j]['lr'] *= 0.1

    igr_net.set_lbsnet(fwd_skin_net)

    n_iter = 0 
    max_train_idx = 0
    start_time = time.time()
    current_number_processed_samples = 0

    train_data_loader.dataset.resample_flag = True

    balenced_epochs = opt['training']['num_epoch_sdf']*500//len(train_data_loader.dataset)
    opt['training']['num_epoch_sdf'] = min(balenced_epochs, opt['training']['num_epoch_sdf'])

    for epoch in range(start_epoch, opt['training']['num_epoch_sdf']):
        set_train()
        igr_net.set_lbsnet(fwd_skin_net)
        if epoch == opt['training']['num_epoch_sdf']//4:
            for j, _ in enumerate(optimizer.param_groups):
                optimizer.param_groups[j]['lr'] *= 0.5
        if epoch == opt['training']['num_epoch_sdf']//2 or epoch == 3*(opt['training']['num_epoch_sdf']//4):
            for j, _ in enumerate(optimizer.param_groups):
                optimizer.param_groups[j]['lr'] *= 0.1

        for train_idx, train_data in enumerate(train_data_loader):
            betas = train_data['betas'].to(device=cuda)
            body_pose = train_data['body_pose'].to(device=cuda)
            transl = train_data['transl'].to(device=cuda)
            f_ids = train_data['f_id'].to(device=cuda)
            smpl_data = train_data['smpl_data']

            sub_ids = train_data['sub_id'].to(device=cuda) # Subject specific model, so sub_ids = 0
           
            scan_v_posed = train_data['scan_cano_uni'].to(device=cuda)
            scan_n_posed = train_data['normals_uni'].to(device=cuda)

            global_orient = body_pose[:,:3]
            body_pose = body_pose[:,3:]

            smpl_neutral = smpl_data['smpl_neutral'].cuda()
            smpl_cano = smpl_data['smpl_cano'].cuda()
            smpl_posed = smpl_data['smpl_posed'].cuda()
            smpl_n_posed = smpl_data['smpl_n_posed'].cuda()
            bmax = smpl_data['bmax'].cuda()
            bmin = smpl_data['bmin'].cuda()
            jT = smpl_data['jT'].cuda()
            inv_rootT = smpl_data['inv_rootT'].cuda()

            scan_v_posed = torch.einsum('bst,bvt->bsv', inv_rootT, homogenize(scan_v_posed))[:,:3,:] # remove root transform
            scan_n_posed = torch.einsum('bst,bvt->bsv', inv_rootT[:,:3,:3], scan_n_posed)

            if opt['model']['inv_skin_net']['g_dim'] > 0:
                lat = lat_vecs_inv_skin(f_ids) # (B, Z)
                inv_skin_net.set_global_feat(lat)

            feat3d_posed = None
            res_lbs_p, err_lbs_p, err_dict_lbs_p = inv_skin_net(feat3d_posed, smpl_posed.permute(0,2,1), gt_lbs_smpl, scan_v_posed, jT=jT, nml_scan=scan_n_posed, bmin=bmin[:,:,None], bmax=bmax[:,:,None])
            scan_cano = res_lbs_p['pred_scan_cano']
            normal_cano = res_lbs_p['normal_scan_cano']
            
            if opt['model']['igr_net']['g_dim'] > 0:
                lat = lat_vecs_igr(sub_ids) # (B, Z)
                igr_net.set_global_feat(lat)

            feat3d_cano = None
            smpl_neutral = smpl_neutral.permute(0,2,1)

            body_pose_feat = batch_rod2quat(body_pose.reshape(-1, 3)).view(betas.shape[0], -1, opt['model']['igr_net']['pose_dim'])
            igr_net.set_pose_feat(body_pose_feat)

            scan_verts = scan_cano[:,:,torch.randperm(scan_cano.shape[2]).to(cuda)[:opt['data']['num_sample_scan_igr']]]
            smpl_verts = smpl_neutral[:,:,torch.randperm(smpl_neutral.shape[2]).to(cuda)[:opt['data']['num_sample_smpl_igr']]]
            body_verts = torch.cat([scan_verts, smpl_verts], -1)
            body_rand = body_verts + torch.normal(torch.zeros_like(body_verts), opt['data']['sigma_body'])
            bbox_rand = torch.rand_like(body_rand[:,:,:opt['data']['num_sample_bbox_igr']]) * (igr_net.bbox_max - igr_net.bbox_min) + igr_net.bbox_min

            smpl_cano_n = compute_normal_v(smpl_cano.permute(0,2,1), smpl_face.expand(smpl_posed.shape[0],-1,-1)).permute(0,2,1)
            smpl_neutral_n = compute_normal_v(smpl_neutral.permute(0,2,1), smpl_face.expand(smpl_posed.shape[0],-1,-1)).permute(0,2,1)
            scan_cano, normal_cano = replace_hands_feet(scan_cano, normal_cano, smpl_neutral, smpl_neutral_n, opt['data']['num_sample_surf'], vitruvian_angle = model.vitruvian_angle)

            verts = scan_cano.permute(0,2,1).detach().cpu().numpy()

            res_sdf, err_sdf, err_dict = igr_net.forward(feat3d_cano, scan_cano, body_rand, bbox_rand, normal_cano)  

            err = err_sdf

            err_dict['All'] = err.item()

            optimizer.zero_grad()
            err.backward()
            optimizer.step()

            if (n_iter+1) % opt['training']['freq_plot'] == 0:
                err_txt = ''.join(['{}: {:.3f} '.format(k, v) for k,v in err_dict.items()])
                time_now = time.time()
                duration = time_now-start_time
                current_number_processed_samples += f_ids.shape[0]
                persample_process_time = duration/current_number_processed_samples
                current_number_processed_samples -f_ids.shape[0]
                # print('[%03d/%03d]:[%04d/%04d] %02f FPS, %s' % (epoch, opt['training']['num_epoch_sdf'], train_idx, len(train_data_loader), 1.0/persample_process_time, err_txt))
                epoch_percent = (epoch - start_epoch)/(opt['training']['num_epoch_sdf']-start_epoch)
                if epoch_percent == 0:
                    remaining_time = '..h..m'
                else:
                    remaining_time_in_minute = int(duration/epoch_percent*(1-epoch_percent)/60)
                    remaining_time ='%dh%dm'%(remaining_time_in_minute//60, remaining_time_in_minute%60)

                print('[%03d/%03d]: %02f FPS, %s, %s' % (epoch, opt['training']['num_epoch_sdf'], 1.0/persample_process_time, remaining_time, err_txt))

            if (n_iter+1) % opt['training']['freq_save'] == 0 or (epoch == opt['training']['num_epoch_sdf']-1 and train_idx == max_train_idx):
                ckpt_dict = {
                    'opt': opt,
                    'epoch': epoch,
                    'iter': n_iter,
                    'igr_net': igr_net.state_dict(),
                    'fwd_skin_net': fwd_skin_net.state_dict(),
                    'lat_vecs_igr': lat_vecs_igr.state_dict(),
                    'lat_vecs_inv_skin': lat_vecs_inv_skin.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
                torch.save(ckpt_dict, '%s/ckpt_%s.pt' %(ckpt_dir, str(epoch).zfill(6)))
                torch.save(ckpt_dict, '%s/ckpt_latest.pt' % ckpt_dir)
                
            if (n_iter+1) % opt['training']['freq_mesh'] == 0 or (epoch == opt['training']['num_epoch_sdf']-1 and train_idx == max_train_idx):
                output_every_n_frame = 10 if (n_iter+1) % opt['training']['freq_mesh'] == 0 else 1
                gen_mesh2(opt, result_dir, igr_net, fwd_skin_net, lat_vecs_igr, model, smpl_vitruvian, test_data_loader, cuda, 
                    reference_body_v=test_data_loader.dataset.Tpose_minimal_v, every_n_frame=output_every_n_frame)

            if max_train_idx < train_idx:
                max_train_idx = train_idx
            n_iter += 1
            current_number_processed_samples += f_ids.shape[0]

def train(opt):
    cuda = torch.device('cuda:0')

    exp_name = opt['experiment']['name']
    ckpt_dir = '%s/%s' % (opt['experiment']['ckpt_dir'], exp_name)
    result_dir = '%s/%s' % (opt['experiment']['result_dir'], exp_name)
    log_dir = '%s/%s' % (opt['experiment']['log_dir'], exp_name)

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Backup config into log_dir
    with open(os.path.join(log_dir, 'config.json'), 'w') as config_file:
         config_file.write(json.dumps(opt))

    # Initialize vitruvian SMPL model
    if 'vitruvian_angle' not in opt['data']:
        opt['data']['vitruvian_angle'] = 25

    model = smpl.create(opt['data']['smpl_dir'], model_type='smpl_vitruvian',
                         gender=opt['data']['smpl_gender'], use_face_contour=False,
                         ext='npz').to(cuda)

    # Initialize dataset
    train_dataset = CapeDataset_scan(opt['data'], phase='train', smpl=model, 
                                    device=cuda)
    test_dataset = CapeDataset_scan(opt['data'], phase='test', smpl=model,  
                                    device=cuda)

    reference_body_vs_train = train_dataset.Tpose_minimal_v
    reference_body_vs_test = test_dataset.Tpose_minimal_v

    smpl_vitruvian = model.initiate_vitruvian(device = cuda, 
                                            body_neutral_v = train_dataset.Tpose_minimal_v, 
                                            vitruvian_angle=opt['data']['vitruvian_angle'])


    train_data_loader = DataLoader(train_dataset,
                                   batch_size=opt['training']['batch_size'], shuffle=False,
                                   num_workers=opt['training']['num_threads'], pin_memory=opt['training']['pin_memory'])
    test_data_loader = DataLoader(test_dataset,
                                  batch_size=1, shuffle=False,
                                  num_workers=0, pin_memory=False)


    # All the hand, face joints are glued to body joints for SMPL
    gt_lbs_smpl = model.lbs_weights[:,:24].clone()
    root_idx = model.parents.cpu().numpy()
    idx_list = list(range(root_idx.shape[0]))
    for i in range(root_idx.shape[0]):
        if i > 23:
            root = idx_list[root_idx[i]]
            gt_lbs_smpl[:,root] += model.lbs_weights[:,i]
            idx_list[i] = root
    gt_lbs_smpl = gt_lbs_smpl[None].permute(0,2,1)

    smpl_vitruvian = model.initiate_vitruvian(device = cuda, 
                                            body_neutral_v = train_dataset.Tpose_minimal_v, 
                                            vitruvian_angle=opt['data']['vitruvian_angle'])
    
    # define bounding box
    bbox_smpl = (smpl_vitruvian[0].cpu().numpy().min(0).astype(np.float32), smpl_vitruvian[0].cpu().numpy().max(0).astype(np.float32))
    bbox_center, bbox_size = 0.5 * (bbox_smpl[0] + bbox_smpl[1]), (bbox_smpl[1] - bbox_smpl[0])
    bbox_min = np.stack([bbox_center[0]-0.55*bbox_size[0],bbox_center[1]-0.6*bbox_size[1],bbox_center[2]-1.5*bbox_size[2]], 0).astype(np.float32)
    bbox_max = np.stack([bbox_center[0]+0.55*bbox_size[0],bbox_center[1]+0.6*bbox_size[1],bbox_center[2]+1.5*bbox_size[2]], 0).astype(np.float32)

    # Initialize networks
    pose_map = get_posemap(opt['model']['posemap_type'], 24, model.parents, opt['model']['n_traverse'], opt['model']['normalize_posemap'])
    
    igr_net = IGRSDFNet(opt['model']['igr_net'], bbox_min, bbox_max, pose_map).to(cuda)
    fwd_skin_net = LBSNet(opt['model']['fwd_skin_net'], bbox_min, bbox_max, posed=False).to(cuda)
    inv_skin_net = LBSNet(opt['model']['inv_skin_net'], bbox_min, bbox_max, posed=True).to(cuda)

    lat_vecs_igr = nn.Embedding(1, opt['model']['igr_net']['g_dim']).to(cuda)
    lat_vecs_inv_skin = nn.Embedding(len(train_dataset), opt['model']['inv_skin_net']['g_dim']).to(cuda)

    torch.nn.init.constant_(lat_vecs_igr.weight.data, 0.0)
    torch.nn.init.normal_(lat_vecs_inv_skin.weight.data, 0.0, 1.0 / math.sqrt(opt['model']['inv_skin_net']['g_dim']))

    print("igr_net:\n", igr_net)
    print("fwd_skin_net:\n", fwd_skin_net)
    print("inv_skin_net:\n", inv_skin_net)

    # Find checkpoints
    ckpt_dict = None
    if opt['experiment']['ckpt_file'] is not None:
        if os.path.isfile(opt['experiment']['ckpt_file']):
            logging.info('loading for ckpt...' + opt['experiment']['ckpt_file'])
            ckpt_dict = torch.load(opt['experiment']['ckpt_file'])
        else:
            logging.warning('ckpt does not exist [%s]' % opt['experiment']['ckpt_file'])
    elif opt['training']['continue_train']:
        model_path = '%s/ckpt_latest.pt' % ckpt_dir
        if os.path.isfile(model_path):
            logging.info('Resuming from '+ model_path)
            ckpt_dict = torch.load(model_path)
        else:
            logging.warning('ckpt does not exist [%s]' % model_path)
            opt['training']['use_trained_skin_nets'] = True
            model_path = '%s/ckpt_trained_skin_nets.pt' % ckpt_dir
            if os.path.isfile(model_path):
                logging.info('Resuming from ' + model_path)
                ckpt_dict = torch.load(model_path)
                logging.info('Pretrained model loaded.')
            else:
                logging.warning('ckpt does not exist [%s]' % model_path)
    elif opt['training']['use_trained_skin_nets']:
        model_path = '%s/ckpt_trained_skin_nets.pt' % ckpt_dir
        if os.path.isfile(model_path):
            logging.info('Resuming from ' + model_path)
            ckpt_dict = torch.load(model_path)
            logging.info('Pretrained model loaded.')
        else:
            logging.warning('ckpt does not exist [%s] \n Failed to resume training, start training from beginning' % model_path)

    # Load checkpoints
    train_igr_start_epoch = 0
    if ckpt_dict is not None:
        if 'igr_net' in ckpt_dict:
            load_network(igr_net, ckpt_dict['igr_net'])
            if 'epoch' in ckpt_dict:
                train_igr_start_epoch = ckpt_dict['epoch']
        else:
            logging.warning("Couldn't find igr_net in checkpoints!")

        if 'fwd_skin_net' in ckpt_dict:
            load_network(fwd_skin_net, ckpt_dict['fwd_skin_net'])
        else:
            logging.warning("Couldn't find fwd_skin_net in checkpoints!")

        if 'inv_skin_net' in ckpt_dict:
            load_network(inv_skin_net, ckpt_dict['inv_skin_net'])
        else:
            logging.warning("Couldn't find inv_skin_net in checkpoints!")

        if 'lat_vecs_igr'in ckpt_dict:
            load_network(lat_vecs_igr, ckpt_dict['lat_vecs_igr'])
        else:
            logging.warning("Couldn't find lat_vecs_igr in checkpoints!")

        if 'lat_vecs_inv_skin'in ckpt_dict:
            load_network(lat_vecs_inv_skin, ckpt_dict['lat_vecs_inv_skin'])
        else:
            logging.warning("Couldn't find lat_vecs_inv_skin in checkpoints!")

            
    logging.info('train data size: %s' % str(len(train_dataset)))
    logging.info('test data size: %s' % str(len(test_dataset)))

    # Skip canonicalization
    if opt['training']['continue_train'] and os.path.isfile('%s/ckpt_trained_skin_nets.pt' % ckpt_dir):
        logging.info("Get fwd_skin_net, inv_skin_net and lat_vecs_inv_skin from trained skinning net!")
        trained_skin_nets_ckpt_dict = torch.load('%s/ckpt_trained_skin_nets.pt' % ckpt_dir)
        fwd_skin_net.load_state_dict(trained_skin_nets_ckpt_dict['fwd_skin_net'])
        inv_skin_net.load_state_dict(trained_skin_nets_ckpt_dict['inv_skin_net'])
        lat_vecs_inv_skin.load_state_dict(trained_skin_nets_ckpt_dict['lat_vecs_inv_skin'])

        opt['training']['skip_pt1'] = True
        opt['training']['skip_pt2'] = True

    # Pretrain fwd_skin_net and inv_skin_net net independently
    if not opt['training']['skip_pt1']:
        logging.info('start pretraining skinning nets (individual)')
        optimizers = pretrain_skinning_net(opt, log_dir, fwd_skin_net, inv_skin_net, lat_vecs_inv_skin, model, smpl_vitruvian, gt_lbs_smpl, train_data_loader, test_data_loader, cuda,
                reference_body_v = reference_body_vs_train)

    # Train fwd_skin_net and inv_skin_net jointly
    if not opt['training']['skip_pt2']:
        logging.info('start training skinning nets (joint)')
        train_skinning_net(opt, log_dir, fwd_skin_net, inv_skin_net, lat_vecs_inv_skin, model, smpl_vitruvian, gt_lbs_smpl, train_data_loader, test_data_loader, cuda,
            reference_body_v = reference_body_vs_train, optimizers = optimizers)

    if not opt['training']['skip_pt1'] and not opt['training']['skip_pt2']:
        ckpt_dict = {
            'opt': opt,
            'fwd_skin_net': fwd_skin_net.state_dict(),
            'inv_skin_net': inv_skin_net.state_dict(),
            'lat_vecs_inv_skin': lat_vecs_inv_skin.state_dict()
        }
        torch.save(ckpt_dict, '%s/ckpt_trained_skin_nets.pt' % ckpt_dir)

    # get only valid triangles
    train_data_loader.dataset.compute_valid_tri(inv_skin_net, model, lat_vecs_inv_skin, smpl_vitruvian)

    train_data_loader.dataset.is_train = False
    gen_mesh1(opt, log_dir, fwd_skin_net, inv_skin_net, lat_vecs_inv_skin, model, smpl_vitruvian, train_data_loader, cuda, '_pt3', reference_body_v=train_data_loader.dataset.Tpose_minimal_v, every_n_frame=1)
    train_data_loader.dataset.is_train = True

    logging.info('Start training igr...')
    optimizer_state = None
    if opt['training']['continue_train']:
        try:
            optimizer_state = ckpt_dict['optimizer']
        except:
            pass
    train_igr(opt, ckpt_dir, result_dir, igr_net, fwd_skin_net, inv_skin_net, 
                lat_vecs_igr, lat_vecs_inv_skin, model, smpl_vitruvian, 
                gt_lbs_smpl, train_data_loader, test_data_loader, 
                cuda, reference_body_v = reference_body_vs_train, 
                optimizer_state = optimizer_state, start_epoch = train_igr_start_epoch)

    with open(os.path.join(result_dir, '../', exp_name+'.txt'), 'w') as finish_file:
        finish_file.write('Done!')
    logging.info('Finished learning geometry!')

def trainWrapper(args=None):
    parser = argparse.ArgumentParser(
        description='Train SCANimate.'
    )
    parser.add_argument('--config', '-c', type=str, help='Path to config file.')
    args = parser.parse_args()

    opt = load_config(args.config, 'configs/default.yaml')

    train(opt)

if __name__ == '__main__':
    trainWrapper()