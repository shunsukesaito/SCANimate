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

from lib.data.CapeDataset import CapeDataset_scan_color
import smpl
from torch.utils.data import DataLoader
from lib.config import load_config
from lib.model.IGRSDFNet import IGRSDFNet
from lib.model.LBSNet import LBSNet
from lib.model.TNet import TNet

from lib.geo_util import compute_normal_v

import argparse
import torch
import os
import json
import numpy as np

from lib.net_util import batch_rod2quat,homogenize, load_network, get_posemap
import torch.nn as nn
import math
from lib.mesh_util import replace_hands_feet_wcolor
from lib.mesh_util import reconstruction, save_obj_mesh, save_obj_mesh_with_color, scalar_to_color
import time
import trimesh
from tqdm import tqdm

def gen_train_color_mesh(opt, result_dir, fwd_skin_net, inv_skin_net, lat_vecs_inv_skin, model, smpl_vitruvian, train_data_loader, cuda, name='', reference_body_v=None):
    dataset = train_data_loader.dataset
    smpl_face = torch.LongTensor(model.faces[:,[0,2,1]].astype(np.int32))[None].to(cuda)

    def process(data, idx=0):
        frame_names = data['frame_name']
        scan_color = data['colors'].to(device=cuda)
        betas = data['betas'][None].to(device=cuda)
        body_pose = data['body_pose'][None].to(device=cuda)
        scan_posed = data['scan_posed'][None].to(device=cuda)
        original_colors = data['original_colors']
        transl = data['transl'][None].to(device=cuda)
        f_ids = torch.LongTensor([data['f_id']]).to(device=cuda)
        smpl_data = data['smpl_data']
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

        if inv_skin_net.opt['g_dim'] > 0:
            lat = lat_vecs_inv_skin(f_ids) # (B, Z)
            inv_skin_net.set_global_feat(lat)
        feat3d_posed = None
        res_scan_p = inv_skin_net(feat3d_posed, scan_posed, jT=jT, bmin=bmin[:,:,None], bmax=bmax[:,:,None])
        pred_scan_cano = res_scan_p['pred_smpl_cano'].permute(0,2,1)

        # res_smpl_p = inv_skin_net(feat3d_posed, smpl_posed.permute(0,2,1), jT=jT, bmin=bmin[:,:,None], bmax=bmax[:,:,None])
        # pred_smpl_cano = res_smpl_p['pred_smpl_cano'].permute(0,2,1)
        # save_obj_mesh('%s/pred_smpl_cano%s%s.obj' % (result_dir, str(idx).zfill(4), name), pred_smpl_cano[0].cpu().numpy(), model.faces[:,[0,2,1]])
        if name=='_pt3':
            scan_faces, scan_mask = dataset.get_raw_scan_face_and_mask(frame_id = f_ids[0].cpu().numpy())
            valid_scan_faces = scan_faces[scan_mask,:]
            pred_scan_cano_mesh = trimesh.Trimesh(vertices = pred_scan_cano[0].cpu().numpy(), faces = valid_scan_faces[:,[0,2,1]], vertex_colors = original_colors, process=False)
            save_obj_mesh_with_color('%s/%s_scan_cano_%s.obj' % (result_dir, frame_names, str(idx).zfill(4)),  pred_scan_cano_mesh.vertices, pred_scan_cano_mesh.faces, original_colors)

        feat3d_cano = None
        pred_scan_reposed = fwd_skin_net(feat3d_cano, pred_scan_cano.permute(0,2,1), jT=jT)['pred_smpl_posed'].permute(0,2,1)
        save_obj_mesh('%s/%s_pred_scan_reposed_%s%s.obj' % (result_dir, frame_names, str(idx).zfill(4), name), pred_scan_reposed[0].cpu().numpy(), faces)

    if True:
        with torch.no_grad():
            print("Output canonicalized train meshes...")
            for i in tqdm(range(len(dataset))):
                if not i % 5 == 0:
                    continue
                data = dataset[i]
                process(data, i)

    
def gen_color_mesh(opt, result_dir, igr_net, fwd_skin_net, lat_vecs_igr, texture_net, model, smpl_vitruvian, test_data_loader, cuda, reference_body_v=None, largest_component=False):
    bbox_min = igr_net.bbox_min.squeeze().cpu().numpy()
    bbox_max = igr_net.bbox_max.squeeze().cpu().numpy()

    with torch.no_grad():
        torch.cuda.empty_cache()
        for test_idx, test_data in enumerate(tqdm(test_data_loader)):
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
            # smpl_posed_joints = output.joints
            # rootT = model.get_root_T(global_orient, transl, smpl_posed_joints[:,0:1,:])

            smpl_neutral = output.v_shaped
            jT = output.joint_transform[:,:24]

            if igr_net.opt['g_dim'] > 0:
                lat = lat_vecs_igr(sub_ids) # (B, Z)
                igr_net.set_global_feat(lat)
           
            set_pose_feat = batch_rod2quat(body_pose.reshape(-1, 3)).view(betas.shape[0], -1, 4)
            igr_net.set_pose_feat(set_pose_feat)

            verts, faces, _, _, vcolors = reconstruction(igr_net, cuda, torch.eye(4)[None].to(cuda), opt['experiment']['vol_res'],\
                                    bbox_min, bbox_max, use_octree=True, thresh=0.0,
                                    texture_net = texture_net)
            # save_obj_mesh_with_color('%s/%s_cano%s.obj' % (result_dir, frame_names[0], str(test_idx).zfill(4)), verts, faces, vcolors)

            verts_torch = torch.Tensor(verts)[None].to(cuda)
            feat3d = None
            res = fwd_skin_net(feat3d, verts_torch.permute(0,2,1), jT=jT)
            pred_lbs = res['pred_lbs_smpl_cano'].permute(0,2,1)
            
            pred_scan_posed = res['pred_smpl_posed'].permute(0,2,1)
            rootT = test_data['rootT'].cuda()
            pred_scan_posed = torch.einsum('bst,bvt->bvs', rootT, homogenize(pred_scan_posed))[0,:,:3]
            pred_scan_posed = pred_scan_posed.cpu().numpy()
            save_obj_mesh_with_color('%s/%s_posed%s.obj' % (result_dir, frame_names[0], str(test_idx).zfill(4)), pred_scan_posed, faces, vcolors)


def train_color(opt, ckpt_dir, result_dir, texture_net, igr_net, fwd_skin_net, inv_skin_net, lat_vecs_igr, lat_vecs_inv_skin, 
              model, smpl_vitruvian, gt_lbs_smpl, train_data_loader, test_data_loader, cuda, reference_body_v=None):
    
    fwd_skin_net.eval()
    igr_net.eval()
    inv_skin_net.eval()
    igr_net.set_lbsnet(fwd_skin_net)

    smpl_face = torch.LongTensor(model.faces[:,[0,2,1]].astype(np.int32))[None].to(cuda)

    optimizer = torch.optim.Adam([{
            "params": texture_net.parameters(),
            "lr": opt['training']['lr_sdf']}])

    n_iter = 0 
    max_train_idx = 0
    start_time = time.time()
    current_number_processed_samples = 0

    train_data_loader.dataset.resample_flag = True

    opt['training']['num_epoch_sdf'] = opt['training']['num_epoch_sdf']//4

    for epoch in range(opt['training']['num_epoch_sdf']):
        texture_net.train()
        if epoch == opt['training']['num_epoch_sdf']//2 or epoch == 3*(opt['training']['num_epoch_sdf']//4):
            for j, _ in enumerate(optimizer.param_groups):
                optimizer.param_groups[j]['lr'] *= 0.1
        for train_idx, train_data in enumerate(train_data_loader):
            betas = train_data['betas'].to(device=cuda)
            body_pose = train_data['body_pose'].to(device=cuda)
            sub_ids = train_data['sub_id'].to(device=cuda)
            transl = train_data['transl'].to(device=cuda)
            f_ids = train_data['f_id'].to(device=cuda)
            smpl_data = train_data['smpl_data']

            scan_v_posed = train_data['scan_cano_uni'].to(device=cuda)
            scan_n_posed = train_data['normals_uni'].to(device=cuda)
            scan_color = train_data['colors'].to(device=cuda)
            scan_color = scan_color.permute(0,2,1)

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
            
            with torch.no_grad():
                scan_v_posed = torch.einsum('bst,bvt->bsv', inv_rootT, homogenize(scan_v_posed))[:,:3,:] # remove root transform
                scan_n_posed = torch.einsum('bst,bvt->bsv', inv_rootT[:,:3,:3], scan_n_posed)

                if opt['model']['inv_skin_net']['g_dim'] > 0:
                    lat = lat_vecs_inv_skin(f_ids) # (B, Z)
                    inv_skin_net.set_global_feat(lat)

                feat3d_posed = None
                res_lbs_p, _, _ = inv_skin_net(feat3d_posed, smpl_posed.permute(0,2,1), gt_lbs_smpl, scan_v_posed, jT=jT, nml_scan=scan_n_posed, bmin=bmin[:,:,None], bmax=bmax[:,:,None])
                scan_cano = res_lbs_p['pred_scan_cano']
                normal_cano = res_lbs_p['normal_scan_cano']

                if opt['model']['igr_net']['g_dim'] > 0:
                    lat = lat_vecs_igr(sub_ids) # (B, Z)
                    # print("subid", sub_ids)
                    igr_net.set_global_feat(lat)

                smpl_neutral = smpl_neutral.permute(0,2,1)

                set_pose_feat = batch_rod2quat(body_pose.reshape(-1, 3)).view(betas.shape[0], -1, 4)
                igr_net.set_pose_feat(set_pose_feat)

                pts0 = scan_cano[0].detach().cpu().permute(1,0).numpy()
                clr0 = scan_color[0].detach().cpu().permute(1,0).numpy()
                clr0 = train_data['colors'][0,:,:].detach().cpu().numpy()

                scan_cano, scan_color = replace_hands_feet_wcolor(scan_cano, 
                                                    scan_color, smpl_neutral, 
                                                    opt['data']['num_sample_surf'], 
                                                    vitruvian_angle = model.vitruvian_angle)

                sdf, last_layer_feature, point_local_feat = igr_net.query(scan_cano, return_last_layer_feature=True)  
                

            err, err_dict = texture_net(point_local_feat, last_layer_feature, scan_color)

            err_dict['All'] = err.item()

            optimizer.zero_grad()
            err.backward()
            optimizer.step()

            if n_iter % opt['training']['freq_plot'] == 0:
                err_txt = ''.join(['{}: {:.3f} '.format(k, v) for k,v in err_dict.items()])
                time_now = time.time()
                duration = time_now-start_time
                current_number_processed_samples += f_ids.shape[0]
                persample_process_time = duration/current_number_processed_samples
                current_number_processed_samples = -f_ids.shape[0]
                print('[%03d/%03d]:[%04d/%04d] %02f FPS, %s' % (epoch, opt['training']['num_epoch_sdf'], 
                    train_idx, len(train_data_loader), 1.0/persample_process_time, err_txt))
                start_time = time.time()

            if (n_iter+1) % 200 == 0 or (epoch == opt['training']['num_epoch_sdf']-1 and train_idx == max_train_idx):
                ckpt_dict = {
                    'opt': opt,
                    'epoch': epoch,
                    'iter': n_iter,
                    'igr_net': igr_net.state_dict(),
                    'fwd_skin_net': fwd_skin_net.state_dict(),
                    'lat_vecs_igr': lat_vecs_igr.state_dict(),
                    'lat_vecs_inv_skin': lat_vecs_inv_skin.state_dict(),
                    'texture_net': texture_net.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
                torch.save(ckpt_dict, '%s/ckpt_color_latest.pt' % ckpt_dir)
                if (n_iter+1) % 1000 == 0:
                    torch.save(ckpt_dict, '%s/ckpt_color_epoch%d.pt' % (ckpt_dir, epoch))

            if n_iter == 0:
                train_data_loader.dataset.is_train = False
                texture_net.eval()
                gen_train_color_mesh(opt, result_dir, fwd_skin_net, inv_skin_net, lat_vecs_inv_skin, model, smpl_vitruvian, train_data_loader, cuda, '_pt3', reference_body_v=train_data_loader.dataset.Tpose_minimal_v)
                train_data_loader.dataset.is_train = True
           
            if (n_iter+1) % opt['training']['freq_mesh'] == 0 or (epoch == opt['training']['num_epoch_sdf']-1 and train_idx == max_train_idx):
                texture_net.eval()
                gen_color_mesh(opt, result_dir, igr_net, fwd_skin_net, lat_vecs_igr, texture_net, model, smpl_vitruvian, test_data_loader, cuda, 
                    reference_body_v=test_data_loader.dataset.Tpose_minimal_v)

            if max_train_idx < train_idx:
                max_train_idx = train_idx
            n_iter += 1
            current_number_processed_samples += f_ids.shape[0]

def train(opt):
    cuda = torch.device('cuda:0')

    exp_name = opt['experiment']['name']
    ckpt_dir = '%s/%s' % (opt['experiment']['ckpt_dir'], exp_name)
    result_dir = '%s/%s' % (opt['experiment']['result_dir']+'_color', exp_name+'_color')
    log_dir = '%s/%s' % (opt['experiment']['log_dir'], exp_name)

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Backup config into log_dir
    with open(os.path.join(log_dir, 'config.json'), 'w') as config_file:
         config_file.write(json.dumps(opt))

    # load checkpoints
    ckpt_dict = None
    if opt['experiment']['ckpt_file'] is not None:
        if os.path.isfile(opt['experiment']['ckpt_file']):
            print('loading for ckpt...', opt['experiment']['ckpt_file'])
            ckpt_dict = torch.load(opt['experiment']['ckpt_file'])
        else:
            print('error: ckpt does not exist [%s]' % opt['experiment']['ckpt_file'])
    elif opt['training']['continue_train']:
        # if opt['training']['resume_epoch'] < 0:
        model_path = '%s/ckpt_latest.pt' % ckpt_dir
        # else:
        #     model_path = '%s/ckpt_epoch_%d.pt' % (ckpt_dir, opt['training']['resume_epoch'])
        if os.path.isfile(model_path):
            print('Resuming from ', model_path)
            ckpt_dict = torch.load(model_path)
        else:
            print('error: ckpt does not exist [%s]' % model_path)
    elif opt['training']['use_pretrain']:
        model_path = '%s/ckpt_pretrain.pt' % ckpt_dir
        if os.path.isfile(model_path):
            print('Resuming from ', model_path)
            ckpt_dict = torch.load(model_path)
            print('Pretrained model loaded.')
        else:
            print('error: ckpt does not exist [%s]' % model_path)


    model = smpl.create(opt['data']['smpl_dir'], model_type='smpl_vitruvian',
                         gender=opt['data']['smpl_gender'], use_face_contour=False,
                         ext='npz').to(cuda)


    train_dataset = CapeDataset_scan_color(opt['data'], phase='train', smpl=model)
    test_dataset = CapeDataset_scan_color(opt['data'], phase='test', smpl=model, full_test = True)

    reference_body_vs_train = train_dataset.Tpose_minimal_v
    reference_body_vs_test = test_dataset.Tpose_minimal_v

    smpl_vitruvian = model.initiate_vitruvian(device = cuda, body_neutral_v = train_dataset.Tpose_minimal_v)


    train_data_loader = DataLoader(train_dataset,
                                   batch_size=8, shuffle=True,#not opt['training']['serial_batch'],
                                   num_workers=16, pin_memory=opt['training']['pin_memory'])
    test_data_loader = DataLoader(test_dataset,
                                  batch_size=1, shuffle=False,
                                  num_workers=0, pin_memory=False)


    # for now, all the hand, face joints are combined with body joints for smpl
    gt_lbs_smpl = model.lbs_weights[:,:24].clone()
    root_idx = model.parents.cpu().numpy()
    idx_list = list(range(root_idx.shape[0]))
    for i in range(root_idx.shape[0]):
        if i > 23:
            root = idx_list[root_idx[i]]
            gt_lbs_smpl[:,root] += model.lbs_weights[:,i]
            idx_list[i] = root
    gt_lbs_smpl = gt_lbs_smpl[None].permute(0,2,1)

    smpl_vitruvian = model.initiate_vitruvian(device = cuda, body_neutral_v = train_dataset.Tpose_minimal_v)
    
    # define bounding box
    bbox_smpl = (smpl_vitruvian[0].cpu().numpy().min(0).astype(np.float32), smpl_vitruvian[0].cpu().numpy().max(0).astype(np.float32))
    bbox_center, bbox_size = 0.5 * (bbox_smpl[0] + bbox_smpl[1]), (bbox_smpl[1] - bbox_smpl[0])
    bbox_min = np.stack([bbox_center[0]-0.55*bbox_size[0],bbox_center[1]-0.6*bbox_size[1],bbox_center[2]-1.5*bbox_size[2]], 0).astype(np.float32)
    bbox_max = np.stack([bbox_center[0]+0.55*bbox_size[0],bbox_center[1]+0.6*bbox_size[1],bbox_center[2]+1.5*bbox_size[2]], 0).astype(np.float32)

    pose_map = get_posemap(opt['model']['posemap_type'], 24, model.parents, opt['model']['n_traverse'], opt['model']['normalize_posemap'])
    
    igr_net = IGRSDFNet(opt['model']['igr_net'], bbox_min, bbox_max, pose_map).to(cuda)
    fwd_skin_net = LBSNet(opt['model']['fwd_skin_net'], bbox_min, bbox_max, posed=False).to(cuda)
    inv_skin_net = LBSNet(opt['model']['inv_skin_net'], bbox_min, bbox_max, posed=True).to(cuda)
    texture_net = TNet(opt['model']['igr_net']).to(cuda)

    lat_vecs_igr = nn.Embedding(1, opt['model']['igr_net']['g_dim']).to(cuda)
    lat_vecs_inv_skin = nn.Embedding(len(train_dataset), opt['model']['inv_skin_net']['g_dim']).to(cuda)

    if opt['model']['igr_net']['g_dim'] > 0:
        torch.nn.init.constant_(lat_vecs_igr.weight.data, 0.0)
        #torch.nn.init.normal_(lat_vecs_igr.weight.data, 0.0, 1.0 / math.sqrt(opt['model']['igr_net']['g_dim']))

    if opt['model']['inv_skin_net']['g_dim'] > 0:
        torch.nn.init.normal_(lat_vecs_inv_skin.weight.data, 0.0, 1.0 / math.sqrt(opt['model']['inv_skin_net']['g_dim']))

    print(igr_net)
    print(fwd_skin_net)
    print(inv_skin_net)
    print(texture_net)

    if ckpt_dict is not None:
        if 'igr_net' in ckpt_dict:
            load_network(igr_net, ckpt_dict['igr_net'])
        else:
            print("Couldn't find igr_net in checkpoints!")

        if 'fwd_skin_net' in ckpt_dict:
            load_network(fwd_skin_net, ckpt_dict['fwd_skin_net'])
        else:
            print("Couldn't find fwd_skin_net in checkpoints!")

        if 'inv_skin_net' in ckpt_dict:
            load_network(inv_skin_net, ckpt_dict['inv_skin_net'])
        else:
            print("Couldn't find inv_skin_net in checkpoints!")
            print("Try to find pretrained model...")
            model_path = '%s/ckpt_trained_skin_nets.pt' % ckpt_dir
            if os.path.isfile(model_path):
                print('Sucessfully found pretrained model of inv_skin_net: ', model_path)
                pretrained_ckpt_dict = torch.load(model_path)
                fwd_skin_net.load_state_dict(pretrained_ckpt_dict['fwd_skin_net'])
                inv_skin_net.load_state_dict(pretrained_ckpt_dict['inv_skin_net'])
                lat_vecs_inv_skin.load_state_dict(pretrained_ckpt_dict['lat_vecs_inv_skin'])
                # load_network(inv_skin_net, pretrained_ckpt_dict['inv_skin_net'])
            else:
                print("No pretrained model has been found")
                exit()

        if 'lat_vecs_igr'in ckpt_dict:
            load_network(lat_vecs_igr, ckpt_dict['lat_vecs_igr'])
        else:
            print("Couldn't find lat_vecs_igr in checkpoints!")

        if 'lat_vecs_inv_skin'in ckpt_dict:
            load_network(lat_vecs_inv_skin, ckpt_dict['lat_vecs_inv_skin'])
        else:
            print("Couldn't find lat_vecs_inv_skin in checkpoints!")
            
    
    print('train data size: ', len(train_data_loader))
    print('test data size: ', len(test_data_loader))


    # get only valid triangles
    print('Computing valid triangles...')
    train_data_loader.dataset.compute_valid_tri(inv_skin_net, model, lat_vecs_inv_skin, smpl_vitruvian)

    # Train color module
    print('Start training color module!')
    
    train_color(opt, ckpt_dir, result_dir, texture_net, igr_net, fwd_skin_net, inv_skin_net, lat_vecs_igr, lat_vecs_inv_skin, model, 
              smpl_vitruvian, gt_lbs_smpl, train_data_loader, test_data_loader, cuda, reference_body_v = reference_body_vs_train)
    
    with open(os.path.join(result_dir, '../', exp_name+'.txt'), 'w') as finish_file:
        finish_file.write('Done!')

def trainWrapper(args=None):
    parser = argparse.ArgumentParser(
        description='Train SCANimate color.'
    )
    parser.add_argument('--config', '-c', type=str, help='Path to config file.')
    args = parser.parse_args()

    opt = load_config(args.config, 'configs/default.yaml')

    train(opt)

if __name__ == '__main__':
    trainWrapper()