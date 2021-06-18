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
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import smpl
from lib.config import load_config
from lib.net_util import load_network, get_posemap
from lib.model.IGRSDFNet import IGRSDFNet
from lib.model.LBSNet import LBSNet
from lib.data.CapeDataset import CapeDataset_scan

import math

from apps.train_scanimate import gen_mesh2

import logging
logging.basicConfig(level=logging.DEBUG)

def test(opt, test_input_dir):
    cuda = torch.device('cuda:0')

    tmp_dirs = test_input_dir.split('/')

    test_input_basedir = tmp_dirs.pop()
    while test_input_basedir == '':
        test_input_basedir = tmp_dirs.pop()
    opt['data']['test_dir'] = test_input_dir

    exp_name = opt['experiment']['name']
    ckpt_dir = '%s/%s' % (opt['experiment']['ckpt_dir'], exp_name)
    result_dir = '%s_test/%s_test_%s' % (opt['experiment']['result_dir'], exp_name, test_input_basedir)
    os.makedirs(result_dir, exist_ok=True)

    model = smpl.create(opt['data']['smpl_dir'], model_type='smpl_vitruvian',
                         gender=opt['data']['smpl_gender'], use_face_contour=False,
                         ext='npz').to(cuda)

    tmp_dir = opt['data']['data_dir']
    tmp_dir_files = sorted([f for f in os.listdir(tmp_dir) if '.ply' in f])
    customized_minimal_ply = os.path.join(tmp_dir, tmp_dir_files[0])
    test_dataset = CapeDataset_scan(opt['data'], phase='test', smpl=model, 
        customized_minimal_ply=customized_minimal_ply, full_test = True, device=cuda)

    reference_body_vs_test = test_dataset.Tpose_minimal_v
    smpl_vitruvian = model.initiate_vitruvian(device = cuda, body_neutral_v = test_dataset.Tpose_minimal_v)

    test_data_loader = DataLoader(test_dataset,
                                  batch_size=1, shuffle=False,
                                  num_workers=0, pin_memory=False)


    # for now, all the hand, face joints are combined with body joints for smplx
    gt_lbs_smpl = model.lbs_weights[:,:24].clone()
    root_idx = model.parents.cpu().numpy()
    idx_list = list(range(root_idx.shape[0]))
    for i in range(root_idx.shape[0]):
        if i > 23:
            root = idx_list[root_idx[i]]
            gt_lbs_smpl[:,root] += model.lbs_weights[:,i]
            idx_list[i] = root
    gt_lbs_smpl = gt_lbs_smpl[None].permute(0,2,1)

    betas = torch.zeros([1, 10], dtype=torch.float32, device=cuda)
    body_pose = torch.zeros([1, 69], dtype=torch.float32, device=cuda) 
    body_pose[:,2] = math.radians(30) # for vitruvian pose
    body_pose[:,5] = math.radians(-30) # for vitruvian pose
    global_orient = torch.zeros((1, 3), dtype=torch.float32, device=cuda)
    transl = torch.zeros((1, 3), dtype=torch.float32, device=cuda)
    
    # define bounding box
    bbox_smpl = (smpl_vitruvian[0].cpu().numpy().min(0).astype(np.float32), smpl_vitruvian[0].cpu().numpy().max(0).astype(np.float32))
    bbox_center, bbox_size = 0.5 * (bbox_smpl[0] + bbox_smpl[1]), (bbox_smpl[1] - bbox_smpl[0])
    bbox_min = np.stack([bbox_center[0]-0.55*bbox_size[0],bbox_center[1]-0.6*bbox_size[1],bbox_center[2]-1.5*bbox_size[2]], 0).astype(np.float32)
    bbox_max = np.stack([bbox_center[0]+0.55*bbox_size[0],bbox_center[1]+0.6*bbox_size[1],bbox_center[2]+1.5*bbox_size[2]], 0).astype(np.float32)
    
    pose_map = get_posemap(opt['model']['posemap_type'], 24, model.parents, opt['model']['n_traverse'], opt['model']['normalize_posemap'])

    igr_net = IGRSDFNet(opt['model']['igr_net'], bbox_min, bbox_max, pose_map).to(cuda)
    fwd_skin_net = LBSNet(opt['model']['fwd_skin_net'], bbox_min, bbox_max, posed=False).to(cuda)
    inv_skin_net = LBSNet(opt['model']['inv_skin_net'], bbox_min, bbox_max, posed=True).to(cuda)

    lat_vecs_igr = nn.Embedding(1, opt['model']['igr_net']['g_dim']).to(cuda)
    
    if opt['model']['igr_net']['g_dim'] > 0:
        torch.nn.init.constant_(lat_vecs_igr.weight.data, 0.0)

    print(igr_net)
    print(fwd_skin_net)
    print(inv_skin_net)

    # load checkpoints
    ckpt_dict = None
    logging.info("Loading checkpoint from %s" % ckpt_dir)
    if os.path.isfile(os.path.join(ckpt_dir, 'ckpt_latest.pt')):
        logging.info('loading ckpt [%s]'%os.path.join(ckpt_dir, 'ckpt_latest.pt'))
        ckpt_dict = torch.load(os.path.join(ckpt_dir, 'ckpt_latest.pt'))
    else:
        logging.error('error: ckpt does not exist [%s]' % opt['experiment']['ckpt_file'])
        exit()

    if ckpt_dict is not None:
        if 'igr_net' in ckpt_dict:
            load_network(igr_net, ckpt_dict['igr_net'])
        else:
            print("Couldn't find igr_net in checkpoints!")

        if 'fwd_skin_net' in ckpt_dict:
            load_network(fwd_skin_net, ckpt_dict['fwd_skin_net'])
        else:
            print("Couldn't find fwd_skin_net in checkpoints!")

        if 'lat_vecs_igr'in ckpt_dict:
            load_network(lat_vecs_igr, ckpt_dict['lat_vecs_igr'])
        else:
            print("Couldn't find lat_vecs_igr in checkpoints!")

    else:
        logging.error("No checkpoint!")
        exit()        
            
    logging.info('test data size: %d'%len(test_data_loader))

    logging.info('Start test inference')
    igr_net.set_lbsnet(fwd_skin_net)

    gen_mesh2(opt, result_dir, igr_net, fwd_skin_net, lat_vecs_igr, model, smpl_vitruvian, test_data_loader, cuda, 
                    reference_body_v=test_data_loader.dataset.Tpose_minimal_v)

    with open(os.path.join(result_dir, '../', exp_name+'_'+test_input_basedir+'.txt'), 'w') as finish_file:
        finish_file.write('Done!')


def testWrapper(args=None):
    parser = argparse.ArgumentParser(
        description='Test SCANimate.'
    )
    parser.add_argument('--config', '-c', type=str, help='Path to config file.')
    parser.add_argument('--test_dir', '-t', type=str, \
                    required=True,\
                    help='Path to test directory')
    args = parser.parse_args()

    opt = load_config(args.config, 'configs/default.yaml')

    test(opt, args.test_dir)

if __name__ == '__main__':
    testWrapper()