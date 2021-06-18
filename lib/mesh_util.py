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

import math
from skimage import measure
import numpy as np
import torch
from .sdf import create_grid, eval_grid_octree, eval_grid
import trimesh
import tqdm
import matplotlib
import matplotlib.cm as cm

import torch.nn.functional as F

def replace_hands_feet(cano_scan_points, cano_scan_normals, cano_body_points, cano_body_normals, n, vitruvian_angle = 0):
    cano_points = torch.zeros_like(cano_scan_points)
    cano_normals = torch.zeros_like(cano_scan_normals)

    left_hand_vertex_index = 2005 # palm center
    right_hand_vertex_index = 5509 # palm center
    left_foot_vertex_index = 3392 # ankle, ~5 cm above the ground
    right_foot_vertex_index = 6730 # ankle, ~5 cm above the ground

    cut_offset = 0.03

    if vitruvian_angle == 0:
        for batch_index in range(cano_scan_points.shape[0]):
            left_hand_x = cano_body_points[batch_index,0,left_hand_vertex_index]
            right_hand_x = cano_body_points[batch_index,0, right_hand_vertex_index]
            feet_y = 0.5*(cano_body_points[batch_index,1,left_foot_vertex_index] + cano_body_points[batch_index,1,right_foot_vertex_index])

            cano_scan_mask = (cano_scan_points[batch_index,0,:] < left_hand_x) &\
                            (cano_scan_points[batch_index,0,:] > right_hand_x) &\
                            (cano_scan_points[batch_index,1,:] > feet_y)

            body_hands_feet_mask = (cano_body_points[batch_index,0,:] > left_hand_x-cut_offset) |\
                            (cano_body_points[batch_index,0,:] < right_hand_x+cut_offset) |\
                            (cano_body_points[batch_index,1,:] < feet_y+cut_offset)

            n_b = body_hands_feet_mask.sum().detach().cpu().numpy()
            n_s = n-n_b

            selected_cano_scan_points = cano_scan_points[batch_index, :, cano_scan_mask]
            selected_cano_scan_normals = cano_scan_normals[batch_index, :, cano_scan_mask]

            selected_cano_body_points = cano_body_points[batch_index, :, body_hands_feet_mask]
            selected_cano_body_normals = cano_body_normals[batch_index, :, body_hands_feet_mask]


            n_s_current = selected_cano_scan_points.shape[1]
            if n_s_current >= n_s:
                random_index = torch.randperm(n_s_current).cuda()[:n_s]
                selected_cano_scan_points = selected_cano_scan_points[:, random_index]
                selected_cano_scan_normals = selected_cano_scan_normals[:, random_index]
            else:
                repeated_times = n_s//n_s_current
                random_index = torch.randperm(n_s_current).cuda()[:n_s-repeated_times*n_s_current]
                selected_cano_scan_points = torch.cat((selected_cano_scan_points.repeat(1,repeated_times), selected_cano_scan_points[:,random_index]), axis = 1)
                selected_cano_scan_normals = torch.cat((selected_cano_scan_normals.repeat(1,repeated_times), selected_cano_scan_normals[:,random_index]), axis = 1)

            cano_points[batch_index, :,:] = torch.cat((selected_cano_body_points, selected_cano_scan_points), axis = 1)
            cano_normals[batch_index, :,:] = torch.cat((-selected_cano_body_normals, selected_cano_scan_normals), axis = 1) 
    else:
        vitruvian_angle_radians = math.radians(vitruvian_angle)
        for batch_index in range(cano_scan_points.shape[0]):
            left_hand_x = cano_body_points[batch_index,0,left_hand_vertex_index]
            right_hand_x = cano_body_points[batch_index,0, right_hand_vertex_index]
            left_foot_rotated_y = cano_body_points[batch_index,1,left_foot_vertex_index]*math.cos(-vitruvian_angle_radians) +\
                                cano_body_points[batch_index,0,left_foot_vertex_index]*math.sin(-vitruvian_angle_radians) 
            right_foot_rotated_y = cano_body_points[batch_index,1,right_foot_vertex_index]*math.cos(vitruvian_angle_radians) +\
                                cano_body_points[batch_index,0,right_foot_vertex_index]*math.sin(vitruvian_angle_radians) 
            
            cano_scan_mask = (cano_scan_points[batch_index,0,:] < left_hand_x) &\
                            (cano_scan_points[batch_index,0,:] > right_hand_x) &\
                            (cano_scan_points[batch_index,1,:]*math.cos(-vitruvian_angle_radians) +\
                                cano_scan_points[batch_index,0,:]*math.sin(-vitruvian_angle_radians) > left_foot_rotated_y) &\
                            (cano_scan_points[batch_index,1,:]*math.cos(vitruvian_angle_radians) +\
                                cano_scan_points[batch_index,0,:]*math.sin(vitruvian_angle_radians) > right_foot_rotated_y)

            body_hands_feet_mask = (cano_body_points[batch_index,0,:] > left_hand_x-cut_offset) |\
                            (cano_body_points[batch_index,0,:] < right_hand_x+cut_offset) |\
                            (cano_body_points[batch_index,1,:]*math.cos(-vitruvian_angle_radians) +\
                                cano_body_points[batch_index,0,:]*math.sin(-vitruvian_angle_radians) < left_foot_rotated_y+cut_offset) |\
                            (cano_body_points[batch_index,1,:]*math.cos(vitruvian_angle_radians) +\
                                cano_body_points[batch_index,0,:]*math.sin(vitruvian_angle_radians) < right_foot_rotated_y+cut_offset)

            n_b = body_hands_feet_mask.sum().detach().cpu().numpy()
            n_s = n-n_b

            selected_cano_scan_points = cano_scan_points[batch_index, :, cano_scan_mask]
            selected_cano_scan_normals = cano_scan_normals[batch_index, :, cano_scan_mask]

            selected_cano_body_points = cano_body_points[batch_index, :, body_hands_feet_mask]
            selected_cano_body_normals = cano_body_normals[batch_index, :, body_hands_feet_mask]


            n_s_current = selected_cano_scan_points.shape[1]
            if n_s_current >= n_s:
                random_index = torch.randperm(n_s_current).cuda()[:n_s]
                selected_cano_scan_points = selected_cano_scan_points[:, random_index]
                selected_cano_scan_normals = selected_cano_scan_normals[:, random_index]
            else:
                repeated_times = n_s//n_s_current
                random_index = torch.randperm(n_s_current).cuda()[:n_s-repeated_times*n_s_current]
                selected_cano_scan_points = torch.cat((selected_cano_scan_points.repeat(1,repeated_times), selected_cano_scan_points[:,random_index]), axis = 1)
                selected_cano_scan_normals = torch.cat((selected_cano_scan_normals.repeat(1,repeated_times), selected_cano_scan_normals[:,random_index]), axis = 1)

            cano_points[batch_index, :,:] = torch.cat((selected_cano_body_points, selected_cano_scan_points), axis = 1)
            cano_normals[batch_index, :,:] = torch.cat((-selected_cano_body_normals, selected_cano_scan_normals), axis = 1) 

    return cano_points, cano_normals

def replace_hands_feet_wcolor(cano_scan_points, cano_scan_colors, cano_body_points, n, cano_body_color = np.array([119, 90,70]), vitruvian_angle = 0):
    cano_points = torch.zeros_like(cano_scan_points)
    cano_colors = torch.zeros_like(cano_scan_colors)

    cano_body_colors = torch.zeros_like(cano_body_points)
    cano_body_colors[:,0,:] += cano_body_color[0]/255.0
    cano_body_colors[:,1,:] += cano_body_color[1]/255.0
    cano_body_colors[:,2,:] += cano_body_color[2]/255.0

    left_hand_vertex_index = 2005 # palm center
    right_hand_vertex_index = 5509 # palm center
    left_foot_vertex_index = 3392 # ankle, ~5 cm above the ground
    right_foot_vertex_index = 6730 # ankle, ~5 cm above the ground

    cut_offset = 0.03

    if vitruvian_angle == 0:
        for batch_index in range(cano_scan_points.shape[0]):
            left_hand_x = cano_body_points[batch_index,0,left_hand_vertex_index]
            right_hand_x = cano_body_points[batch_index,0, right_hand_vertex_index]
            feet_y = 0.5*(cano_body_points[batch_index,1,left_foot_vertex_index] + cano_body_points[batch_index,1,right_foot_vertex_index])

            cano_scan_mask = (cano_scan_points[batch_index,0,:] < left_hand_x) &\
                            (cano_scan_points[batch_index,0,:] > right_hand_x) &\
                            (cano_scan_points[batch_index,1,:] > feet_y)

            body_hands_feet_mask = (cano_body_points[batch_index,0,:] > left_hand_x-cut_offset) |\
                            (cano_body_points[batch_index,0,:] < right_hand_x+cut_offset) |\
                            (cano_body_points[batch_index,1,:] < feet_y+cut_offset)

            n_b = body_hands_feet_mask.sum().detach().cpu().numpy()
            n_s = n-n_b

            selected_cano_scan_points = cano_scan_points[batch_index, :, cano_scan_mask]
            selected_cano_scan_colors = cano_scan_colors[batch_index, :, cano_scan_mask]

            selected_cano_body_points = cano_body_points[batch_index, :, body_hands_feet_mask]
            selected_cano_body_colors = cano_body_colors[batch_index, :, body_hands_feet_mask]


            n_s_current = selected_cano_scan_points.shape[1]
            if n_s_current >= n_s:
                random_index = torch.randperm(n_s_current).cuda()[:n_s]
                selected_cano_scan_points = selected_cano_scan_points[:, random_index]
                selected_cano_scan_colors = selected_cano_scan_colors[:, random_index]
            else:
                repeated_times = n_s//n_s_current
                random_index = torch.randperm(n_s_current).cuda()[:n_s-repeated_times*n_s_current]
                selected_cano_scan_points = torch.cat((selected_cano_scan_points.repeat(1,repeated_times), selected_cano_scan_points[:,random_index]), axis = 1)
                selected_cano_scan_colors = torch.cat((selected_cano_scan_colors.repeat(1,repeated_times), selected_cano_scan_normals[:,random_index]), axis = 1)

            cano_points[batch_index, :,:] = torch.cat((selected_cano_body_points, selected_cano_scan_points), axis = 1)
            cano_colors[batch_index, :,:] = torch.cat((selected_cano_body_colors, selected_cano_scan_colors), axis = 1) 
    else:
        vitruvian_angle_radians = math.radians(vitruvian_angle)
        for batch_index in range(cano_scan_points.shape[0]):
            left_hand_x = cano_body_points[batch_index,0,left_hand_vertex_index]
            right_hand_x = cano_body_points[batch_index,0, right_hand_vertex_index]
            left_foot_rotated_y = cano_body_points[batch_index,1,left_foot_vertex_index]*math.cos(-vitruvian_angle_radians) +\
                                cano_body_points[batch_index,0,left_foot_vertex_index]*math.sin(-vitruvian_angle_radians) 
            right_foot_rotated_y = cano_body_points[batch_index,1,right_foot_vertex_index]*math.cos(vitruvian_angle_radians) +\
                                cano_body_points[batch_index,0,right_foot_vertex_index]*math.sin(vitruvian_angle_radians) 
            
            cano_scan_mask = (cano_scan_points[batch_index,0,:] < left_hand_x) &\
                            (cano_scan_points[batch_index,0,:] > right_hand_x) &\
                            (cano_scan_points[batch_index,1,:]*math.cos(-vitruvian_angle_radians) +\
                                cano_scan_points[batch_index,0,:]*math.sin(-vitruvian_angle_radians) > left_foot_rotated_y) &\
                            (cano_scan_points[batch_index,1,:]*math.cos(vitruvian_angle_radians) +\
                                cano_scan_points[batch_index,0,:]*math.sin(vitruvian_angle_radians) > right_foot_rotated_y)

            body_hands_feet_mask = (cano_body_points[batch_index,0,:] > left_hand_x-cut_offset) |\
                            (cano_body_points[batch_index,0,:] < right_hand_x+cut_offset) |\
                            (cano_body_points[batch_index,1,:]*math.cos(-vitruvian_angle_radians) +\
                                cano_body_points[batch_index,0,:]*math.sin(-vitruvian_angle_radians) < left_foot_rotated_y+cut_offset) |\
                            (cano_body_points[batch_index,1,:]*math.cos(vitruvian_angle_radians) +\
                                cano_body_points[batch_index,0,:]*math.sin(vitruvian_angle_radians) < right_foot_rotated_y+cut_offset)

            n_b = body_hands_feet_mask.sum().detach().cpu().numpy()
            n_s = n-n_b

            selected_cano_scan_points = cano_scan_points[batch_index, :, cano_scan_mask]
            selected_cano_scan_colors = cano_scan_colors[batch_index, :, cano_scan_mask]

            selected_cano_body_points = cano_body_points[batch_index, :, body_hands_feet_mask]
            selected_cano_body_colors = cano_body_colors[batch_index, :, body_hands_feet_mask]


            n_s_current = selected_cano_scan_points.shape[1]
            if n_s_current >= n_s:
                random_index = torch.randperm(n_s_current).cuda()[:n_s]
                selected_cano_scan_points = selected_cano_scan_points[:, random_index]
                selected_cano_scan_colors = selected_cano_scan_colors[:, random_index]
            else:
                repeated_times = n_s//n_s_current
                random_index = torch.randperm(n_s_current).cuda()[:n_s-repeated_times*n_s_current]
                selected_cano_scan_points = torch.cat((selected_cano_scan_points.repeat(1,repeated_times), selected_cano_scan_points[:,random_index]), axis = 1)
                selected_cano_scan_colors = torch.cat((selected_cano_scan_colors.repeat(1,repeated_times), selected_cano_scan_colors[:,random_index]), axis = 1)

            cano_points[batch_index, :,:] = torch.cat((selected_cano_body_points, selected_cano_scan_points), axis = 1)
            cano_colors[batch_index, :,:] = torch.cat((selected_cano_body_colors, selected_cano_scan_colors), axis = 1) 

    return cano_points, cano_colors

def replace_hands_feet_mesh(cano_scan_mesh, cano_body_mesh, vitruvian_angle = 0):

    # cano_scan_mesh and cano_body_mesh both should be trimesh object

    left_hand_vertex_index = 2005 # palm center
    right_hand_vertex_index = 5509 # palm center
    left_foot_vertex_index = 3392 # ankle, ~5 cm above the ground
    right_foot_vertex_index = 6730 # ankle, ~5 cm above the ground

    cut_offset = 0.05

    if vitruvian_angle == 0:
        left_hand_mesh = trimesh.intersections.slice_mesh_plane(cano_body_mesh, (1,0,0), cano_body_mesh.vertices[left_hand_vertex_index,:]-np.array([cut_offset,0,0]))
        right_hand_mesh = trimesh.intersections.slice_mesh_plane(cano_body_mesh, (-1,0,0), cano_body_mesh.vertices[right_hand_vertex_index,:]+np.array([cut_offset,0,0]))
        feet_mesh = trimesh.intersections.slice_mesh_plane(cano_body_mesh, (0,-1,0), 0.5*(cano_body_mesh.vertices[left_foot_vertex_index,:]+cano_body_mesh.vertices[right_foot_vertex_index, :])+np.array([0,cut_offset,0]))

        cano_scan_new = trimesh.intersections.slice_mesh_plane(cano_scan_mesh, (-1,0,0), cano_body_mesh.vertices[left_hand_vertex_index,:])
        cano_scan_new = trimesh.intersections.slice_mesh_plane(cano_scan_new, (1,0,0), cano_body_mesh.vertices[right_hand_vertex_index,:])
        cano_scan_new = trimesh.intersections.slice_mesh_plane(cano_scan_new, (0,1,0), 0.5*(cano_body_mesh.vertices[left_foot_vertex_index,:]+cano_body_mesh.vertices[right_foot_vertex_index, :]))

    else:
        vitruvian_angle_radians = math.radians(vitruvian_angle)
        sinv = math.sin(vitruvian_angle_radians)
        cosv = math.cos(vitruvian_angle_radians)

        left_hand_mesh = trimesh.intersections.slice_mesh_plane(cano_body_mesh, (1,0,0), cano_body_mesh.vertices[left_hand_vertex_index,:]-np.array([cut_offset,0,0]))
        right_hand_mesh = trimesh.intersections.slice_mesh_plane(cano_body_mesh, (-1,0,0), cano_body_mesh.vertices[right_hand_vertex_index,:]+np.array([cut_offset,0,0]))
        left_foot_mesh = trimesh.intersections.slice_mesh_plane(cano_body_mesh, (sinv,-cosv,0), 
                                                                cano_body_mesh.vertices[left_foot_vertex_index,:]+np.array([-sinv*cut_offset,cosv*cut_offset,0]))
        right_foot_mesh = trimesh.intersections.slice_mesh_plane(cano_body_mesh, (-sinv,-cosv,0), 
                                                                cano_body_mesh.vertices[right_foot_vertex_index,:]+np.array([sinv*cut_offset,cosv*cut_offset,0]))
        feet_mesh = trimesh.Trimesh(vertices = np.vstack((left_foot_mesh.vertices, right_foot_mesh.vertices)),
                                    faces = np.vstack((left_foot_mesh.faces, right_foot_mesh.faces+left_foot_mesh.vertices.shape[0])),
                                    process=False)

        cano_scan_new = trimesh.intersections.slice_mesh_plane(cano_scan_mesh, (-1,0,0), cano_body_mesh.vertices[left_hand_vertex_index,:])
        cano_scan_new = trimesh.intersections.slice_mesh_plane(cano_scan_new, (1,0,0), cano_body_mesh.vertices[right_hand_vertex_index,:])
        cano_scan_new = trimesh.intersections.slice_mesh_plane(cano_scan_new, (-sinv,cosv,0), cano_body_mesh.vertices[left_foot_vertex_index,:])
        cano_scan_new = trimesh.intersections.slice_mesh_plane(cano_scan_new, (sinv,cosv,0), cano_body_mesh.vertices[right_foot_vertex_index,:])

    f0 = cano_scan_new.vertices.shape[0]
    f1 = f0+left_hand_mesh.vertices.shape[0]
    f2 = f1+right_hand_mesh.vertices.shape[0]

    cano_scan_new.vertices = np.vstack((cano_scan_new.vertices, left_hand_mesh.vertices, right_hand_mesh.vertices, feet_mesh.vertices))
    cano_scan_new.faces = np.vstack((cano_scan_new.faces, left_hand_mesh.faces+f0, right_hand_mesh.faces+f1, feet_mesh.faces+f2))

    # cano_scan_new.vertices = np.vstack((left_hand_mesh.vertices, right_hand_mesh.vertices, feet_mesh.vertices))
    # cano_scan_new.faces = np.vstack((left_hand_mesh.faces, right_hand_mesh.faces+left_hand_mesh.vertices.shape[0], feet_mesh.faces+left_hand_mesh.vertices.shape[0]+right_hand_mesh.vertices.shape[0]))

    # import ipdb
    # ipdb.set_trace()
    
    return cano_scan_new

def build_smooth_conv3D(in_channels=1, out_channels=1, kernel_size=3, padding=1):
    smooth_conv = torch.nn.Conv3d(
        in_channels=in_channels, out_channels=out_channels, 
        kernel_size=kernel_size, padding=padding
    )
    smooth_conv.weight.data = torch.ones(
        (kernel_size, kernel_size, kernel_size), 
        dtype=torch.float32
    ).reshape(in_channels, out_channels, kernel_size, kernel_size, kernel_size) / (kernel_size**3)
    smooth_conv.bias.data = torch.zeros(out_channels)
    return smooth_conv

def reconstruction(net, cuda, calib_tensor,
                   resolution, b_min, b_max,
                   use_octree=False, num_samples=10000, transform=None, thresh=0.5, texture_net = None):
    '''
    Reconstruct meshes from sdf predicted by the network.
    :param net: a BasePixImpNet object. call image filter beforehead.
    :param cuda: cuda device
    :param calib_tensor: calibration tensor
    :param resolution: resolution of the grid cell
    :param b_min: bounding box corner [x_min, y_min, z_min]
    :param b_max: bounding box corner [x_max, y_max, z_max]
    :param use_octree: whether to use octree acceleration
    :param num_samples: how many points to query each gpu iteration
    :return: marching cubes results.
    '''
    # Then we define the lambda function for cell evaluation
    color_flag = False if texture_net is None else True

    def eval_func(points):
        samples = points.unsqueeze(0)
        pred = net.query(samples, calib_tensor)[0][0]
        return pred

    def batch_eval(points, num_samples=4096):
        num_pts = points.shape[1]
        sdf = []
        num_batches = num_pts // num_samples
        for i in range(num_batches):
            sdf.append(
                eval_func(points[:, i * num_samples:i * num_samples + num_samples])
            )
        if num_pts % num_samples:
            sdf.append(
                eval_func(points[:, num_batches * num_samples:])
            )
        if num_pts == 0:
            return None
        sdf = torch.cat(sdf)
        return sdf

    # Then we evaluate the grid    
    max_level = int(math.log2(resolution))
    sdf = eval_progressive(batch_eval, 4, max_level, cuda, b_min, b_max, thresh)

    # calculate matrix
    mat = np.eye(4)
    length = b_max - b_min
    mat[0, 0] = length[0] / sdf.shape[0]
    mat[1, 1] = length[1] / sdf.shape[1]
    mat[2, 2] = length[2] / sdf.shape[2]
    mat[0:3, 3] = b_min

    # Finally we do marching cubes
    try:
        verts, faces, normals, values = measure.marching_cubes(sdf, thresh, gradient_direction='ascent')
    except:
        print('error cannot marching cubes')
        return -1

    # transform verts into world coordinate system
    verts = np.matmul(mat[:3, :3], verts.T) + mat[:3, 3:4]
    verts = verts.T
    if np.linalg.det(mat) > 0:
        faces = faces[:,[0,2,1]]

    if color_flag:
        torch_verts = torch.Tensor(verts).unsqueeze(0).permute(0,2,1).to(cuda)

        with torch.no_grad():
            _, last_layer_feature, point_local_feat = net.query(torch_verts, calib_tensor, return_last_layer_feature=True)
            vertex_colors = texture_net.query(point_local_feat, last_layer_feature)
            vertex_colors = vertex_colors.squeeze(0).permute(1,0).detach().cpu().numpy()
        return verts, faces, normals, values, vertex_colors
    else:
        return verts, faces, normals, values
    
     
def eval_progressive(batch_eval, min_level, max_level, cuda, b_min, b_max, thresh=0.5):
    steps = [i for i in range(min_level, max_level+1)]

    b_min = torch.tensor(b_min).to(cuda)
    b_max = torch.tensor(b_max).to(cuda)

    # init
    smooth_conv3x3 = build_smooth_conv3D(in_channels=1, out_channels=1, kernel_size=3, padding=1).to(cuda)

    arrange = torch.linspace(0, 2**steps[-1], 2**steps[0]+1).long().to(cuda)
    coords = torch.stack(torch.meshgrid([
        arrange, arrange, arrange
    ])) # [3, 2**step+1, 2**step+1, 2**step+1]
    coords = coords.view(3, -1).t() # [N, 3]
    calculated = torch.zeros(
        (2**steps[-1]+1, 2**steps[-1]+1, 2**steps[-1]+1), dtype=torch.bool
    ).to(cuda)
        
    gird8_offsets = torch.stack(torch.meshgrid([
        torch.tensor([-1, 0, 1]), torch.tensor([-1, 0, 1]), torch.tensor([-1, 0, 1])
    ])).int().to(cuda).view(3, -1).t() #[27, 3]

    with torch.no_grad():
        for step in steps:
            resolution = 2**step + 1
            stride = 2**(steps[-1]-step)

            if step == steps[0]:
                coords2D = coords.float() / (2**steps[-1]+1) * (b_max - b_min) + b_min
                sdf_all = batch_eval(
                    coords2D.t(),
                ).view(resolution, resolution, resolution)
                coords_accum = coords / stride
                coords_accum = coords_accum.long()
                calculated[coords[:, 0], coords[:, 1], coords[:, 2]] = True

            else:
                valid = F.interpolate(
                    (sdf_all>thresh).view(1, 1, *sdf_all.size()).float(), 
                    size=resolution, mode="trilinear", align_corners=True
                )[0, 0]
                
                sdf_all = F.interpolate(
                    sdf_all.view(1, 1, *sdf_all.size()), 
                    size=resolution, mode="trilinear", align_corners=True
                )[0, 0]

                coords_accum *= 2

                is_boundary = (valid > 0.0) & (valid < 1.0)
                is_boundary = smooth_conv3x3(is_boundary.float().view(1, 1, *is_boundary.size()))[0, 0] > 0

                is_boundary[coords_accum[:, 0], coords_accum[:, 1], coords_accum[:, 2]] = False

                # coords = is_boundary.nonzero() * stride
                coords = torch.nonzero(is_boundary) * stride
                coords2D = coords.float() / (2**steps[-1]+1) * (b_max - b_min) + b_min
                # coords2D = coords.float() / (2**steps[-1]+1)
                sdf = batch_eval(
                    coords2D.t(), 
                ) #[N]
                if sdf is not None:
                    sdf_all[is_boundary] = sdf
                voxels = coords / stride
                voxels = voxels.long()
                coords_accum = torch.cat([
                    voxels, 
                    coords_accum
                ], dim=0).unique(dim=0)
                calculated[coords[:, 0], coords[:, 1], coords[:, 2]] = True

                for n_iter in range(14):
                    sdf_valid = valid[voxels[:, 0], voxels[:, 1], voxels[:, 2]]
                    idxs_danger = ((sdf_valid==1) & (sdf<thresh)) | ((sdf_valid==0) & (sdf>thresh)) #[N,]
                    coords_danger = coords[idxs_danger, :] #[N, 3]
                    if coords_danger.size(0) == 0:
                        break

                    coords_arround = coords_danger.int() + gird8_offsets.view(-1, 1, 3) * stride
                    coords_arround = coords_arround.reshape(-1, 3).long()
                    coords_arround = coords_arround.unique(dim=0)
                    
                    coords_arround[:, 0] = coords_arround[:, 0].clamp(0, calculated.size(0)-1)
                    coords_arround[:, 1] = coords_arround[:, 1].clamp(0, calculated.size(1)-1)
                    coords_arround[:, 2] = coords_arround[:, 2].clamp(0, calculated.size(2)-1)

                    coords = coords_arround[
                        calculated[coords_arround[:, 0], coords_arround[:, 1], coords_arround[:, 2]] == False
                    ]
                    
                    if coords.size(0) == 0:
                        break
                    
                    coords2D = coords.float() / (2**steps[-1]+1) * (b_max - b_min) + b_min
                    # coords2D = coords.float() / (2**steps[-1]+1)
                    sdf = batch_eval(
                        coords2D.t(), 
                    ) #[N]

                    voxels = coords / stride
                    voxels = voxels.long()
                    sdf_all[voxels[:, 0], voxels[:, 1], voxels[:, 2]] = sdf
                    
                    coords_accum = torch.cat([
                        voxels, 
                        coords_accum
                    ], dim=0).unique(dim=0)
                    calculated[coords[:, 0], coords[:, 1], coords[:, 2]] = True

        return sdf_all.data.cpu().numpy()

def scalar_to_color(val, min=None, max=None):
    if min is None:
        min = val.min()
    if max is None:
        max = val.max()

    norm = matplotlib.colors.Normalize(vmin=min, vmax=max, clip=True)
    # use jet colormap
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.jet)
        
    return mapper.to_rgba(val)[:,:3]

def save_obj_mesh(mesh_path, verts, faces):
    file = open(mesh_path, 'w')

    for v in verts:
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
    if faces is not None:
        for f in faces:
            f_plus = f + 1
            file.write('f %d %d %d\n' % (f_plus[0], f_plus[2], f_plus[1]))
    file.close()


def save_obj_mesh_with_color(mesh_path, verts, faces, colors):
    file = open(mesh_path, 'w')

    for idx, v in enumerate(verts):
        c = colors[idx]
        file.write('v %.4f %.4f %.4f %.4f %.4f %.4f\n' % (v[0], v[1], v[2], c[0], c[1], c[2]))
    if faces is not None:
        for f in faces:
            f_plus = f + 1
            file.write('f %d %d %d\n' % (f_plus[0], f_plus[2], f_plus[1]))
    file.close()


def save_obj_mesh_with_uv(mesh_path, verts, faces, uvs):
    file = open(mesh_path, 'w')

    for idx, v in enumerate(verts):
        vt = uvs[idx]
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
        file.write('vt %.4f %.4f\n' % (vt[0], vt[1]))

    for f in faces:
        f_plus = f + 1
        file.write('f %d/%d %d/%d %d/%d\n' % (f_plus[0], f_plus[0],
                                              f_plus[2], f_plus[2],
                                              f_plus[1], f_plus[1]))
    file.close()


def save_samples_truncted_prob(fname, points, prob):
    '''
    Save the visualization of sampling to a ply file.
    Red points represent positive predictions.
    Green points represent negative predictions.
    :param fname: File name to save
    :param points: [N, 3] array of points
    :param prob: [N, 1] array of predictions in the range [0~1]
    :return:
    '''
    r = (prob > 0.5).reshape([-1, 1]) * 255
    g = (prob < 0.5).reshape([-1, 1]) * 255
    b = np.zeros(r.shape)

    to_save = np.concatenate([points, r, g, b], axis=-1)
    return np.savetxt(fname,
                      to_save,
                      fmt='%.6f %.6f %.6f %d %d %d',
                      comments='',
                      header=(
                          'ply\nformat ascii 1.0\nelement vertex {:d}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header').format(
                          points.shape[0])
                      )

def save_samples_sdf(fname, points, sdf):
    '''
    Save the visualization of sampling to a ply file.
    Red points represent positive predictions.
    Green points represent negative predictions.
    :param fname: File name to save
    :param points: [N, 3] array of points
    :param sdf: [N, 1] array of predictions in the range [0~1]
    :return:
    '''
    sdf = sdf.clip(-1,1)
    r = (0.5 + 0.5 * sdf).reshape([-1,1]) * 255
    g = (0.5 - 0.5 * sdf).reshape([-1,1]) * 255
    b = np.zeros(r.shape)

    to_save = np.concatenate([points, r, g, b], axis=-1)
    return np.savetxt(fname,
                      to_save,
                      fmt='%.6f %.6f %.6f %d %d %d',
                      comments='',
                      header=(
                          'ply\nformat ascii 1.0\nelement vertex {:d}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header').format(
                          points.shape[0])
                      )

def save_samples_rgb(fname, points, rgb):
    '''
    Save the visualization of sampling to a ply file.
    Red points represent positive predictions.
    Green points represent negative predictions.
    :param fname: File name to save
    :param points: [N, 3] array of points
    :param rgb: [N, 3] array of rgb values in the range [0~1]
    :return:
    '''
    to_save = np.concatenate([points, rgb * 255], axis=-1)
    return np.savetxt(fname,
                      to_save,
                      fmt='%.6f %.6f %.6f %d %d %d',
                      comments='',
                      header=(
                          'ply\nformat ascii 1.0\nelement vertex {:d}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header').format(
                          points.shape[0])
                      )

def load_trimesh(root_dir):
    folders = os.listdir(root_dir)
    meshs = {}
    for i, f in tqdm(enumerate(folders)):
        sub_name = f
        meshs[sub_name] = trimesh.load(os.path.join(root_dir, f, '%s_100k.obj' % sub_name))

    return meshs

def v2g(V, F):
    """
    args:
        vertices: (B, V, C)
        faces: (B, F, 3)
    return: 
        (B, F, 3, C)
    """
    assert (F.ndimension() == 3)
    assert (V.shape[0] == F.shape[0])

    device = V.device
    B, nV, C = V.size()
    B, nF = F.size()[:2]
    F = F + (torch.arange(B, dtype=torch.int32).to(device)*nV)[:, None, None]
    V = V.reshape((B * nV, C))

    return V[F.long()]

def compute_v_normal(Vs, Fs, norm=True):
    '''
    args:
        Vs: (B, N, 3)
        Fs: (B, F, 3)
    return:
        normal: (B, N, 3)
    '''
    Vf = v2g(Vs, Fs)
    Vf = Vf.reshape((Fs.size(0) * Fs.size(1), 3, 3))
    v10 = Vf[:, 1] - Vf[:, 0]
    v20 = Vf[:, 2] - Vf[:, 0]
    nf = torch.cross(v10, v20).view(Fs.size(0), Fs.size(1), 3)
    
    # NOTE: this may cause non-deterministic behavior in CUDA backend
    # if you cannot accept this, you can switch to cpu version here.

    # determinstic version (CPU)
    # Ns = torch.zeros(Vs.size()) # (B, N, 3)
    # Fs = Fs.view(Fs.size(0),Fs.size(1),3,1).expand(Fs.size(0),Fs.size(1),3,3)
    # nf = nf.view(nf.size(0),nf.size(1),1,nf.size(2)).expand_as(Fs).contiguous()
    # Ns = Ns.scatter_add_(1, Fs.long().view(Fs.size(0),-1,3).cpu(), nf.view(Fs.size(0),-1,3).cpu()).type_as(Vs)

    # nondetermistic version (CUDA)
    Ns = torch.zeros_like(Vs) # (B, N, 3)
    Fs = Fs.view(Fs.size(0),Fs.size(1),3,1).expand(Fs.size(0),Fs.size(1),3,3)
    nf = nf.view(nf.size(0),nf.size(1),1,nf.size(2)).expand_as(Fs).contiguous()
    Ns = Ns.scatter_add_(1, Fs.long().view(Fs.size(0),-1,3), nf.view(Fs.size(0),-1,3))

    if norm:
        # Ns = normalize(Ns, dim=2)
        Ns = F.normalize(Ns, dim=2)

    return Ns