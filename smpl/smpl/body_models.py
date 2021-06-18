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

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import os.path as osp

try:
    import cPickle as pickle
except ImportError:
    import pickle

import numpy as np

from collections import namedtuple

import torch
import torch.nn as nn

from .lbs import (
    lbs, vertices2landmarks, find_dynamic_lmk_idx_and_bcoords, batch_rodrigues, transform_mat)

from .vertex_ids import vertex_ids as VERTEX_IDS
from .utils import Struct, to_np, to_tensor
from .vertex_joint_selector import VertexJointSelector
import math


ModelOutput = namedtuple('ModelOutput',
                         ['vertices', 'joints', 'full_pose', 'betas',
                          'global_orient',
                          'body_pose', 'expression',
                          'left_hand_pose', 'right_hand_pose',
                          'jaw_pose', 'joint_transform', 'vertex_transform', 'v_posed', 'v_shaped'])
ModelOutput.__new__.__defaults__ = (None,) * len(ModelOutput._fields)


def create(model_path, model_type='smpl',
           **kwargs):
    ''' Method for creating a model from a path and a model type

        Parameters
        ----------
        model_path: str
            Either the path to the model you wish to load or a folder,
            where each subfolder contains the differents types, i.e.:
            model_path:
            |
            |-- smpl
                |-- SMPL_FEMALE
                |-- SMPL_NEUTRAL
                |-- SMPL_MALE
        model_type: str, optional
            When model_path is a folder, then this parameter specifies  the
            type of model to be loaded
        **kwargs: dict
            Keyword arguments

        Returns
        -------
            body_model: nn.Module
                The PyTorch module that implements the corresponding body model
        Raises
        ------
            ValueError: In case the model type is not one of SMPL
    '''

    # If it's a folder, assume
    if osp.isdir(model_path):
        if model_type.lower() == 'smpl_vitruvian':
            return SMPL_vitruvian(os.path.join(model_path, 'smpl'), **kwargs)
        else:
            model_path = os.path.join(model_path, model_type)

    if model_type.lower() == 'smpl':
        return SMPL(model_path, **kwargs)
    else:
        raise ValueError('Unknown model type {}, exiting!'.format(model_type))

class SMPL(nn.Module):

    NUM_JOINTS = 23
    NUM_BODY_JOINTS = 23
    NUM_BETAS = 10

    def __init__(self, model_path, data_struct=None,
                 create_betas=True,
                 betas=None,
                 create_global_orient=True,
                 global_orient=None,
                 create_body_pose=True,
                 body_pose=None,
                 create_transl=True,
                 transl=None,
                 dtype=torch.float32,
                 batch_size=1,
                 joint_mapper=None, gender='neutral',
                 vertex_ids=None,
                 **kwargs):
        ''' SMPL model constructor

            Parameters
            ----------
            model_path: str
                The path to the folder or to the file where the model
                parameters are stored
            data_struct: Strct
                A struct object. If given, then the parameters of the model are
                read from the object. Otherwise, the model tries to read the
                parameters from the given `model_path`. (default = None)
            create_global_orient: bool, optional
                Flag for creating a member variable for the global orientation
                of the body. (default = True)
            global_orient: torch.tensor, optional, Bx3
                The default value for the global orientation variable.
                (default = None)
            create_body_pose: bool, optional
                Flag for creating a member variable for the pose of the body.
                (default = True)
            body_pose: torch.tensor, optional, Bx(Body Joints * 3)
                The default value for the body pose variable.
                (default = None)
            create_betas: bool, optional
                Flag for creating a member variable for the shape space
                (default = True).
            betas: torch.tensor, optional, Bx10
                The default value for the shape member variable.
                (default = None)
            create_transl: bool, optional
                Flag for creating a member variable for the translation
                of the body. (default = True)
            transl: torch.tensor, optional, Bx3
                The default value for the transl variable.
                (default = None)
            dtype: torch.dtype, optional
                The data type for the created variables
            batch_size: int, optional
                The batch size used for creating the member variables
            joint_mapper: object, optional
                An object that re-maps the joints. Useful if one wants to
                re-order the SMPL joints to some other convention (e.g. MSCOCO)
                (default = None)
            gender: str, optional
                Which gender to load
            vertex_ids: dict, optional
                A dictionary containing the indices of the extra vertices that
                will be selected
        '''

        self.gender = gender

        if data_struct is None:
            if osp.isdir(model_path):
                model_fn = 'SMPL_{}.{ext}'.format(gender.upper(), ext='pkl')
                smpl_path = os.path.join(model_path, model_fn)
            else:
                smpl_path = model_path
            assert osp.exists(smpl_path), 'Path {} does not exist!'.format(
                smpl_path)

            with open(smpl_path, 'rb') as smpl_file:
                data_struct = Struct(**pickle.load(smpl_file,
                                                   encoding='latin1'))

        super(SMPL, self).__init__()
        self.batch_size = batch_size

        if vertex_ids is None:
            # SMPL and SMPL-H share the same topology, so any extra joints can
            # be drawn from the same place
            vertex_ids = VERTEX_IDS['smplh']

        self.dtype = dtype

        self.joint_mapper = joint_mapper

        self.vertex_joint_selector = VertexJointSelector(
            vertex_ids=vertex_ids, **kwargs)

        self.faces = data_struct.f
        self.register_buffer('faces_tensor',
                             to_tensor(to_np(self.faces, dtype=np.int64),
                                       dtype=torch.long))

        if create_betas:
            if betas is None:
                default_betas = torch.zeros([batch_size, self.NUM_BETAS],
                                            dtype=dtype)
            else:
                if 'torch.Tensor' in str(type(betas)):
                    default_betas = betas.clone().detach()
                else:
                    default_betas = torch.tensor(betas,
                                                 dtype=dtype)

            self.register_parameter('betas', nn.Parameter(default_betas,
                                                          requires_grad=True))

        # The tensor that contains the global rotation of the model
        # It is separated from the pose of the joints in case we wish to
        # optimize only over one of them
        if create_global_orient:
            if global_orient is None:
                default_global_orient = torch.zeros([batch_size, 3],
                                                    dtype=dtype)
            else:
                if 'torch.Tensor' in str(type(global_orient)):
                    default_global_orient = global_orient.clone().detach()
                else:
                    default_global_orient = torch.tensor(global_orient,
                                                         dtype=dtype)

            global_orient = nn.Parameter(default_global_orient,
                                         requires_grad=True)
            self.register_parameter('global_orient', global_orient)

        if create_body_pose:
            if body_pose is None:
                default_body_pose = torch.zeros(
                    [batch_size, self.NUM_BODY_JOINTS * 3], dtype=dtype)
            else:
                if 'torch.Tensor' in str(type(body_pose)):
                    default_body_pose = body_pose.clone().detach()
                else:
                    default_body_pose = torch.tensor(body_pose,
                                                     dtype=dtype)
            self.register_parameter(
                'body_pose',
                nn.Parameter(default_body_pose, requires_grad=True))

        if create_transl:
            if transl is None:
                default_transl = torch.zeros([batch_size, 3],
                                             dtype=dtype,
                                             requires_grad=True)
            else:
                default_transl = torch.tensor(transl, dtype=dtype)
            self.register_parameter(
                'transl',
                nn.Parameter(default_transl, requires_grad=True))

        # The vertices of the template model
        self.register_buffer('v_template',
                             to_tensor(to_np(data_struct.v_template),
                                       dtype=dtype))

        # The shape components
        shapedirs = data_struct.shapedirs
        # The shape components
        self.register_buffer(
            'shapedirs',
            to_tensor(to_np(shapedirs), dtype=dtype))

        j_regressor = to_tensor(to_np(
            data_struct.J_regressor), dtype=dtype)
        self.register_buffer('J_regressor', j_regressor)

        # Pose blend shape basis: 6890 x 3 x 207, reshaped to 6890*3 x 207
        num_pose_basis = data_struct.posedirs.shape[-1]
        # 207 x 20670
        posedirs = np.reshape(data_struct.posedirs, [-1, num_pose_basis]).T
        self.register_buffer('posedirs',
                             to_tensor(to_np(posedirs), dtype=dtype))

        # indices of parents for each joints
        parents = to_tensor(to_np(data_struct.kintree_table[0])).long()
        parents[0] = -1
        self.register_buffer('parents', parents)

        self.register_buffer('lbs_weights',
                             to_tensor(to_np(data_struct.weights), dtype=dtype))

    def create_mean_pose(self, data_struct):
        pass

    @torch.no_grad()
    def reset_params(self, **params_dict):
        for param_name, param in self.named_parameters():
            if param_name in params_dict:
                param[:] = torch.tensor(params_dict[param_name])
            else:
                param.fill_(0)

    def get_num_verts(self):
        return self.v_template.shape[0]

    def get_num_faces(self):
        return self.faces.shape[0]

    def extra_repr(self):
        return 'Number of betas: {}'.format(self.NUM_BETAS)
    
    def get_root_T(self, global_orient=None, transl=None, orient_origin=None):
        global_orient = (global_orient if global_orient is not None else
                         self.global_orient)
        if transl is None and hasattr(self, 'transl'):
            transl = self.transl

        rot_mat = batch_rodrigues(global_orient.view(-1, 3)).view([-1, 3, 3])

        if not orient_origin is None:
            transl = transl - (torch.einsum('bst,bvt->bvs', rot_mat, orient_origin) - orient_origin).squeeze(1)

        A = transform_mat(rot_mat, transl[:,:,None])

        return A

    def forward(self, betas=None, body_neutral_v = None, body_pose=None, global_orient=None,
                transl=None, return_verts=True, return_full_pose=False, pose2rot=True, custom_out=False,
                **kwargs):
        ''' Forward pass for the SMPL model

            Parameters
            ----------
            global_orient: torch.tensor, optional, shape Bx3
                If given, ignore the member variable and use it as the global
                rotation of the body. Useful if someone wishes to predicts this
                with an external model. (default=None)
            betas: torch.tensor, optional, shape Bx10
                If given, ignore the member variable `betas` and use it
                instead. For example, it can used if shape parameters
                `betas` are predicted from some external model.
                (default=None)
            body_pose: torch.tensor, optional, shape Bx(J*3)
                If given, ignore the member variable `body_pose` and use it
                instead. For example, it can used if someone predicts the
                pose of the body joints are predicted from some external model.
                It should be a tensor that contains joint rotations in
                axis-angle format. (default=None)
            transl: torch.tensor, optional, shape Bx3
                If given, ignore the member variable `transl` and use it
                instead. For example, it can used if the translation
                `transl` is predicted from some external model.
                (default=None)
            return_verts: bool, optional
                Return the vertices. (default=True)
            return_full_pose: bool, optional
                Returns the full axis-angle pose vector (default=False)

            Returns
            -------
        '''
        # If no shape and pose parameters are passed along, then use the
        # ones from the module
        global_orient = (global_orient if global_orient is not None else
                         self.global_orient)
        body_pose = body_pose if body_pose is not None else self.body_pose
        betas = betas if betas is not None else self.betas

        apply_trans = transl is not None or hasattr(self, 'transl')
        if transl is None and hasattr(self, 'transl'):
            transl = self.transl

        full_pose = torch.cat([global_orient, body_pose], dim=1)

        batch_size = max(betas.shape[0], global_orient.shape[0],
                         body_pose.shape[0])

        if betas.shape[0] != batch_size:
            num_repeats = int(batch_size / betas.shape[0])
            betas = betas.expand(num_repeats, -1)

        ret = lbs(betas, full_pose, self.v_template,
                self.shapedirs, self.posedirs,
                self.J_regressor, self.parents,
                self.lbs_weights, pose2rot=pose2rot, 
                body_neutral_v = body_neutral_v,
                dtype=self.dtype, custom_out=custom_out)
        vertices = ret['verts']
        joints = ret['joints']

        if custom_out:
            joints, joints_transform = self.vertex_joint_selector(vertices, joints, True, ret['vT'], ret['jT'])
        else:
            joints = self.vertex_joint_selector(ret['verts'], ret['joints'])

        # Map the joints to the current dataset
        if self.joint_mapper is not None:
            joints = self.joint_mapper(joints)
            if custom_out:
                joints_transform = self.joint_mapper(joints_transform)

        if apply_trans:
            joints += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)
            if custom_out:
                joints_transform[:,:,:3,3] += transl.unsqueeze(dim=1)

        output = ModelOutput(vertices=vertices if return_verts else None,
                             global_orient=global_orient,
                             body_pose=body_pose,
                             joints=joints,
                             betas=betas,
                             full_pose=full_pose if return_full_pose else None,
                             joint_transform=joints_transform if custom_out else None,
                             vertex_transform=ret['vT'] if custom_out else None,
                             v_shaped=ret['v_shaped'] if custom_out else None,
                             v_posed=ret['v_posed'] if custom_out else None)

        return output

class SMPL_vitruvian(nn.Module):

    NUM_JOINTS = 23
    NUM_BODY_JOINTS = 23
    NUM_BETAS = 10

    def __init__(self, model_path, data_struct=None,
                 create_betas=True,
                 betas=None,
                 create_global_orient=True,
                 global_orient=None,
                 create_body_pose=True,
                 body_pose=None,
                 create_transl=True,
                 transl=None,
                 dtype=torch.float32,
                 batch_size=1,
                 joint_mapper=None, gender='neutral',
                 vertex_ids=None,
                 half_angle=25,
                 **kwargs):
        ''' SMPL model constructor

            Parameters
            ----------
            model_path: str
                The path to the folder or to the file where the model
                parameters are stored
            data_struct: Strct
                A struct object. If given, then the parameters of the model are
                read from the object. Otherwise, the model tries to read the
                parameters from the given `model_path`. (default = None)
            create_global_orient: bool, optional
                Flag for creating a member variable for the global orientation
                of the body. (default = True)
            global_orient: torch.tensor, optional, Bx3
                The default value for the global orientation variable.
                (default = None)
            create_body_pose: bool, optional
                Flag for creating a member variable for the pose of the body.
                (default = True)
            body_pose: torch.tensor, optional, Bx(Body Joints * 3)
                The default value for the body pose variable.
                (default = None)
            create_betas: bool, optional
                Flag for creating a member variable for the shape space
                (default = True).
            betas: torch.tensor, optional, Bx10
                The default value for the shape member variable.
                (default = None)
            create_transl: bool, optional
                Flag for creating a member variable for the translation
                of the body. (default = True)
            transl: torch.tensor, optional, Bx3
                The default value for the transl variable.
                (default = None)
            dtype: torch.dtype, optional
                The data type for the created variables
            batch_size: int, optional
                The batch size used for creating the member variables
            joint_mapper: object, optional
                An object that re-maps the joints. Useful if one wants to
                re-order the SMPL joints to some other convention (e.g. MSCOCO)
                (default = None)
            gender: str, optional
                Which gender to load
            vertex_ids: dict, optional
                A dictionary containing the indices of the extra vertices that
                will be selected
        '''
        self.vitruvian_angle = half_angle
        self.vitruvian_pose = torch.zeros([1, 69], dtype=torch.float32)
        self.vitruvian_pose[:,2] = math.radians(self.vitruvian_angle) # for vitruvian pose
        self.vitruvian_pose[:,5] = math.radians(-self.vitruvian_angle) # for vitruvian pose 

        self.vitruvian_joints_transform = None
        self.vitruvian_vertices_transform = None

        self.gender = gender

        if data_struct is None:
            if osp.isdir(model_path):
                model_fn = 'SMPL_{}.{ext}'.format(gender.upper(), ext='pkl')
                smpl_path = os.path.join(model_path, model_fn)
            else:
                smpl_path = model_path
            assert osp.exists(smpl_path), 'Path {} does not exist!'.format(
                smpl_path)

            with open(smpl_path, 'rb') as smpl_file:
                data_struct = Struct(**pickle.load(smpl_file,
                                                   encoding='latin1'))

        super(SMPL_vitruvian, self).__init__()
        self.batch_size = batch_size

        if vertex_ids is None:
            # SMPL and SMPL-H share the same topology, so any extra joints can
            # be drawn from the same place
            vertex_ids = VERTEX_IDS['smplh']

        self.dtype = dtype

        self.joint_mapper = joint_mapper

        self.vertex_joint_selector = VertexJointSelector(
            vertex_ids=vertex_ids, **kwargs)

        self.faces = data_struct.f
        self.register_buffer('faces_tensor',
                             to_tensor(to_np(self.faces, dtype=np.int64),
                                       dtype=torch.long))

        if create_betas:
            if betas is None:
                default_betas = torch.zeros([batch_size, self.NUM_BETAS],
                                            dtype=dtype)
            else:
                if 'torch.Tensor' in str(type(betas)):
                    default_betas = betas.clone().detach()
                else:
                    default_betas = torch.tensor(betas,
                                                 dtype=dtype)

            self.register_parameter('betas', nn.Parameter(default_betas,
                                                          requires_grad=True))

        # The tensor that contains the global rotation of the model
        # It is separated from the pose of the joints in case we wish to
        # optimize only over one of them
        if create_global_orient:
            if global_orient is None:
                default_global_orient = torch.zeros([batch_size, 3],
                                                    dtype=dtype)
            else:
                if 'torch.Tensor' in str(type(global_orient)):
                    default_global_orient = global_orient.clone().detach()
                else:
                    default_global_orient = torch.tensor(global_orient,
                                                         dtype=dtype)

            global_orient = nn.Parameter(default_global_orient,
                                         requires_grad=True)
            self.register_parameter('global_orient', global_orient)

        if create_body_pose:
            if body_pose is None:
                default_body_pose = torch.zeros(
                    [batch_size, self.NUM_BODY_JOINTS * 3], dtype=dtype)
            else:
                if 'torch.Tensor' in str(type(body_pose)):
                    default_body_pose = body_pose.clone().detach()
                else:
                    default_body_pose = torch.tensor(body_pose,
                                                     dtype=dtype)
            self.register_parameter(
                'body_pose',
                nn.Parameter(default_body_pose, requires_grad=True))

        if create_transl:
            if transl is None:
                default_transl = torch.zeros([batch_size, 3],
                                             dtype=dtype,
                                             requires_grad=True)
            else:
                default_transl = torch.tensor(transl, dtype=dtype)
            self.register_parameter(
                'transl',
                nn.Parameter(default_transl, requires_grad=True))

        # The vertices of the template model
        self.register_buffer('v_template',
                             to_tensor(to_np(data_struct.v_template),
                                       dtype=dtype))

        # The shape components
        shapedirs = data_struct.shapedirs
        # The shape components
        self.register_buffer(
            'shapedirs',
            to_tensor(to_np(shapedirs), dtype=dtype))

        j_regressor = to_tensor(to_np(
            data_struct.J_regressor), dtype=dtype)
        self.register_buffer('J_regressor', j_regressor)

        # Pose blend shape basis: 6890 x 3 x 207, reshaped to 6890*3 x 207
        num_pose_basis = data_struct.posedirs.shape[-1]
        # 207 x 20670
        posedirs = np.reshape(data_struct.posedirs, [-1, num_pose_basis]).T
        self.register_buffer('posedirs',
                             to_tensor(to_np(posedirs), dtype=dtype))

        # indices of parents for each joints
        parents = to_tensor(to_np(data_struct.kintree_table[0])).long()
        parents[0] = -1
        self.register_buffer('parents', parents)

        self.register_buffer('lbs_weights',
                             to_tensor(to_np(data_struct.weights), dtype=dtype))


    def create_mean_pose(self, data_struct):
        pass

    @torch.no_grad()
    def reset_params(self, **params_dict):
        for param_name, param in self.named_parameters():
            if param_name in params_dict:
                param[:] = torch.tensor(params_dict[param_name])
            else:
                param.fill_(0)

    def get_num_verts(self):
        return self.v_template.shape[0]

    def get_num_faces(self):
        return self.faces.shape[0]

    def extra_repr(self):
        return 'Number of betas: {}'.format(self.NUM_BETAS)
    
    def get_root_T(self, global_orient=None, transl=None, orient_origin=None):
        global_orient = (global_orient if global_orient is not None else
                         self.global_orient)
        if transl is None and hasattr(self, 'transl'):
            transl = self.transl

        rot_mat = batch_rodrigues(global_orient.view(-1, 3)).view([-1, 3, 3])

        if not orient_origin is None:
            transl = transl - (torch.einsum('bst,bvt->bvs', rot_mat, orient_origin) - orient_origin).squeeze(1)

        A = transform_mat(rot_mat, transl[:,:,None])

        return A

    def initiate_vitruvian(self, betas=None, body_neutral_v = None, device = None, vitruvian_angle = 25,
                return_verts=True, return_full_pose=False, pose2rot=True,
                **kwargs):

        self.vitruvian_angle = vitruvian_angle
        self.vitruvian_pose = torch.zeros([1, 69], dtype=torch.float32)
        self.vitruvian_pose[:,2] = math.radians(self.vitruvian_angle) # for vitruvian pose
        self.vitruvian_pose[:,5] = math.radians(-self.vitruvian_angle) # for vitruvian pose 

        body_pose = self.vitruvian_pose.to(device)

        betas = betas if betas is not None else torch.zeros([1, 10], dtype=torch.float32, device=device)
        global_orient=torch.zeros((1, 3), dtype=torch.float32, device=device)
        transl=torch.zeros((1, 3), dtype=torch.float32, device=device)

        full_pose = torch.cat([global_orient, body_pose], dim=1)

        batch_size = max(betas.shape[0], global_orient.shape[0],
                         body_pose.shape[0])

        if betas.shape[0] != batch_size:
            num_repeats = int(batch_size / betas.shape[0])
            betas = betas.expand(num_repeats, -1)

        ret = lbs(betas, full_pose, self.v_template,
                self.shapedirs, self.posedirs,
                self.J_regressor, self.parents,
                self.lbs_weights, pose2rot=pose2rot, 
                body_neutral_v = body_neutral_v,
                dtype=self.dtype, custom_out=True)
        vitruvian_vertices = ret['verts']
        vitruvian_joints = ret['joints']
        vitruvian_vertices_transform = ret['vT']
        _, vitruvian_joints_transform = self.vertex_joint_selector(vitruvian_vertices, vitruvian_joints, True, ret['vT'], ret['jT'])
        
        self.vitruvian_vertices_transform = vitruvian_vertices_transform.detach()
        self.vitruvian_joints_transform = vitruvian_joints_transform.detach()
        self.vitruvian_vertices = vitruvian_vertices.detach()

        with torch.no_grad():
            self.inverse_vitruvian_vertices_transform = self.vitruvian_vertices_transform.view(-1, 4, 4).inverse().view(batch_size, -1, 4, 4)
            self.inverse_vitruvian_joints_transform = self.vitruvian_joints_transform.view(-1, 4, 4).inverse().view(batch_size, -1, 4, 4)
        return self.vitruvian_vertices


    def forward(self, betas=None, body_neutral_v = None, body_pose=None, global_orient=None,
                transl=None, return_verts=True, return_full_pose=False, pose2rot=True, custom_out=False,
                **kwargs):
        ''' Forward pass for the SMPL model

            Parameters
            ----------
            global_orient: torch.tensor, optional, shape Bx3
                If given, ignore the member variable and use it as the global
                rotation of the body. Useful if someone wishes to predicts this
                with an external model. (default=None)
            betas: torch.tensor, optional, shape Bx10
                If given, ignore the member variable `betas` and use it
                instead. For example, it can used if shape parameters
                `betas` are predicted from some external model.
                (default=None)
            body_pose: torch.tensor, optional, shape Bx(J*3)
                If given, ignore the member variable `body_pose` and use it
                instead. For example, it can used if someone predicts the
                pose of the body joints are predicted from some external model.
                It should be a tensor that contains joint rotations in
                axis-angle format. (default=None)
            transl: torch.tensor, optional, shape Bx3
                If given, ignore the member variable `transl` and use it
                instead. For example, it can used if the translation
                `transl` is predicted from some external model.
                (default=None)
            return_verts: bool, optional
                Return the vertices. (default=True)
            return_full_pose: bool, optional
                Returns the full axis-angle pose vector (default=False)

            Returns
            -------
        '''
        # If no shape and pose parameters are passed along, then use the
        # ones from the module
        global_orient = (global_orient if global_orient is not None else
                         self.global_orient)
        body_pose = body_pose if body_pose is not None else self.body_pose
        if body_pose == 'vitruvian_pose':
            body_pose = self.vitruvian_pose.to(global_orient.device)
        betas = betas if betas is not None else self.betas

        apply_trans = transl is not None or hasattr(self, 'transl')
        if transl is None and hasattr(self, 'transl'):
            transl = self.transl

        full_pose = torch.cat([global_orient, body_pose], dim=1)

        batch_size = max(betas.shape[0], global_orient.shape[0],
                         body_pose.shape[0])

        if betas.shape[0] != batch_size:
            num_repeats = int(batch_size / betas.shape[0])
            betas = betas.expand(num_repeats, -1)

        ret = lbs(betas, full_pose, self.v_template,
                self.shapedirs, self.posedirs,
                self.J_regressor, self.parents,
                self.lbs_weights, pose2rot=pose2rot, 
                body_neutral_v = body_neutral_v,
                dtype=self.dtype, custom_out=custom_out)
        ret['v_shaped'] = self.vitruvian_vertices.expand(ret['v_shaped'].shape[0], -1, -1)
        homogen_ones = torch.ones_like(ret['v_posed'])[:,:,0:1]
        ret['v_posed'] = torch.cat((ret['v_posed'],homogen_ones), axis=2)
        ret['v_posed'] = torch.einsum('bvst,bvt->bvs', 
                        self.vitruvian_vertices_transform.expand(ret['v_posed'].shape[0], -1, -1, -1),
                        ret['v_posed'])[:,:,:3]
        vertices = ret['verts']
        joints = ret['joints']

        if custom_out:
            joints, joints_transform = self.vertex_joint_selector(vertices, joints, True, ret['vT'], ret['jT'])
        else:
            joints = self.vertex_joint_selector(ret['verts'], ret['joints'])

        # Map the joints to the current dataset
        if self.joint_mapper is not None:
            joints = self.joint_mapper(joints)
            if custom_out:
                joints_transform = self.joint_mapper(joints_transform)

        if apply_trans:
            joints += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)
            if custom_out:
                joints_transform[:,:,:3,3] += transl.unsqueeze(dim=1)
                vertex_transform = ret['vT']
                vertex_transform[:,:,:3,3] += transl.unsqueeze(dim=1)

                if self.vitruvian_joints_transform is None:
                    print("Error! self.vitruvian_joints_transform is None!")
                    print("Please run self.initiate_vitruvian() before requesting joints or vertices transformation!")
                    exit()
                else:
                    joints_transform = torch.bmm(joints_transform.view(-1,4,4), self.inverse_vitruvian_joints_transform.expand(batch_size,-1,-1,-1).view(-1,4,4))
                    joints_transform = joints_transform.view(batch_size,-1,4,4)
                    vertex_transform = torch.bmm(vertex_transform.view(-1,4,4), self.inverse_vitruvian_vertices_transform.expand(batch_size,-1,-1,-1).view(-1,4,4))
                    vertex_transform = vertex_transform.view(batch_size,-1,4,4)

        output = ModelOutput(vertices=vertices if return_verts else None,
                             global_orient=global_orient,
                             body_pose=body_pose,
                             joints=joints,
                             betas=betas,
                             full_pose=full_pose if return_full_pose else None,
                             joint_transform=joints_transform if custom_out else None,
                             vertex_transform=vertex_transform if custom_out else None,
                             v_shaped=ret['v_shaped'] if custom_out else None,
                             v_posed=ret['v_posed'] if custom_out else None)

        return output

