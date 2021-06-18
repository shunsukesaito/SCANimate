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
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

def detectBoundary(F):
    '''
    input:
        F: (F, 3) numpy triangle list
    return:
        (F) boundary flag
    '''
    tri_dic = {}
    nV = F.max()
    for i in range(F.shape[0]):
        idx = [F[i,0],F[i,1],F[i,2]]

        if (idx[1],idx[0]) in tri_dic:
            tri_dic[(idx[1],idx[0])].append(i)
        else:
            tri_dic[(idx[0],idx[1])] = [i]

        if (idx[2],idx[1]) in tri_dic:
            tri_dic[(idx[2],idx[1])].append(i)
        else:
            tri_dic[(idx[1],idx[2])] = [i]

        if (idx[0],idx[2]) in tri_dic:
            tri_dic[(idx[0],idx[2])].append(i)
        else:
            tri_dic[(idx[2],idx[0])] = [i]

    v_boundary = np.array((nV+1)*[False])
    for key in tri_dic:
        if len(tri_dic[key]) != 2:
            v_boundary[key[0]] = True
            v_boundary[key[1]] = True

    boundary = v_boundary[F[:,0]] | v_boundary[F[:,1]] | v_boundary[F[:,2]]
    
    return boundary

def computeMeanCurvature(V, N, F, norm_factor=10.0):
    '''
    input:
        V: (B, N, 3)
        N: (B, N, 3)
        F: (B, F, 3)
    output:
        (B, F, 3) cotangent weight, corresponding edge is ordered in 23, 31, 12
    '''
    B, nF = F.size()[:2]

    indices_repeat = F[:,:,None].expand(*F.size()[:2],3,*F.size()[2:])

    v1 = torch.gather(V, 1, indices_repeat[:, :, :, 0].long())
    v2 = torch.gather(V, 1, indices_repeat[:, :, :, 1].long())
    v3 = torch.gather(V, 1, indices_repeat[:, :, :, 2].long())
    
    n1 = torch.gather(N, 1, indices_repeat[:, :, :, 0].long())
    n2 = torch.gather(N, 1, indices_repeat[:, :, :, 1].long())
    n3 = torch.gather(N, 1, indices_repeat[:, :, :, 2].long())

    dv1 = v2 - v3
    dv2 = v3 - v1
    dv3 = v1 - v2

    lsq1 = dv1.pow(2).sum(2)
    lsq2 = dv2.pow(2).sum(2)
    lsq3 = dv3.pow(2).sum(2)

    dn1 = n2 - n3
    dn2 = n3 - n1
    dn3 = n1 - n2

    c1 = (dv1 * dn1).sum(2) / (lsq1 + 1e-8) # (B, F)
    c2 = (dv2 * dn2).sum(2) / (lsq2 + 1e-8)
    c3 = (dv3 * dn3).sum(2) / (lsq3 + 1e-8)

    C = torch.stack([c1, c2, c3], 2)[:,:,:,None].expand(B, nF, 3, 2).contiguous().view(B, -1, 1)

    idx1 = F[:,:,0:1]
    idx2 = F[:,:,1:2]
    idx3 = F[:,:,2:]

    idx23 = torch.stack([idx2, idx3], 3)
    idx31 = torch.stack([idx3, idx1], 3)
    idx12 = torch.stack([idx1, idx2], 3)

    Fst = torch.cat([idx23, idx31, idx12], 2).contiguous().view(B, -1, 1)

    Hv = torch.zeros_like(V[:,:,0:1]) # (B, N)
    Cv = torch.zeros_like(V[:,:,0:1]) # (B, N)
    Cnt = torch.ones_like(C) # (B, N)
    Hv = Hv.scatter_add_(1, Fst.long(), C)
    Cv = Cv.scatter_add_(1, Fst.long(), Cnt)
    
    Hv = Hv / Cv / (2.0*norm_factor) + 0.5 # to roughly range [-1, 1]

    return Hv

def vertices_to_faces(vertices, faces):
    assert (faces.ndimension() == 3)
    assert (vertices.shape[0] == faces.shape[0])
    assert (faces.shape[2] == 3)

    bs, nv, c = vertices.shape
    bs, nf = faces.shape[:2]
    device = vertices.device
    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
    vertices = vertices.reshape((bs * nv, c))
    # pytorch only supports long and byte tensors for indexing
    return vertices[faces.long()]

def compute_normal_v(Vs, Fs, norm=True):
    B, nF = Fs.size()[:2]
    Vf = vertices_to_faces(Vs, Fs)
    Vf = Vf.reshape((B * nF, 3, 3))
    v10 = Vf[:, 1] - Vf[:, 0]
    v20 = Vf[:, 2] - Vf[:, 0]
    nf = torch.cross(v10, v20).view(B, nF, 3) # (B * nF, 3)
    
    Ns = torch.zeros(Vs.size()) # (B, N, 3)
    Fs = Fs.view(Fs.size(0),Fs.size(1),3,1).expand(Fs.size(0),Fs.size(1),3,3)
    nf = nf.view(nf.size(0),nf.size(1),1,nf.size(2)).expand_as(Fs).contiguous()
    Ns = Ns.scatter_add_(1, Fs.long().reshape(Fs.size(0),-1,3).cpu(), nf.reshape(Fs.size(0),-1,3).cpu()).type_as(Vs)

    # Ns = torch.zeros_like(Vs) # (B, N, 3)
    # Fs = Fs.view(B,nF,3,1).expand(B,nF,3,3)
    # nf = nf.view(B,nF,1,3).expand_as(Fs).contiguous()
    # Ns = Ns.scatter_add_(1, Fs.long().view(B,-1,3), nf.view(B,-1,3))

    if norm:
        Ns = Fn.normalize(Ns, dim=2)

    return Ns
