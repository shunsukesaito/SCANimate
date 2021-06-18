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

def index_custom(feat, uv):
    '''
    args:
        feat: (B, C, H, W)
        uv: (B, 2, N)
    return:
        (B, C, N)
    '''
    device = feat.device
    B, C, H, W = feat.size()
    _, _, N = uv.size()
    
    x, y = uv[:,0], uv[:,1]
    x = (W-1.0) * (0.5 * x.contiguous().view(-1) + 0.5)
    y = (H-1.0) * (0.5 * y.contiguous().view(-1) + 0.5)

    x0 = torch.floor(x).int()
    x1 = x0 + 1
    y0 = torch.floor(y).int()
    y1 = y0 + 1

    max_x = W - 1
    max_y = H - 1

    x0_clamp = torch.clamp(x0, 0, max_x)
    x1_clamp = torch.clamp(x1, 0, max_x)
    y0_clamp = torch.clamp(y0, 0, max_y)
    y1_clamp = torch.clamp(y1, 0, max_y)

    dim2 = W
    dim1 = W * H

    base = (dim1 * torch.arange(B).int()).view(B, 1).expand(B, N).contiguous().view(-1).to(device)

    base_y0 = base + y0_clamp * dim2
    base_y1 = base + y1_clamp * dim2

    idx_y0_x0 = base_y0 + x0_clamp
    idx_y0_x1 = base_y0 + x1_clamp
    idx_y1_x0 = base_y1 + x0_clamp
    idx_y1_x1 = base_y1 + x1_clamp

    # (B,C,H,W) -> (B,H,W,C)
    im_flat = feat.permute(0,2,3,1).contiguous().view(-1, C)
    i_y0_x0 = torch.gather(im_flat, 0, idx_y0_x0.unsqueeze(1).expand(-1,C).long())
    i_y0_x1 = torch.gather(im_flat, 0, idx_y0_x1.unsqueeze(1).expand(-1,C).long())
    i_y1_x0 = torch.gather(im_flat, 0, idx_y1_x0.unsqueeze(1).expand(-1,C).long())
    i_y1_x1 = torch.gather(im_flat, 0, idx_y1_x1.unsqueeze(1).expand(-1,C).long())
    
    # Check the out-of-boundary case.
    x0_valid = (x0 <= max_x) & (x0 >= 0)
    x1_valid = (x1 <= max_x) & (x1 >= 0)
    y0_valid = (y0 <= max_y) & (y0 >= 0)
    y1_valid = (y1 <= max_y) & (y1 >= 0)

    x0 = x0.float()
    x1 = x1.float()
    y0 = y0.float()
    y1 = y1.float()

    w_y0_x0 = ((x1 - x) * (y1 - y) * (x1_valid * y1_valid).float()).unsqueeze(1)
    w_y0_x1 = ((x - x0) * (y1 - y) * (x0_valid * y1_valid).float()).unsqueeze(1)
    w_y1_x0 = ((x1 - x) * (y - y0) * (x1_valid * y0_valid).float()).unsqueeze(1)
    w_y1_x1 = ((x - x0) * (y - y0) * (x0_valid * y0_valid).float()).unsqueeze(1)

    output = w_y0_x0 * i_y0_x0 + w_y0_x1 * i_y0_x1 + w_y1_x0 * i_y1_x0 + w_y1_x1 * i_y1_x1 # (B, N, C)

    return output.view(B, N, C).permute(0,2,1).contiguous()

def index3d_custom(feat, pts):
    '''
    args:
        feat: (B, C, D, H, W)
        pts: (B, 3, N)
    return:
        (B, C, N)
    '''
    device = feat.device
    B, C, D, H, W = feat.size()
    _, _, N = pts.size()

    x, y, z = pts[:,0], pts[:,1], pts[:,2]
    x = (W-1.0) * (0.5 * x.contiguous().view(-1) + 0.5)
    y = (H-1.0) * (0.5 * y.contiguous().view(-1) + 0.5)
    z = (D-1.0) * (0.5 * z.contiguous().view(-1) + 0.5)

    x0 = torch.floor(x).int()
    x1 = x0 + 1
    y0 = torch.floor(y).int()
    y1 = y0 + 1
    z0 = torch.floor(z).int()
    z1 = z0 + 1

    max_x = W - 1
    max_y = H - 1
    max_z = D - 1

    x0_clamp = torch.clamp(x0, 0, max_x)
    x1_clamp = torch.clamp(x1, 0, max_x)
    y0_clamp = torch.clamp(y0, 0, max_y)
    y1_clamp = torch.clamp(y1, 0, max_y)
    z0_clamp = torch.clamp(z0, 0, max_z)
    z1_clamp = torch.clamp(z1, 0, max_z)

    dim3 = W
    dim2 = W * H
    dim1 = W * H * D

    base = (dim1 * torch.arange(B).int()).view(B, 1).expand(B, N).contiguous().view(-1).to(device)

    base_z0_y0 = base + z0_clamp * dim2 + y0_clamp * dim3
    base_z0_y1 = base + z0_clamp * dim2 + y1_clamp * dim3
    base_z1_y0 = base + z1_clamp * dim2 + y0_clamp * dim3
    base_z1_y1 = base + z1_clamp * dim2 + y1_clamp * dim3

    idx_z0_y0_x0 = base_z0_y0 + x0_clamp
    idx_z0_y0_x1 = base_z0_y0 + x1_clamp
    idx_z0_y1_x0 = base_z0_y1 + x0_clamp
    idx_z0_y1_x1 = base_z0_y1 + x1_clamp
    idx_z1_y0_x0 = base_z1_y0 + x0_clamp
    idx_z1_y0_x1 = base_z1_y0 + x1_clamp
    idx_z1_y1_x0 = base_z1_y1 + x0_clamp
    idx_z1_y1_x1 = base_z1_y1 + x1_clamp

    # (B,C,D,H,W) -> (B,D,H,W,C)
    im_flat = feat.permute(0,2,3,4,1).contiguous().view(-1, C)
    i_z0_y0_x0 = torch.gather(im_flat, 0, idx_z0_y0_x0.unsqueeze(1).expand(-1,C).long())
    i_z0_y0_x1 = torch.gather(im_flat, 0, idx_z0_y0_x1.unsqueeze(1).expand(-1,C).long())
    i_z0_y1_x0 = torch.gather(im_flat, 0, idx_z0_y1_x0.unsqueeze(1).expand(-1,C).long())
    i_z0_y1_x1 = torch.gather(im_flat, 0, idx_z0_y1_x1.unsqueeze(1).expand(-1,C).long())
    i_z1_y0_x0 = torch.gather(im_flat, 0, idx_z1_y0_x0.unsqueeze(1).expand(-1,C).long())
    i_z1_y0_x1 = torch.gather(im_flat, 0, idx_z1_y0_x1.unsqueeze(1).expand(-1,C).long())
    i_z1_y1_x0 = torch.gather(im_flat, 0, idx_z1_y1_x0.unsqueeze(1).expand(-1,C).long())
    i_z1_y1_x1 = torch.gather(im_flat, 0, idx_z1_y1_x1.unsqueeze(1).expand(-1,C).long())
    
    # Check the out-of-boundary case.
    x0_valid = (x0 <= max_x) & (x0 >= 0)
    x1_valid = (x1 <= max_x) & (x1 >= 0)
    y0_valid = (y0 <= max_y) & (y0 >= 0)
    y1_valid = (y1 <= max_y) & (y1 >= 0)
    z0_valid = (z0 <= max_z) & (z0 >= 0)
    z1_valid = (z1 <= max_z) & (z1 >= 0)

    x0 = x0.float()
    x1 = x1.float()
    y0 = y0.float()
    y1 = y1.float()
    z0 = z0.float()
    z1 = z1.float()

    w_z0_y0_x0 = ((x1 - x) * (y1 - y) * (z1 - z) * (x1_valid * y1_valid * z1_valid).float()).unsqueeze(1)
    w_z0_y0_x1 = ((x - x0) * (y1 - y) * (z1 - z) * (x0_valid * y1_valid * z1_valid).float()).unsqueeze(1)
    w_z0_y1_x0 = ((x1 - x) * (y - y0) * (z1 - z) * (x1_valid * y0_valid * z1_valid).float()).unsqueeze(1)
    w_z0_y1_x1 = ((x - x0) * (y - y0) * (z1 - z) * (x0_valid * y0_valid * z1_valid).float()).unsqueeze(1)
    w_z1_y0_x0 = ((x1 - x) * (y1 - y) * (z - z0) * (x1_valid * y1_valid * z0_valid).float()).unsqueeze(1)
    w_z1_y0_x1 = ((x - x0) * (y1 - y) * (z - z0) * (x0_valid * y1_valid * z0_valid).float()).unsqueeze(1)
    w_z1_y1_x0 = ((x1 - x) * (y - y0) * (z - z0) * (x1_valid * y0_valid * z0_valid).float()).unsqueeze(1)
    w_z1_y1_x1 = ((x - x0) * (y - y0) * (z - z0) * (x0_valid * y0_valid * z0_valid).float()).unsqueeze(1)

    output = w_z0_y0_x0 * i_z0_y0_x0 + w_z0_y0_x1 * i_z0_y0_x1 + w_z0_y1_x0 * i_z0_y1_x0 + w_z0_y1_x1 * i_z0_y1_x1 \
            + w_z1_y0_x0 * i_z1_y0_x0 + w_z1_y0_x1 * i_z1_y0_x1 + w_z1_y1_x0 * i_z1_y1_x0 + w_z1_y1_x1 * i_z1_y1_x1

    return output.view(B, N, C).permute(0,2,1).contiguous()

def index3d_nearest(feat, pts):
    '''
    args:
        feat: (B, C, D, H, W)
        pts: (B, 3, N)
    return:
        (B, C, N)
    '''
    device = feat.device
    B, C, D, H, W = feat.size()
    _, _, N = pts.size()

    x, y, z = pts[:,0], pts[:,1], pts[:,2]
    x = (W-1.0) * (0.5 * x.contiguous().view(-1) + 0.5)
    y = (H-1.0) * (0.5 * y.contiguous().view(-1) + 0.5)
    z = (D-1.0) * (0.5 * z.contiguous().view(-1) + 0.5)

    x0 = torch.floor(x).int()
    y0 = torch.floor(y).int()
    z0 = torch.floor(z).int()

    max_x = W - 1
    max_y = H - 1
    max_z = D - 1

    x0_clamp = torch.clamp(x0, 0, max_x)
    y0_clamp = torch.clamp(y0, 0, max_y)
    z0_clamp = torch.clamp(z0, 0, max_z)

    s = x - x0.float()
    t = y - y0.float()
    v = z - z0.float()

    dim3 = W
    dim2 = W * H
    dim1 = W * H * D

    x0_valid = (x0 <= max_x) & (x0 >= 0)
    y0_valid = (y0 <= max_y) & (y0 >= 0)
    z0_valid = (z0 <= max_z) & (z0 >= 0)

    base = (dim1 * torch.arange(B).int()).view(B, 1).expand(B, N).contiguous().view(-1).to(device)

    base_z0_y0 = base + z0_clamp * dim2 + y0_clamp * dim3
    idx_z0_y0_x0 = base_z0_y0 + x0_clamp

    # (B,C,D,H,W) -> (B,D,H,W,C)
    im_flat = feat.permute(0,2,3,4,1).contiguous().view(-1, C)
    i_z0_y0_x0 = torch.gather(im_flat, 0, idx_z0_y0_x0.unsqueeze(1).expand(-1,C).long())

    w_z0_y0_x0 = ((x0_valid * y0_valid * z0_valid).float()).unsqueeze(1)

    stv = torch.stack([s.view(B,-1), t.view(B,-1), v.view(B,-1)], 1)

    output = (w_z0_y0_x0 * i_z0_y0_x0).view(B, N, C).permute(0,2,1).contiguous()

    return output, stv-0.5 # (-0.5, 0.5)

def index3d_nearest_overlap(feat, pts):
    '''
    args:
        feat: (B, C, D, H, W)
        pts: (B, 3, N)
    return:
        (B, C, N*8)
    '''
    device = feat.device
    B, C, D, H, W = feat.size()
    _, _, N = pts.size()

    x, y, z = pts[:,0], pts[:,1], pts[:,2]
    x = (W-1.0) * (0.5 * x.contiguous().view(-1) + 0.5)
    y = (H-1.0) * (0.5 * y.contiguous().view(-1) + 0.5)
    z = (D-1.0) * (0.5 * z.contiguous().view(-1) + 0.5)

    x0 = torch.floor(x).int()
    y0 = torch.floor(y).int()
    z0 = torch.floor(z).int()

    s = x - x0.float()
    t = y - y0.float()
    v = z - z0.float()

    s_side = (s >= 0.5)
    t_side = (t >= 0.5)
    v_side = (v >= 0.5)

    xn = x0[:,None].expand(-1,2).contiguous()
    yn = y0[:,None].expand(-1,2).contiguous()
    zn = z0[:,None].expand(-1,2).contiguous()

    s = s[:,None].expand(-1,2).contiguous()
    t = t[:,None].expand(-1,2).contiguous()
    v = v[:,None].expand(-1,2).contiguous()

    xn[s_side,1] += 1
    xn[~s_side,1] -= 1
    yn[t_side,1] += 1
    yn[~t_side,1] -= 1
    zn[v_side,1] += 1
    zn[~v_side,1] -= 1

    s[s_side,1] -= 1.0
    s[~s_side,1] += 1.0
    t[t_side,1] -= 1.0
    t[~t_side,1] += 1.0
    v[v_side,1] -= 1.0
    v[~v_side,1] += 1.0

    s = s[:,None,None,:].expand(-1,2,2,-1).contiguous().view(-1)
    t = t[:,None,:,None].expand(-1,2,-1,2).contiguous().view(-1)
    v = v[:,:,None,None].expand(-1,-1,2,2).contiguous().view(-1)

    max_x = W - 1
    max_y = H - 1
    max_z = D - 1

    xn_clamp = torch.clamp(xn, 0, max_x)
    yn_clamp = torch.clamp(yn, 0, max_y)
    zn_clamp = torch.clamp(zn, 0, max_z)

    dim3 = W
    dim2 = W * H
    dim1 = W * H * D

    xn_valid = (xn <= max_x) & (xn >= 0)
    yn_valid = (yn <= max_y) & (yn >= 0)
    zn_valid = (zn <= max_z) & (zn >= 0)

    base = (dim1 * torch.arange(B).int()).view(B, 1).expand(B, N).contiguous().view(-1).to(device)

    base_zn_yn = base[:,None,None,None] + zn_clamp[:,:,None,None] * dim2 + yn_clamp[:,None,:,None] * dim3
    idx_zn_yn_xn = base_zn_yn + xn_clamp[:,None,None,:] # (BN, 2, 2, 2)

    # (B,C,D,H,W) -> (B,D,H,W,C)
    im_flat = feat.permute(0,2,3,4,1).contiguous().view(-1, C)
    i_z0_y0_x0 = torch.gather(im_flat, 0, idx_zn_yn_xn.view(-1,1).expand(-1,C).long())

    w_z0_y0_x0 = ((xn_valid[:,None,None,:] * yn_valid[:,None,:,None] * zn_valid[:,:,None,None]).float()).view(-1,1)

    stv = torch.stack([s.view(B,-1), t.view(B,-1), v.view(B,-1)], 1)

    output = (w_z0_y0_x0 * i_z0_y0_x0).view(B, N*8, C).permute(0,2,1).contiguous()

    return output, stv-0.5 # (-1.0, 1.0)

def index3d_nearest_boundary(feat, pts):
    '''
    args:
        feat: (B, C, D, H, W)
        pts: (B, 3, N)
    return:
        sampled feature (B, C, N*6)
        stv (B, 3, N*6)
    '''
    device = feat.device
    B, C, D, H, W = feat.size()
    _, _, N = pts.size()

    x, y, z = pts[:,0], pts[:,1], pts[:,2]
    x = (W-1.0) * (0.5 * x.contiguous().view(-1) + 0.5)
    y = (H-1.0) * (0.5 * y.contiguous().view(-1) + 0.5)
    z = (D-1.0) * (0.5 * z.contiguous().view(-1) + 0.5)

    x0 = torch.floor(x).int()
    y0 = torch.floor(y).int()
    z0 = torch.floor(z).int()

    s = x - x0.float()
    t = y - y0.float()
    v = z - z0.float()

    s_side = (s >= 0.5)
    t_side = (t >= 0.5)
    v_side = (v >= 0.5)

    s_mid = torch.stack([s, s_side.float()], -1)
    t_mid = torch.stack([t, t_side.float()], -1)
    v_mid = torch.stack([v, v_side.float()], -1)

    s_mid = s_mid[:,[1,0,0]]
    t_mid = t_mid[:,[0,1,0]]
    v_mid = v_mid[:,[0,0,1]]

    xn = x0[:,None].expand(-1,2).contiguous()
    yn = y0[:,None].expand(-1,2).contiguous()
    zn = z0[:,None].expand(-1,2).contiguous()

    s = s[:,None].expand(-1,2).contiguous()
    t = t[:,None].expand(-1,2).contiguous()
    v = v[:,None].expand(-1,2).contiguous()

    xn[s_side,1] += 1
    xn[~s_side,1] -= 1
    yn[t_side,1] += 1
    yn[~t_side,1] -= 1
    zn[v_side,1] += 1
    zn[~v_side,1] -= 1

    s[s_side,1] = 0.0
    s[~s_side,1] = 1.0
    t[t_side,1] = 0.0
    t[~t_side,1] = 1.0
    v[v_side,1] = 0.0
    v[~v_side,1] = 1.0

    s = s[:,[1,0,0]]
    t = t[:,[0,1,0]]
    v = v[:,[0,0,1]]

    max_x = W - 1
    max_y = H - 1
    max_z = D - 1

    xn_clamp = torch.clamp(xn, 0, max_x)
    yn_clamp = torch.clamp(yn, 0, max_y)
    zn_clamp = torch.clamp(zn, 0, max_z)

    dim3 = W
    dim2 = W * H
    dim1 = W * H * D

    xn_valid = (xn <= max_x) & (xn >= 0)
    yn_valid = (yn <= max_y) & (yn >= 0)
    zn_valid = (zn <= max_z) & (zn >= 0)

    base = (dim1 * torch.arange(B).int()).view(B, 1).expand(B, N).contiguous().view(-1).to(device)

    base_zn_yn = base[:,None] + zn_clamp[:,[0,0,0,1]] * dim2 + yn_clamp[:,[0,0,1,0]] * dim3
    idx_zn_yn_xn = base_zn_yn + xn_clamp[:,[0,1,0,0]] # (BN, 4)

    # (B,C,D,H,W) -> (B,D,H,W,C)
    im_flat = feat.permute(0,2,3,4,1).contiguous().view(-1, C)
    i_z0_y0_x0 = torch.gather(im_flat, 0, idx_zn_yn_xn.view(-1,1).expand(-1,C).long())
    w_z0_y0_x0 = ((xn_valid[:,[0,1,0,0]] * yn_valid[:,[0,0,1,0]] * zn_valid[:,[0,0,0,1]]).float()).view(-1,1)

    output = (w_z0_y0_x0 * i_z0_y0_x0).view(B,N,4,C).permute(0,3,1,2)

    out_mid = output[:,:,:,:1].expand(-1,-1,-1,3)
    out_bound = output[:,:,:,1:]

    s = torch.cat([s_mid, s], -1)
    t = torch.cat([t_mid, t], -1)
    v = torch.cat([v_mid, v], -1)

    stv = torch.stack([s.view(B,-1), t.view(B,-1), v.view(B,-1)], 1)
    output = torch.cat([out_mid, out_bound], -1)
    output = output.view(B,C,-1)

    return output, stv-0.5 # (-1.0, 1.0)

def index(feat, uv, mode='bilinear'):
    '''
    :param feat: [B, C, H, W] image features
    :param uv: [B, 2, N] uv coordinates in the image plane, range [-1, 1]
    :return: [B, C, N] image features at the uv coordinates
    '''
    uv = uv.transpose(1, 2)  # [B, N, 2]
    uv = uv.unsqueeze(2)  # [B, N, 1, 2]
    samples = torch.nn.functional.grid_sample(feat, uv, align_corners=True, mode=mode)  # [B, C, N, 1]
    return samples[:, :, :, 0]  # [B, C, N]

def index3d(feat, pts, mode='bilinear'):
    '''
    :param feat: [B, C, D, H, W] image features
    :param pts: [B, 3, N] normalized 3d coordinates, range [-1, 1]
    :return: [B, C, N] image features at the pts coordinates
    '''
    pts = pts.transpose(1, 2)  # [B, N, 3]
    pts = pts[:,:,None,None]  # [B, N, 1, 1, 3]
    samples = torch.nn.functional.grid_sample(feat, pts, align_corners=True, mode=mode)  # [B, C, N, 1, 1]
    return samples[:, :, :, 0, 0]  # [B, C, N]

def orthogonal(points, calibrations):
    '''
    Compute the orthogonal projections of 3D points into the image plane by given projection matrix
    :param points: [B, 3, N] Tensor of 3D points
    :param calibrations: [B, 3, 4] Tensor of projection matrix
    :param transforms: [B, 2, 3] Tensor of image transform matrix
    :return: xyz: [B, 3, N] Tensor of xyz coordinates in the image plane
    '''
    rot = calibrations[:, :3, :3]
    trans = calibrations[:, :3, 3:4]
    pts = torch.baddbmm(trans, rot, points)  # [B, 3, N]
    return pts


def perspective(points, calibrations):
    '''
    Compute the perspective projections of 3D points into the image plane by given projection matrix
    :param points: [Bx3xN] Tensor of 3D points
    :param calibrations: [Bx3x4] Tensor of projection matrix
    :param transforms: [Bx2x3] Tensor of image transform matrix
    :return: xy: [Bx2xN] Tensor of xy coordinates in the image plane
    '''
    rot = calibrations[:, :3, :3]
    trans = calibrations[:, :3, 3:4]
    homo = torch.baddbmm(trans, rot, points)  # [B, 3, N]
    xy = homo[:, :2, :] / homo[:, 2:3, :]
    xyz = torch.cat([xy, homo[:, 2:3, :]], 1)
    return xyz
