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
import torch
from torch.nn import init
import torch.nn as nn
import torch.nn.functional as F
import functools

import numpy as np
from tqdm import tqdm

from pytorch3d.ops import knn_gather, knn_points

def compute_knn_feat(vsrc, vtar, vfeat, K=1):
    dist, idx, Vnn = knn_points(vsrc, vtar, K=K, return_nn=True)
    return knn_gather(vfeat, idx)

def homogenize(v,dim=2):
    '''
    args:
        v: (B, N, C)
    return:
        (B, N, C+1)
    '''
    if dim == 2:
        return torch.cat([v, torch.ones_like(v[:,:,:1])], -1)
    elif dim == 1:
        return torch.cat([v, torch.ones_like(v[:,:1,:])], 1)
    else:
        raise NotImplementedError('unsupported homogenize dimension [%d]' % dim)

def transform_normal(net, x, n):
    '''
    args:
        flow network that returns (B, 3, N)
        x: (B, N, 3)
        n: (B, N, 3)
    '''
    x = x.permute(0,2,1)
    with torch.enable_grad():
        x.requires_grad_()

        pred = net.query(x)

        dfdx = autograd.grad(
                [pred.sum()], [x], 
                create_graph=True, retain_graph=True, only_inputs=True)[0]
        print(dfdx.shape)
        # torch.einsum('bc')           
        #     if normalize:
        #         normal = F.normalize(normal, dim=1, eps=1e-6)

def get_posemap(map_type, n_joints, parents, n_traverse=1, normalize=True):
    pose_map = torch.zeros(n_joints,n_joints-1)
    if map_type == 'parent':
        for i in range(n_joints-1):
            pose_map[i+1,i] = 1.0
    elif map_type == 'children':
        for i in range(n_joints-1):
            parent = parents[i+1]
            for j in range(n_traverse):
                pose_map[parent, i] += 1.0
                if parent == 0:
                    break
                parent = parents[parent]
        if normalize:
            pose_map /= pose_map.sum(0,keepdim=True)+1e-16
    elif map_type == 'both':
        for i in range(n_joints-1):
            pose_map[i+1,i] += 1.0
            parent = parents[i+1]
            for j in range(n_traverse):
                pose_map[parent, i] += 1.0
                if parent == 0:
                    break
                parent = parents[parent]
        if normalize:
            pose_map /= pose_map.sum(0,keepdim=True)+1e-16
    else:
        raise NotImplementedError('unsupported pose map type [%s]' % map_type)
    return pose_map

def batch_rot2euler(R):
    '''
    args:
        Rs: (B, 3, 3)
    return:
        (B, 3) euler angle (x, y, z)
    '''
    sy = torch.sqrt(R[:,0,0] * R[:,0,0] +  R[:,1,0] * R[:,1,0])
    singular = (sy < 1e-6).float()[:,None]

    x = torch.atan2(R[:,2,1] , R[:,2,2])
    y = torch.atan2(-R[:,2,0], sy)
    z = torch.atan2(R[:,1,0], R[:,0,0])
    euler = torch.stack([x,y,z],1)

    euler_s = euler.clone()
    euler_s[:,0] = torch.atan2(-R[:,1,2], R[:,1,1])
    euler_s[:,1] = torch.atan2(-R[:,2,0], sy)
    euler_s[:,2] = 0

    return (1.0-singular)*euler + singular * euler_s


def batch_rod2euler(rot_vecs):
    R = batch_rodrigues(rot_vecs)
    return batch_rot2euler(R)

def batch_rod2quat(rot_vecs):
    batch_size = rot_vecs.shape[0]

    angle = torch.norm(rot_vecs + 1e-16, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.cos(angle / 2)
    sin = torch.sin(angle / 2)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)

    qx = rx * sin
    qy = ry * sin
    qz = rz * sin
    qw = cos-1.0

    return torch.cat([qx,qy,qz,qw], dim=1)

def batch_rodrigues(rot_vecs, epsilon=1e-8, dtype=torch.float32):
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat

def quat_to_matrix(rvec):
    '''
    args:
        rvec: (B, N, 4)
    '''
    B, N, _ = rvec.size()

    theta = torch.sqrt(1e-5 + torch.sum(rvec ** 2, dim=2))
    rvec = rvec / theta[:, :, None]
    return torch.stack((
        1. - 2. * rvec[:, :, 1] ** 2 - 2. * rvec[:, :, 2] ** 2,
        2. * (rvec[:, :, 0] * rvec[:, :, 1] - rvec[:, :, 2] * rvec[:, :, 3]),
        2. * (rvec[:, :, 0] * rvec[:, :, 2] + rvec[:, :, 1] * rvec[:, :, 3]),

        2. * (rvec[:, :, 0] * rvec[:, :, 1] + rvec[:, :, 2] * rvec[:, :, 3]),
        1. - 2. * rvec[:, :, 0] ** 2 - 2. * rvec[:, :, 2] ** 2,
        2. * (rvec[:, :, 1] * rvec[:, :, 2] - rvec[:, :, 0] * rvec[:, :, 3]),

        2. * (rvec[:, :, 0] * rvec[:, :, 2] - rvec[:, :, 1] * rvec[:, :, 3]),
        2. * (rvec[:, :, 0] * rvec[:, :, 3] + rvec[:, :, 1] * rvec[:, :, 2]),
        1. - 2. * rvec[:, :, 0] ** 2 - 2. * rvec[:, :, 1] ** 2
        ), dim=2).view(B, N, 3, 3)

def rot6d_to_matrix(rot6d):
    '''
    args:
        rot6d: (B, N, 6)
    return:
        rotation matrix: (B, N, 3, 3)
    '''
    x_raw = rot6d[:,:,0:3]
    y_raw = rot6d[:,:,3:6]
        
    x = F.normalize(x_raw, dim=2)
    z = torch.cross(x, y_raw, dim=2)
    z = F.normalize(z, dim=2)
    y = torch.cross(z, x, dim=2)
        
    rotmat = torch.cat((x[:,:,:,None],y[:,:,:,None],z[:,:,:,None]), -1) # (B, 3, 3)
    
    return rotmat

def compute_affinemat(param, rot_dim):
    '''
    args:
        param: (B, N, 9/12)
    return:
        (B, N, 4, 4)
    '''
    B, N, C = param.size()
    rot = param[:,:,:rot_dim]

    if C - rot_dim == 3:
        trans = param[:,:,rot_dim:]
        scale = torch.ones_like(trans)
    elif C - rot_dim == 6:
        trans = param[:,:,rot_dim:(rot_dim+3)]
        scale = param[:,:,(rot_dim+3):]
    else:
        raise ValueError('unsupported dimension [%d]' % C)
    
    if rot_dim == 3:
        rotmat = batch_rodrigues(rot)
    elif rot_dim == 4:
        rotmat = quat_to_matrix(rot)
    elif rot_dim == 6:
        rotmat = rot6d_to_matrix(rot)
    else:
        raise NotImplementedError('unsupported rot dimension [%d]' % rot_dim)
    
    A = torch.eye(4)[None,None].to(param.device).expand(B, N, -1, -1).contiguous()
    A[:,:,:3, 3] = trans # (B, N, 3, 1)
    A[:,:,:3,:3] = rotmat * scale[:,:,None,:] # (B, N, 3, 3)

    return A

def compositional_affine(param, num_comp, rot_dim):
    '''
    args:
        param: (B, N, M*(9/12)+M)
    return:
        (B, N, 4, 4)
    '''
    B, N, _ = param.size()

    weight = torch.exp(param[:,:,:num_comp])[:,:,:,None,None]

    affine_param = param[:,:,num_comp:].reshape(B, N*num_comp, -1)
    A = compute_affinemat(affine_param, rot_dim).view(B, N, num_comp, 4, 4)

    return (weight * A).sum(2) / weight.sum(dim=2).clamp(min=0.001)


class Embedder:
    def __init__(self, **kwargs):
        
        self.kwargs = kwargs
        self.create_embedding_fn()
        
        
    def create_embedding_fn(self):
        
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

def get_embedder(multires, i=0, input_dims=3):
    
    if i == -1:
        return nn.Identity(), input_dims
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : input_dims,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim

class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        #inlining this saves 1 second per epoch (V100 GPU) vs having a temp x and then returning x(!)
        return x *( torch.tanh(F.softplus(x)))

class Sin(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)

def conv3x3(in_planes, out_planes, strd=1, padding=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=strd, padding=padding, bias=bias)

def conv3x3x3(in_planes, out_planes, strd=1, padding=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv3d(in_planes, out_planes, kernel_size=3,
                     stride=strd, padding=padding, bias=bias)

def init_mlp_siren(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        d_in = m.weight.data.size()[1]
        init.uniform_(m.weight.data, -math.sqrt(6/d_in), math.sqrt(6/d_in))
        if hasattr(m, 'bias') and m.bias is not None:
            init.constant_(m.bias.data, 0.0)

# From IGR paper
def init_mlp_geometric(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        d_out = m.weight.data.size()[0]
        if d_out == 1:
            d_in = m.weight.data.size()[1]
            init.constant_(m.weight.data, math.sqrt(math.pi/d_in))
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, -1.0)
        else:
            init.normal_(m.weight.data, 0.0, math.sqrt(2/d_out))
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':  # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(
                *real_data.shape)
            alpha = alpha.to(device)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp  # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None

def get_norm_layer(norm_type='instance'):
    """Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'group':
        norm_layer = functools.partial(nn.GroupNorm, 32)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def load_network(network, state_dict):        
    try:
        network.load_state_dict(state_dict)
    except:   
        model_dict = network.state_dict()
        try:
            state_dict = {k: v for k, v in state_dict.items() if k in model_dict}                    
            network.load_state_dict(state_dict)
        except:
            print('Pretrained network has fewer layers; The following are not initialized:')
            for k, v in state_dict.items():                      
                if v.size() == model_dict[k].size():
                    model_dict[k] = v

            not_initialized = set()

            for k, v in model_dict.items():
                if k not in state_dict or v.size() != state_dict[k].size():
                    not_initialized.add(k.split('.')[0])
            
            print(sorted(not_initialized))
            network.load_state_dict(model_dict)  


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, norm='batch'):
        super(ConvBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, int(out_planes / 2))
        self.conv2 = conv3x3(int(out_planes / 2), int(out_planes / 4))
        self.conv3 = conv3x3(int(out_planes / 4), int(out_planes / 4))

        if norm == 'batch':
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.bn2 = nn.BatchNorm2d(int(out_planes / 2))
            self.bn3 = nn.BatchNorm2d(int(out_planes / 4))
            self.bn4 = nn.BatchNorm2d(in_planes)
        elif norm == 'group':
            self.bn1 = nn.GroupNorm(32, in_planes)
            self.bn2 = nn.GroupNorm(32, int(out_planes / 2))
            self.bn3 = nn.GroupNorm(32, int(out_planes / 4))
            self.bn4 = nn.GroupNorm(32, in_planes)
        
        if in_planes != out_planes:
            self.downsample = nn.Sequential(
                self.bn4,
                nn.ReLU(True),
                nn.Conv2d(in_planes, out_planes,
                          kernel_size=1, stride=1, bias=False),
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        out1 = self.bn1(x)
        out1 = F.relu(out1, True)
        out1 = self.conv1(out1)

        out2 = self.bn2(out1)
        out2 = F.relu(out2, True)
        out2 = self.conv2(out2)

        out3 = self.bn3(out2)
        out3 = F.relu(out3, True)
        out3 = self.conv3(out3)

        out3 = torch.cat((out1, out2, out3), 1)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out3 += residual

        return out3
  

class ConvBlock3d(nn.Module):
    def __init__(self, in_planes, out_planes, norm='batch'):
        super(ConvBlock3d, self).__init__()
        self.conv1 = conv3x3x3(in_planes, int(out_planes / 2))
        self.conv2 = conv3x3x3(int(out_planes / 2), int(out_planes / 4))
        self.conv3 = conv3x3x3(int(out_planes / 4), int(out_planes / 4))

        if norm == 'batch':
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.bn2 = nn.BatchNorm2d(int(out_planes / 2))
            self.bn3 = nn.BatchNorm2d(int(out_planes / 4))
            self.bn4 = nn.BatchNorm2d(in_planes)
        elif norm == 'group':
            self.bn1 = nn.GroupNorm(min(32,in_planes), in_planes)
            self.bn2 = nn.GroupNorm(min(32,int(out_planes / 2)), int(out_planes / 2))
            self.bn3 = nn.GroupNorm(min(32,int(out_planes / 4)), int(out_planes / 4))
            self.bn4 = nn.GroupNorm(min(32,in_planes), in_planes)
        
        if in_planes != out_planes:
            self.downsample = nn.Sequential(
                self.bn4,
                nn.ReLU(True),
                nn.Conv3d(in_planes, out_planes,
                          kernel_size=1, stride=1, bias=False),
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        out1 = self.bn1(x)
        out1 = F.relu(out1, True)
        out1 = self.conv1(out1)

        out2 = self.bn2(out1)
        out2 = F.relu(out2, True)
        out2 = self.conv2(out2)

        out3 = self.bn3(out2)
        out3 = F.relu(out3, True)
        out3 = self.conv3(out3)

        out3 = torch.cat((out1, out2, out3), 1)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out3 += residual

        return out3
  

class Unet3d(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=16, norm_layer=nn.GroupNorm):
        super(Unet3d, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock3d(ngf * 4, ngf * 4, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 4):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock3d(ngf * 4, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock3d(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock3d(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock3d(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock3d(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.GroupNorm):
        super(UnetSkipConnectionBlock3d, self).__init__()
        self.outermost = outermost
        use_bias = False
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv3d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(16, inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(16, outer_nc)

        if outermost:
            upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose3d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)