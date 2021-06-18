import open3d as o3d
import numpy as np

import os
from os.path import isdir, isfile, join

import json

import argparse
import subprocess

from tqdm import tqdm
import time
import random
import subprocess

import matplotlib.pyplot as plt

def render_single_image(result_mesh_file, output_image_file, vis, yprs, raw_color = True):
    # Render result image
    mesh = o3d.io.read_triangle_mesh(result_mesh_file)
    mesh.compute_vertex_normals()
    if not raw_color:
        mesh.paint_uniform_color([0.7, 0.7, 0.7])
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices) - y_axis_offset)

    vis.add_geometry(mesh)
    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(cam_params)
    
    for ypr in yprs:
        ctr.rotate(0, RENDER_RESOLUTION/180*ypr[1])
        ctr.rotate(RENDER_RESOLUTION/180*ypr[0], 0)

    vis.poll_events()
    vis.update_renderer()
    # time_stamp_result_image = str(time.time())
    # output_result_image_file = join(output_dir, result_name+'_'+time_stamp_result_image+'.png')
    # vis.capture_screen_image(output_result_image_file, True)
    result_image = vis.capture_screen_float_buffer(False)
    vis.clear_geometries()

    result_img = np.asarray(result_image)

    plt.imsave(output_image_file, np.asarray(result_img), dpi = 1)

    return result_img


y_axis_offset = np.array([0.0, 1.25, 0.0])
# o3d.utility.set_verbosity_level(o3d.cpu.pybind.utility.VerbosityLevel(1))
random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_dir', type=str, required=True, help='result directory')
parser.add_argument('-o', '--out_dir', type=str, default='demo_result', help='Output directory or filename')
parser.add_argument('-v', '--video_name', type=str, default='video', help='Output directory or filename')
parser.add_argument('-n', '--num', type=int, default=0, help='Output directory or filename')

args = parser.parse_args()

input_dirs = [args.input_dir]

vis = o3d.visualization.Visualizer()
RENDER_RESOLUTION = 512
FOCAL_LENGTH = 1.5
vis.create_window(width=RENDER_RESOLUTION, height=RENDER_RESOLUTION)

opt = vis.get_render_option()
# opt.background_color = np.asarray([0, 0, 0])
# opt.mesh_color_option = o3d.visualization.MeshColorOption.Normal
# opt.mesh_shade_option = o3d.visualization.MeshShadeOption.Color
# opt.mesh_show_wireframe = True
opt.light_on = True

for dir_index in tqdm(range(len(input_dirs))):
    args.input_dir = input_dirs[dir_index]

    input_dir = args.input_dir
    output_dir = args.out_dir
    if output_dir == '':
        output_dir = input_dir
    output_dir = output_dir[:-1] if output_dir[-1] == '/' else output_dir
    video_name = args.video_name

    cam_intrinsics = o3d.camera.PinholeCameraIntrinsic()
    INTRINSIC = np.eye(3, dtype=np.float32)
    INTRINSIC[0,0] = FOCAL_LENGTH*RENDER_RESOLUTION
    INTRINSIC[1,1] = FOCAL_LENGTH*RENDER_RESOLUTION
    INTRINSIC[0,2] = RENDER_RESOLUTION/2-0.5
    INTRINSIC[1,2] = RENDER_RESOLUTION/2-0.5
    cam_intrinsics.intrinsic_matrix = INTRINSIC
    # print(cam_intrinsics.intrinsic_matrix)

    cam_intrinsics.width = RENDER_RESOLUTION
    cam_intrinsics.height = RENDER_RESOLUTION


    EXTRINSIC = np.array([[ 1.0,     0.0,     0.0,     0.0],
                          [ 0.0,    -1.0,     0.0,     0.5],
                          [ 0.0,     0.0,    -1.0,     2.7],
                          [ 0.0,     0.0,     0.0,     1.0]])
    cam_params = o3d.camera.PinholeCameraParameters()
    cam_params.intrinsic = cam_intrinsics
    cam_params.extrinsic = EXTRINSIC

    if isdir(join(output_dir, 'tmp')):
        tmp_dir = join(output_dir, 'tmp')
        command = 'rm -rf ' + tmp_dir
        subprocess.run(command, shell=True, stdout=subprocess.DEVNULL)
    tmp_dir = join(output_dir, 'tmp')
    os.makedirs(tmp_dir, exist_ok = True)

    has_color = True if 'color' in input_dir else False

    mesh_files = sorted([f for f in os.listdir(input_dir) if f[-3:]=='obj'])

    yprs = [[0, 0]]
    # if 'knocking1_poses' in input_dir:
    #     yprs = [[0, -90], [-90, 0]]
    # if 'misc_poses' in input_dir or 'misc2_poses' in input_dir:
    #     yprs = [[0, -90], [180, 0]]
    # if 'irish_dance' in input_dir:
    #     # yprs = [[0, -90]]
    #     yprs = [[0, -90], [180, 0]]

    for i, mesh_file in enumerate(tqdm(mesh_files[::1])):
        mesh_file_path = join(input_dir, mesh_file)
        output_image_file = join(tmp_dir, str(i).zfill(5)+'.png')

        _ = render_single_image(mesh_file_path, output_image_file, vis, yprs, has_color)

    if video_name == '':
        video_file = join(output_dir+'/../', output_dir.split('/')[-1]+'.mp4')
    else:
        video_file = join(output_dir, video_name+'.mp4')
    command = 'ffmpeg -r 30 -i '+join(tmp_dir,'%05d.png') + ' -c:v libx264 -vf fps=30 -pix_fmt yuv420p -y '+video_file
    subprocess.run(command, shell=True)

    command = 'rm -rf ' + tmp_dir
    subprocess.run(command, shell=True, stdout=subprocess.DEVNULL)


