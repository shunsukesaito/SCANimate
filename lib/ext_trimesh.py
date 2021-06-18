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

import numpy as np

from trimesh.sample import util
from trimesh.sample import transformations
import trimesh
import torch

def slice_mesh_plane_with_mesh_color(mesh, plane_normal, plane_origin):
    """
    Slice a mesh with vertex color into two by a plane, 
    return the half indicated by the normal direction.
    
    Parameters
    mesh: trimesh.Trimesh
        Mesh with mesh.visual.vertex_colors
    plane_normal: (3,) float
        Normal direction of the slice plane
    plane_origion: (3,) float
        One point on the slice plane
    """

    dots = np.einsum('i,ij->j', plane_normal,
                         (mesh.vertices - plane_origin).T)[mesh.faces]

    # Find vertex orientations w.r.t. faces for all triangles:
    #  -1 -> vertex "inside" plane (positive normal direction)
    #   0 -> vertex on plane
    #   1 -> vertex "outside" plane (negative normal direction)
    signs = np.zeros(mesh.faces.shape, dtype=np.int8)
    signs[dots < -1e-8] = 1
    signs[dots > 1e-8] = -1
    signs[np.logical_and(dots >= -1e-8, dots <= 1e-8)] = 0

    # Find all triangles that intersect this plane
    # onedge <- indices of all triangles intersecting the plane
    # inside <- indices of all triangles "inside" the plane (positive normal)
    signs_sum = signs.sum(axis=1, dtype=np.int8)
    signs_asum = np.abs(signs).sum(axis=1, dtype=np.int8)

    # Cases:
    # (0,0,0),  (-1,0,0),  (-1,-1,0), (-1,-1,-1) <- inside
    # (1,0,0),  (1,1,0),   (1,1,1)               <- outside
    # (1,0,-1), (1,-1,-1), (1,1,-1)              <- onedge
    # onedge = np.logical_and(signs_asum >= 2,
    #                         np.abs(signs_sum) <= 1)
    inside = (signs_sum == -signs_asum)

    # Automatically include all faces that are "inside"
    new_faces = mesh.faces[inside]

    selected_vertex_ids = np.unique(new_faces)

    new_vertices = mesh.vertices[selected_vertex_ids]
    new_vertex_colors = mesh.visual.vertex_colors[selected_vertex_ids]

    old_vid2new_vid = np.zeros((mesh.vertices.shape[0]), dtype = np.int64) - 1
    for new_vid, old_vid in enumerate(selected_vertex_ids):
        old_vid2new_vid[old_vid] = new_vid
    new_faces = old_vid2new_vid[new_faces]

    half_mesh = trimesh.Trimesh(vertices = new_vertices, faces = new_faces, process=False)
    half_mesh.visual.vertex_colors[:,:] = new_vertex_colors[:,:]

    return half_mesh

def slice_mesh_plane_with_texture_coordinates(mesh, plane_normal, plane_origin):
    """
    Slice a mesh with vertex color into two by a plane, 
    return the half indicated by the normal direction.
    
    Parameters
    mesh: trimesh.Trimesh
        Mesh with mesh.visual.vertex_colors
    plane_normal: (3,) float
        Normal direction of the slice plane
    plane_origion: (3,) float
        One point on the slice plane
    """

    dots = np.einsum('i,ij->j', plane_normal,
                         (mesh.vertices - plane_origin).T)[mesh.faces]

    # Find vertex orientations w.r.t. faces for all triangles:
    #  -1 -> vertex "inside" plane (positive normal direction)
    #   0 -> vertex on plane
    #   1 -> vertex "outside" plane (negative normal direction)
    signs = np.zeros(mesh.faces.shape, dtype=np.int8)
    signs[dots < -1e-8] = 1
    signs[dots > 1e-8] = -1
    signs[np.logical_and(dots >= -1e-8, dots <= 1e-8)] = 0

    # Find all triangles that intersect this plane
    # onedge <- indices of all triangles intersecting the plane
    # inside <- indices of all triangles "inside" the plane (positive normal)
    signs_sum = signs.sum(axis=1, dtype=np.int8)
    signs_asum = np.abs(signs).sum(axis=1, dtype=np.int8)

    # Cases:
    # (0,0,0),  (-1,0,0),  (-1,-1,0), (-1,-1,-1) <- inside
    # (1,0,0),  (1,1,0),   (1,1,1)               <- outside
    # (1,0,-1), (1,-1,-1), (1,1,-1)              <- onedge
    # onedge = np.logical_and(signs_asum >= 2,
    #                         np.abs(signs_sum) <= 1)
    inside = (signs_sum == -signs_asum)

    # Automatically include all faces that are "inside"
    new_faces = mesh.faces[inside]

    selected_vertex_ids = np.unique(new_faces)

    new_vertices = mesh.vertices[selected_vertex_ids]
    new_uv = mesh.visual.uv[selected_vertex_ids]

    old_vid2new_vid = np.zeros((mesh.vertices.shape[0]), dtype = np.int64) - 1
    for new_vid, old_vid in enumerate(selected_vertex_ids):
        old_vid2new_vid[old_vid] = new_vid
    new_faces = old_vid2new_vid[new_faces]

    half_mesh = trimesh.Trimesh(vertices = new_vertices, faces = new_faces, process=False)

    return half_mesh, new_uv

def sample_surface_wnormal(mesh, count, mask=None):
    """
    Sample the surface of a mesh, returning the specified
    number of points
    For individual triangle sampling uses this method:
    http://mathworld.wolfram.com/TrianglePointPicking.html
    Parameters
    ---------
    mesh : trimesh.Trimesh
      Geometry to sample the surface of
    count : int
      Number of points to return
    Returns
    ---------
    samples : (count, 3) float
      Points in space on the surface of mesh
    face_index : (count,) int
      Indices of faces for each sampled point
    """

    # len(mesh.faces) float, array of the areas
    # of each face of the mesh
    area = mesh.area_faces
    if mask is not None:
      area = mask * area
    # total area (float)
    area_sum = np.sum(area)
    # cumulative area (len(mesh.faces))
    area_cum = np.cumsum(area)
    face_pick = np.random.random(count) * area_sum
    face_index = np.searchsorted(area_cum, face_pick)

    # pull triangles into the form of an origin + 2 vectors
    tri_origins = mesh.triangles[:, 0]
    tri_vectors = mesh.triangles[:, 1:].copy()
    tri_vectors -= np.tile(tri_origins, (1, 2)).reshape((-1, 2, 3))

    # do the same for normal
    normals = mesh.vertex_normals.view(np.ndarray)[mesh.faces]
    nml_origins = normals[:, 0]
    nml_vectors = normals[:, 1:]#.copy()
    nml_vectors -= np.tile(nml_origins, (1, 2)).reshape((-1, 2, 3))

    # pull the vectors for the faces we are going to sample from
    tri_origins = tri_origins[face_index]
    tri_vectors = tri_vectors[face_index]

    # pull the vectors for the faces we are going to sample from
    nml_origins = nml_origins[face_index]
    nml_vectors = nml_vectors[face_index]

    # randomly generate two 0-1 scalar components to multiply edge vectors by
    random_lengths = np.random.random((len(tri_vectors), 2, 1))

    # points will be distributed on a quadrilateral if we use 2 0-1 samples
    # if the two scalar components sum less than 1.0 the point will be
    # inside the triangle, so we find vectors longer than 1.0 and
    # transform them to be inside the triangle
    random_test = random_lengths.sum(axis=1).reshape(-1) > 1.0
    random_lengths[random_test] -= 1.0
    random_lengths = np.abs(random_lengths)

    # multiply triangle edge vectors by the random lengths and sum
    sample_vector = (tri_vectors * random_lengths).sum(axis=1)
    sample_normal = (nml_vectors * random_lengths).sum(axis=1)

    # finally, offset by the origin to generate
    # (n,3) points in space on the triangle
    samples = sample_vector + tri_origins

    normals = sample_normal + nml_origins

    return samples, normals, face_index

def sample_surface_wnormalcolor(mesh, count, mask=None):
    """
    Sample the surface of a mesh, returning the specified
    number of points
    For individual triangle sampling uses this method:
    http://mathworld.wolfram.com/TrianglePointPicking.html
    Parameters
    ---------
    mesh : trimesh.Trimesh
      Geometry to sample the surface of
    count : int
      Number of points to return
    Returns
    ---------
    samples : (count, 3) float
      Points in space on the surface of mesh
    face_index : (count,) int
      Indices of faces for each sampled point
    """

    # len(mesh.faces) float, array of the areas
    # of each face of the mesh
    area = mesh.area_faces
    if mask is not None:
      area = mask * area
    # total area (float)
    area_sum = np.sum(area)
    # cumulative area (len(mesh.faces))
    area_cum = np.cumsum(area)
    face_pick = np.random.random(count) * area_sum
    face_index = np.searchsorted(area_cum, face_pick)

    # pull triangles into the form of an origin + 2 vectors
    tri_origins = mesh.triangles[:, 0]
    tri_vectors = mesh.triangles[:, 1:].copy()
    tri_vectors -= np.tile(tri_origins, (1, 2)).reshape((-1, 2, 3))

    # do the same for normal
    normals = mesh.vertex_normals.view(np.ndarray)[mesh.faces]
    nml_origins = normals[:, 0]
    nml_vectors = normals[:, 1:]#.copy()
    nml_vectors -= np.tile(nml_origins, (1, 2)).reshape((-1, 2, 3))

    colors = mesh.visual.vertex_colors[:,:3].astype(np.float32)
    colors = colors / 255.0
    colors = colors.view(np.ndarray)[mesh.faces]
    clr_origins = colors[:, 0]
    clr_vectors = colors[:, 1:]#.copy()
    clr_vectors -= np.tile(clr_origins, (1, 2)).reshape((-1, 2, 3))

    # pull the vectors for the faces we are going to sample from
    tri_origins = tri_origins[face_index]
    tri_vectors = tri_vectors[face_index]

    # pull the vectors for the faces we are going to sample from
    nml_origins = nml_origins[face_index]
    nml_vectors = nml_vectors[face_index]

    clr_origins = clr_origins[face_index]
    clr_vectors = clr_vectors[face_index]

    # randomly generate two 0-1 scalar components to multiply edge vectors by
    random_lengths = np.random.random((len(tri_vectors), 2, 1))

    # points will be distributed on a quadrilateral if we use 2 0-1 samples
    # if the two scalar components sum less than 1.0 the point will be
    # inside the triangle, so we find vectors longer than 1.0 and
    # transform them to be inside the triangle
    random_test = random_lengths.sum(axis=1).reshape(-1) > 1.0
    random_lengths[random_test] -= 1.0
    random_lengths = np.abs(random_lengths)

    # multiply triangle edge vectors by the random lengths and sum
    sample_vector = (tri_vectors * random_lengths).sum(axis=1)
    sample_normal = (nml_vectors * random_lengths).sum(axis=1)
    sample_color = (clr_vectors * random_lengths).sum(axis=1)

    # finally, offset by the origin to generate
    # (n,3) points in space on the triangle
    samples = sample_vector + tri_origins

    normals = sample_normal + nml_origins

    colors = sample_color + clr_origins

    # mesh.export('train_sample_before_sample.ply')
    # print(mesh.visual.vertex_colors.dtype)
    # tmp_mesh = trimesh.Trimesh(vertices = samples, faces = np.zeros((0,3), dtype = np.int64), process=False)
    # tmp_mesh.visual.vertex_colors[:,:3] = (colors[:,:]*255.0).astype(np.int8)
    # tmp_mesh.export('train_sample.ply')
    # exit()

    return samples, normals, colors, face_index

