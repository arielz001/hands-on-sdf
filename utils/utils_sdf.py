import numpy as np
import results 
import os
import point_cloud_utils as pcu
# import data.cherry_models as cherry_models

from glob import glob
from datetime import datetime
import yaml
import pybullet as pb
import matplotlib.pyplot as plt
import trimesh


def as_mesh_trimesh(mesh):
    # Utils function to get a mesh from a trimesh.Trimesh() or trimesh.scene.Scene()
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([
            trimesh.Trimesh(vertices=m.vertices, faces=m.faces)
            for m in mesh.geometry.values()])
    else:
        mesh = mesh
    return mesh

def combine_sample_latent(samples, latent_class):
    """Combine each sample (x, y, z) with the latent code generated for this object.
    Args:
        samples: collected points, np.array of shape (N, 3)
        latent: randomly generated latent code, np.array of shape (1, args.latent_size)
    Returns:
        combined hstacked latent code and samples, np.array of shape (N, args.latent_size + 3)
    """
    latent_class_full = np.tile(latent_class, (samples.shape[0], 1))   # repeat the latent code N times for stacking
    return np.hstack((latent_class_full, samples))

def shapenet_rotate(mesh_original):
    '''In Shapenet, the front is the -Z axis with +Y still being the up axis. This function rotates the object to align with the canonical reference frame.
    Args:
        mesh_original: trimesh.Trimesh(), mesh from ShapeNet
    Returns:
        mesh: trimesh.Trimesh(), rotate mesh so that the front is the +X axis and +Y is the up axis.
    '''
    verts_original = np.array(mesh_original.vertices)

    rot_M = pb.getMatrixFromQuaternion(pb.getQuaternionFromEuler([np.pi/2, 0, -np.pi/2]))
    rot_M = np.array(rot_M).reshape(3, 3)

    rpy_BA = [np.pi / 2, 0, -np.pi / 2]  # rotation from Shapenet to canonical frame
    rot_Q = pb.getQuaternionFromEuler(rpy_BA)
    rot_M = np.array(pb.getMatrixFromQuaternion(rot_Q)).reshape(3, 3)

    verts = np.einsum('ij,kj->ki', rot_M, verts_original)

    mesh = trimesh.Trimesh(vertices=verts, faces=mesh_original.faces)

    return mesh

def visualize_sdf(samples_dict, p_surf, obj_idx):
    example_sdf = samples_dict[obj_idx]['sdf']
    example_surface_points = p_surf
    sdf_normalized = (example_sdf - np.min(example_sdf)) / (np.max(example_sdf) - np.min(example_sdf))
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111, projection='3d')
    if len(example_surface_points) != len(sdf_normalized):
        sdf_normalized = sdf_normalized[:len(example_surface_points)]
    sc = ax.scatter(example_surface_points[:, 0], example_surface_points[:, 1], example_surface_points[:, 2],
                    c=sdf_normalized, cmap='seismic', s=1)
    plt.colorbar(sc, ax=ax, label='Signed Distance')
    ax.set_title(f"SDF values for Object {obj_idx}")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()



# ===============================================
#          UTILS FOR SHAPE COMPLETION
# ===============================================

def generate_partial_pointcloud(cfg, obj_path):
    """Load mesh and generate partial point cloud. The ratio of the visible bounding box is defined in the config file.
    Args:
        cfg: config file
    Return:
        samples: np.array, shape (N, 3), where N is the number of points in the partial point cloud.
        """
    # Load mesh
    print(obj_path)
    mesh_original = as_mesh_trimesh(trimesh.load(obj_path))

    # In Shapenet, the front is the -Z axis with +Y still being the up axis. 
    # Rotate objects to align with the canonical axis. 
    mesh = shapenet_rotate(mesh_original)

    # Sample on the object surface
    samples = np.array(trimesh.sample.sample_surface(mesh, 10000)[0])

    # Infer object bounding box and collect samples on the surface of the objects when the x-axis is lower than a certain threshold t.
    # This is to simulate a partial point cloud.
    t = [cfg['x_axis_ratio_bbox'], cfg['y_axis_ratio_bbox'], cfg['z_axis_ratio_bbox']]

    v_min, v_max = mesh.bounds

    for i in range(3):
        t_max = v_min[i] + t[i] * (v_max[i] - v_min[i])
        samples = samples[samples[:, i] < t_max]
    
    return samples


def get_pointcloud(pointcloud_path):
    """Load point cloud from a .npy file specified in the config.
    Args:
        cfg: configuration file
    Returns:
        pointcloud: np.array, shape (N, 3), where N is the number of points in the point cloud.
    """
    pointcloud = np.load(pointcloud_path)  # Cargar la nube de puntos desde el archivo
    return pointcloud


