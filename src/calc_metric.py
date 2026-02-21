import numpy as np
import open3d as o3d
import trimesh
import json
import copy
import argparse

def extract_transformation(json_file):
    
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Extract rotation and translation
    R = np.array(data['Rot'])  # 3x3 rotation matrix
    t = np.array(data['t'])     # 3x1 translation vector

    # Create 4x4 transformation matrix
    trans = np.eye(4)
    trans[:3, :3] = R
    trans[:3, 3] = t

    return trans 

def apply_transform(o3d_mesh, transform):
    mesh_copy = copy.deepcopy(o3d_mesh)
    mesh_copy.transform(transform)
    return mesh_copy

def voxelize_mesh(mesh, voxel_size=0.01):
    return o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size=voxel_size)

def voxel_grid_to_pcd_fast(voxel_grid, color):
    """Convert a VoxelGrid to a lightweight colored PointCloud."""
    pcd = o3d.geometry.PointCloud()
    voxel_centers = [voxel_grid.get_voxel_center_coordinate(v.grid_index)
                     for v in voxel_grid.get_voxels()]
    pcd.points = o3d.utility.Vector3dVector(voxel_centers)
    pcd.paint_uniform_color(color)
    return pcd

def voxel_iou_visualize_fast(o3d_mesh, transform1, transform2, voxel_size=0.01):
    mesh1 = apply_transform(o3d_mesh, transform1)
    mesh2 = apply_transform(o3d_mesh, transform2)

    vox1 = voxelize_mesh(mesh1, voxel_size)
    vox2 = voxelize_mesh(mesh2, voxel_size)

    coords1 = set(map(tuple, [v.grid_index for v in vox1.get_voxels()]))
    coords2 = set(map(tuple, [v.grid_index for v in vox2.get_voxels()]))

    inter = coords1 & coords2
    union = coords1 | coords2

    iou = len(inter) / len(union) if len(union) else 0.0
    voxel_volume = voxel_size**3
    vol_inter = len(inter) * voxel_volume
    vol_union = len(union) * voxel_volume

    print(f"IoU ≈ {iou:.4f}")
    print(f"Intersection Volume ≈ {vol_inter:.6f}")
    print(f"Union Volume ≈ {vol_union:.6f}")

    # Visualization as points (lightweight)
    pcd1 = voxel_grid_to_pcd_fast(vox1, [0, 0, 1])
    pcd2 = voxel_grid_to_pcd_fast(vox2, [1, 0, 0])

    # Intersection voxels (green)
    if inter:
        inter_centers = [vox1.get_voxel_center_coordinate(v) for v in inter]
        pcd_inter = o3d.geometry.PointCloud()
        pcd_inter.points = o3d.utility.Vector3dVector(inter_centers)
        pcd_inter.paint_uniform_color([0, 1, 0])
        o3d.visualization.draw_geometries([pcd1, pcd2, pcd_inter])
    else:
        o3d.visualization.draw_geometries([pcd1, pcd2])

    return iou, vol_inter, vol_union


def mesh_3diou(mesh_path, transform1, transform2, voxel_size=0.01, fill=True):
    """
    Compute volumetric IoU between two transformed versions of a mesh
    using solid voxelization in trimesh.
    Works even for non-watertight meshes.
    """
    # Load mesh
    mesh = trimesh.load(mesh_path, force='mesh')
    if not isinstance(mesh, trimesh.Trimesh):
        # In case it's a scene (multiple parts)
        mesh = mesh.dump().sum()

    # Apply transformations
    mesh1 = mesh.copy().apply_transform(transform1)
    mesh2 = mesh.copy().apply_transform(transform2)

    # Voxelize (Trimesh automatically handles open meshes)
    vox1 = mesh1.voxelized(pitch=voxel_size)
    vox2 = mesh2.voxelized(pitch=voxel_size)

    # Fill inside (solidify) if desired
    if fill:
        vox1 = vox1.fill()
        vox2 = vox2.fill()

    # Get centers (world coordinates) of occupied voxels
    centers1 = vox1.points  # occupied voxel centers
    centers2 = vox2.points

    # Align them in same coordinate space
    min_bound = np.minimum(centers1.min(axis=0), centers2.min(axis=0))

    # Quantize centers to voxel grid indices
    idx1 = np.round((centers1 - min_bound) / voxel_size).astype(int)
    idx2 = np.round((centers2 - min_bound) / voxel_size).astype(int)

    # Convert to sets for intersection / union
    set1 = set(map(tuple, idx1))
    set2 = set(map(tuple, idx2))

    inter = set1 & set2
    union = set1 | set2

    voxel_vol = voxel_size ** 3
    vol_inter = len(inter) * voxel_vol
    vol_union = len(union) * voxel_vol
    iou = vol_inter / vol_union if vol_union > 0 else 0.0

    return iou, vol_inter, vol_union


def bbox_3diou(mesh_path, transform1, transform2):
    """
    Compute 3D IoU between bounding boxes of a mesh
    transformed by two different 4x4 matrices.

    Parameters
    ----------
    mesh_path : str
        Path to mesh file (.obj, .ply, .stl, etc.)
    transform1, transform2 : np.ndarray, shape (4,4)
        Homogeneous transformation matrices.

    Returns
    -------
    iou : float
        Intersection-over-Union of the two bounding boxes.
    inter_vol : float
        Intersection volume.
    union_vol : float
        Union volume.
    """
    # Load mesh
    mesh = trimesh.load(mesh_path, force='mesh')
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump().sum()  # merge scene parts

    # Apply transformations
    mesh1 = mesh.copy().apply_transform(transform1)
    mesh2 = mesh.copy().apply_transform(transform2)

    # Get tight axis-aligned bounding boxes
    box1 = mesh1.bounds  # [[minx, miny, minz], [maxx, maxy, maxz]]
    box2 = mesh2.bounds

    # Compute intersection box corners
    min_corner = np.maximum(box1[0], box2[0])
    max_corner = np.minimum(box1[1], box2[1])

    # Compute intersection volume
    inter_dims = np.maximum(max_corner - min_corner, 0.0)
    inter_vol = np.prod(inter_dims)

    # Compute volumes
    vol1 = np.prod(box1[1] - box1[0])
    vol2 = np.prod(box2[1] - box2[0])

    union_vol = vol1 + vol2 - inter_vol
    iou = inter_vol / union_vol if union_vol > 0 else 0.0

    return iou, inter_vol, union_vol


def calculate_pose_error(gt_json_path, pred_json_path):
    """
    Calculate 3D rotation error and 3D translation error between 
    ground truth and prediction JSON files.
    
    Args:
        gt_json_path: Path to ground truth JSON file
        pred_json_path: Path to prediction JSON file
    
    Returns:
        dict containing:
            - 'rotation_error_degrees': Rotation error in degrees (geodesic distance)
            - 'rotation_error_radians': Rotation error in radians
            - 'translation_error': Euclidean distance between translations
            - 'translation_error_per_axis': Error per axis (x, y, z)
    """
    # Load JSON files
    with open(gt_json_path, 'r') as f:
        gt_data = json.load(f)
    
    with open(pred_json_path, 'r') as f:
        pred_data = json.load(f)
    
    # Extract rotation matrices
    R_gt = np.array(gt_data['Rot'])
    R_pred = np.array(pred_data['Rot'])
    
    # Extract translation vectors
    t_gt = np.array(gt_data['t'])
    t_pred = np.array(pred_data['t'])
    
    # Calculate rotation error using geodesic distance
    # R_error = R_gt^T * R_pred
    R_error = R_gt.T @ R_pred
    
    # Rotation error in radians: arccos((trace(R_error) - 1) / 2)
    trace = np.trace(R_error)
    # Clamp to avoid numerical errors in arccos
    trace_clamped = np.clip((trace - 1) / 2, -1.0, 1.0)
    rotation_error_rad = np.arccos(trace_clamped)
    rotation_error_deg = np.degrees(rotation_error_rad)
    
    # Calculate translation error (Euclidean distance)
    translation_error = np.linalg.norm(t_gt - t_pred)
    
    # Per-axis translation error
    translation_error_per_axis = {
        'x': abs(t_gt[0] - t_pred[0]),
        'y': abs(t_gt[1] - t_pred[1]),
        'z': abs(t_gt[2] - t_pred[2])
    }
    
    return {
        'rotation_error_degrees': float(rotation_error_deg),
        'rotation_error_radians': float(rotation_error_rad),
        'translation_error': float(translation_error),
        'translation_error_per_axis': translation_error_per_axis
    }

def calc_pose_error(root_dir, txt_file):
    '''
    Calculates rotational and translation error between the groundtruth pose and predicted pose of the object
    '''

    with open(f"{root_dir}/{txt_file}", 'r') as f:
        lines = f.readlines()
    
    delta_R = []
    delta_t = []
    for line in lines:
        path = line.strip()
        gt_file = path.replace("pointclouds", "gt_poses").replace(".ply", ".json")
        pred_file = path.replace("pointclouds", "pred_poses").replace(".ply", ".json")
    
        errors = calculate_pose_error(f"{root_dir}/{gt_file}", f"{root_dir}/{pred_file}")

        delta_R.append(errors['rotation_error_degrees'])
        delta_t.append(errors['translation_error'])

    mean_delta_R = np.mean(delta_R)
    std_delta_R = np.std(delta_R)
    print(f"Mean delta R (degrees): {mean_delta_R:.4f}")
    print(f"Std Dev R (degrees): {std_delta_R:.4f}")

    mean_delta_t = np.mean(delta_t)
    std_delta_t = np.std(delta_t)
    print(f"Mean t (meters): {mean_delta_t:.4f}")
    print(f"Std Dev t (meters): {std_delta_t:.4f}")


def calc_iou(root_dir, txt_file, mesh_path):
    '''
    Calculates Mesh 3D IoU and Bounding box 3D IoU values between the groundtruth pose and predicted pose of the 3D mesh
    '''

    with open(f"{root_dir}/{txt_file}", 'r') as f:
        lines = f.readlines()
    
    mesh_iou_list = []
    bbox_iou_list = []
    for line in lines:
        path = line.strip()
        print(path)
        gt_file = path.replace("pointclouds", "gt_poses").replace(".ply", ".json")
        pred_file = path.replace("pointclouds", "pred_poses").replace(".ply", ".json")
    
        # Define transformation matrices
        transform_gt = extract_transformation(f"{root_dir}/{gt_file}")
        transform_pred = extract_transformation(f"{root_dir}/{pred_file}")
    
        # Calculate Mesh IoU --comment out if you do not wish to calculate
        iou, intersection_vol, union_vol = mesh_3diou(f"{root_dir}/{mesh_path}", transform_gt, transform_pred, voxel_size=0.01)
        mesh_iou_list.append(iou)
        print(f"Mesh IoU: {iou:.4f}")
        print(f"Mesh Intersection Volume: {intersection_vol:.4f}")
        print(f"Mesh Union Volume: {union_vol:.4f}")

        # Calculate Bbox IoU --comment out if you do not wish to calculate
        iou, intersection_vol, union_vol = bbox_3diou(f"{root_dir}/{mesh_path}", transform_gt, transform_pred)
        bbox_iou_list.append(iou)
        print(f"BBox IoU: {iou:.4f}")
        print(f"Bbox Intersection Volume: {intersection_vol:.4f}")
        print(f"Bbox Union Volume: {union_vol:.4f}")
    
    print("Mean Mesh IoU:", np.mean(mesh_iou_list))
    print("Mean Bbox IoU:", np.mean(bbox_iou_list))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Calculate quantitative metrics')

    parser.add_argument('TXT_FILE', help='txt file with pointcloud file paths')
    parser.add_argument('MESH_PATH', help='obj file of the equipment')
    parser.add_argument('--ROOT_DIR', help='Input data directory path', default=".")

    args = parser.parse_args()

    root_dir = args.ROOT_DIR #"/mnt/data2/Ankita/perception_evaluation/"
    txt_file = args.TXT_FILE #"dataset/ortable_pcd.txt"
    mesh_path = args.MESH_PATH #"models_scaled/or_table/or_table.obj"

    calc_pose_error(root_dir, txt_file)
    calc_iou(root_dir, txt_file, mesh_path)
    
