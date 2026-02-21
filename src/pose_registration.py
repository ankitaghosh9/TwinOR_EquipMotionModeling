import open3d as o3d
import argparse
import numpy as np
import open3d as o3d
import copy
import json
import os

def load_data(mesh_path, pointcloud_path):
    """Load mesh and point cloud from files"""
    mesh = o3d.io.read_triangle_mesh(mesh_path, enable_post_processing=True)

    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    pcd = o3d.io.read_point_cloud(pointcloud_path)
    
    return mesh, pcd

def mesh_to_pointcloud(mesh, num_points=10000):
    """Sample points from mesh surface"""
    pcd_from_mesh = mesh.sample_points_uniformly(number_of_points=num_points)
    return pcd_from_mesh

def preprocess_point_cloud(pcd, voxel_size):
    """Downsample and compute normals and features"""

    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    
    return pcd_down, pcd_fpfh

def coarse_registration_ransac(source, target, source_fpfh, target_fpfh, voxel_size):
    """Perform global registration using RANSAC"""
    distance_threshold = voxel_size * 2.0
    
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source, target, source_fpfh, target_fpfh, 
        mutual_filter=False,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 1000)
    )
    
    return result

def fine_registration_icp(source, target, initial_transform, voxel_size):
    """Perform fine registration using Point-to-Plane ICP"""
    distance_threshold = voxel_size * 0.4
    
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, initial_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
    )
    
    return result

def fine_registration_coloricp(source, target, initial_transform, voxel_size):
    """Perform fine registration using Point-to-Plane ICP"""
    max_correspondence_distance = voxel_size * 1.5
    
    # Colored ICP registration
    result = o3d.pipelines.registration.registration_colored_icp(
        source, target,
        max_correspondence_distance=max_correspondence_distance,
        init=initial_transform,
        estimation_method=o3d.pipelines.registration.TransformationEstimationForColoredICP(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-6,
            relative_rmse=1e-6,
            max_iteration=50
        )
    )
    
    return result

def register_mesh_to_pointcloud(mesh, pointcloud, voxel_size=0.05, visualize=True):
    """
    Complete registration pipeline: mesh to point cloud
    
    Args:
        mesh: Open3D TriangleMesh object
        pointcloud: Open3D PointCloud object (target)
        voxel_size: Voxel size for downsampling
        visualize: Whether to visualize results
    
    Returns:
        transformation: 4x4 transformation matrix
        registered_mesh: Transformed mesh aligned to point cloud
    """
    
    source_pcd = mesh_to_pointcloud(mesh, num_points=5000)
    
    source_down, source_fpfh = preprocess_point_cloud(source_pcd, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(pointcloud, voxel_size)
    
    coarse_result = coarse_registration_ransac(
        source_down, target_down, 
        source_fpfh, target_fpfh, 
        voxel_size
    )

    registered_mesh = copy.deepcopy(mesh)
    registered_mesh.transform(coarse_result.transformation)

    fine_result = fine_registration_coloricp(
        source_down, target_down,
        coarse_result.transformation,
        voxel_size
    )

    registered_mesh = copy.deepcopy(mesh)
    registered_mesh.transform(fine_result.transformation)

    if visualize:
        visualize_registration(mesh, registered_mesh, pointcloud)
    
    return fine_result.transformation, registered_mesh

def visualize_registration(original_mesh, registered_mesh, target_pcd):
    """Visualize original, registered mesh and target point cloud"""
    
    original_mesh_colored = copy.deepcopy(original_mesh)
    registered_mesh_colored = copy.deepcopy(registered_mesh)
    target_colored = copy.deepcopy(target_pcd)

    o3d.visualization.draw_geometries(
        [original_mesh_colored, registered_mesh_colored, target_colored],
        window_name="Registration Result",
        width=1024, height=768
    )


def rotation_matrix_to_euler(R):
    """
    Convert a 3x3 rotation matrix to Euler angles (in degrees).
    Uses XYZ convention.
    
    Args:
        R: 3x3 rotation matrix
    
    Returns:
        tuple of (x, y, z) angles in degrees
    """
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    
    singular = sy < 1e-6
    
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    
    return (np.degrees(x), np.degrees(y), np.degrees(z))

def transform_matrix_to_json(transform_matrix, filepath, frame=0, output_file="transform.json"):
    """
    Convert a 3x4 transformation matrix to JSON format.
    
    Args:
        transform_matrix: 3x4 or 4x4 numpy array or list representing the transformation
                         Format: [R | t] where R is 3x3 rotation and t is 3x1 translation
        filepath: path string for the JSON
        frame: frame number
        output_file: output JSON filename
    
    Returns:
        dict containing the JSON data
    """
    # Convert to numpy array if it's a list
    if isinstance(transform_matrix, list):
        transform_matrix = np.array(transform_matrix)
    
    # Handle 4x4 matrix (take only first 3 rows)
    if transform_matrix.shape[0] == 4:
        transform_matrix = transform_matrix[:3, :]
    
    # Extract rotation matrix (first 3x3) and translation vector (last column)
    R = transform_matrix[:3, :3]
    t = transform_matrix[:3, 3]
    
    # Convert rotation matrix to Euler angles
    angles = rotation_matrix_to_euler(R)
    
    # Create data structure
    data = {
        "filepath": filepath,
        "frame": frame,
        "angle": {
            "x": float(angles[0]),
            "y": float(angles[1]),
            "z": float(angles[2])
        },
        "location": {
            "x": float(t[0]),
            "y": float(t[1]),
            "z": float(t[2])
        },
        "Rot": R.tolist(),
        "t": t.tolist()
    }
    
    # Write to JSON file
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"JSON file created: {output_file}")
    return data

# Example usage
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Generate 6D Pose Estimations and save in JSON format')

    parser.add_argument('TXT_FILE', help='txt file with pointcloud file paths')
    parser.add_argument('MESH_PATH', help='obj file of the equipment')
    parser.add_argument('--ROOT_DIR', help='Input data directory path', default=".")

    args = parser.parse_args()

    ROOT_DIR = args.ROOT_DIR #"/mnt/data2/Ankita/perception_evaluation/"
    txt_file = args.TXT_FILE #"dataset/ortable_pcd.txt"
    mesh_path = args.MESH_PATH #"models_scaled/or_table/or_table.obj"

    with open(f"{ROOT_DIR}/{txt_file}", 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        pcd_path = line.strip()
        print(pcd_path)
        mesh, pcd = load_data(f"{ROOT_DIR}/{mesh_path}", f"{ROOT_DIR}/{pcd_path}")
        transform, registered = register_mesh_to_pointcloud(mesh, pcd, voxel_size=0.05, visualize=False)
        print(transform)

        path = pcd_path.split('/')
        path[-2] = 'pred_poses'
        path[-1] = path[-1].replace(".ply", ".json")
        filepath = "/".join(path[:-2])
        frame = path[-1][:-5]
        output_file = ROOT_DIR + "/".join(path)
        os.makedirs(ROOT_DIR + "/".join(path[:-1]), exist_ok=True)

        transform_matrix_to_json(transform, 
                                 filepath=filepath, 
                                 frame=frame, 
                                 output_file=output_file)