import numpy as np
import cv2
import open3d as o3d
from typing import Tuple
import json
import os
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

def extract_intr_and_extr(json_file, folder, resize_factor=0.333333):
    
    #Load json file
    with open(json_file, 'r') as f:
        json_data = json.load(f)

    cam_data = json_data[str(folder)[:-1]]
    
    Rot = np.array(cam_data["R"]).reshape(3, 3)
    T = np.array(cam_data["t"]).reshape(3, 1)
    homo = np.array([0.0, 0.0, 0.0, 1.0])
    extr = np.vstack([np.hstack([Rot, T]), homo])  # Shape: [3, 4]
    
    K = np.array(cam_data["K"]).reshape(3, 3)
    K[:-1, :] = K[:-1, :] * resize_factor
    intr = K

    return intr, extr
    

def project_mesh_to_2d(
    mesh: o3d.geometry.TriangleMesh,
    transformation_matrix: np.ndarray,
    camera_intrinsics: np.ndarray,
    camera_extrinsics: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project 3D mesh vertices onto a 2D image plane.
    
    Parameters:
    -----------
    mesh : o3d.geometry.TriangleMesh
        Open3D triangle mesh
    transformation_matrix : np.ndarray
        4x4 transformation matrix (rotation + translation)
    camera_intrinsics : np.ndarray
        3x3 camera intrinsic matrix
    camera_extrinsics : np.ndarray
        4x4 camera extrinsic matrix (world to camera transformation)
    
    Returns:
    --------
    projected_vertices : np.ndarray
        2D projected vertices, shape (N, 2)
    visible_mask : np.ndarray
        Boolean mask for vertices in front of camera
    depths : np.ndarray
        Depth values for each vertex
    """
    # Get vertices and convert to numpy array
    vertices = np.asarray(mesh.vertices)
    
    # Apply transformation to mesh
    vertices_homo = np.hstack([vertices, np.ones((vertices.shape[0], 1))])
    transformed_vertices = (transformation_matrix @ vertices_homo.T).T
    
    # Apply camera extrinsics (world to camera space)
    camera_vertices = (camera_extrinsics @ transformed_vertices.T).T
    camera_vertices_3d = camera_vertices[:, :3]
    
    # Check visibility (positive z)
    depths = camera_vertices_3d[:, 2]
    visible_mask = depths > 0
    
    # Project to image plane
    projected_homo = (camera_intrinsics @ camera_vertices_3d.T).T
    projected_vertices = np.zeros((vertices.shape[0], 2))
    projected_vertices[visible_mask] = (
        projected_homo[visible_mask, :2] / projected_homo[visible_mask, 2:3]
    )
    
    return projected_vertices, visible_mask, depths


def render_mesh_on_image(
    image: np.ndarray,
    mesh: o3d.geometry.TriangleMesh,
    projected_vertices: np.ndarray,
    visible_mask: np.ndarray,
    depths: np.ndarray,
    render_mode: str = 'wireframe',
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    Render projected mesh on image.
    
    Parameters:
    -----------
    image : np.ndarray
        Output image
    mesh : o3d.geometry.TriangleMesh
        Open3D mesh
    projected_vertices : np.ndarray
        2D projected vertices
    visible_mask : np.ndarray
        Visibility mask
    depths : np.ndarray
        Depth values
    render_mode : str
        'wireframe' or 'solid'
    color : Tuple[int, int, int]
        Color for rendering (BGR)
    thickness : int
        Line thickness for wireframe
    
    Returns:
    --------
    image : np.ndarray
        Rendered image
    """
    h, w = image.shape[:2]
    faces = np.asarray(mesh.triangles)
    
    if render_mode == 'wireframe':
        # Render wireframe
        for face in faces:
            if not all(visible_mask[face]):
                continue
            
            for i in range(3):
                v1 = projected_vertices[face[i]]
                v2 = projected_vertices[face[(i + 1) % 3]]
                
                if (0 <= v1[0] < w and 0 <= v1[1] < h and
                    0 <= v2[0] < w and 0 <= v2[1] < h):
                    pt1 = tuple(v1.astype(int))
                    pt2 = tuple(v2.astype(int))
                    cv2.line(image, pt1, pt2, color, thickness)
    elif render_mode == 'solid':
        # Sort faces by depth (painter's algorithm)
        face_depths = []
        for face in faces:
            if all(visible_mask[face]):
                avg_depth = np.mean(depths[face])
                face_depths.append((face, avg_depth))
        
        face_depths.sort(key=lambda x: x[1], reverse=True)
        
        # Render filled triangles
        for face, avg_depth in face_depths:
            pts = projected_vertices[face].astype(np.int32)
            
            if np.all((pts[:, 0] >= 0) & (pts[:, 0] < w) & 
                      (pts[:, 1] >= 0) & (pts[:, 1] < h)):
                
                # Depth-based shading
                min_depth = depths[visible_mask].min()
                max_depth = depths[visible_mask].max()
                if max_depth > min_depth:
                    depth_factor = 1.0 - (avg_depth - min_depth) / (max_depth - min_depth)
                    depth_factor = 0.3 + 0.7 * depth_factor
                else:
                    depth_factor = 1.0
                
                shaded_color = tuple(int(c * depth_factor) for c in color)
                cv2.fillPoly(image, [pts], shaded_color)
                cv2.polylines(image, [pts], True, (50, 50, 50), 1)
    
    return image


# Example usage
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Resize and process images')

    parser.add_argument('TXT_FILE', help='txt file with pointcloud file paths')
    parser.add_argument('MESH_PATH', help='obj file of the equipment')
    parser.add_argument('--ROOT_DIR', help='Input data directory path', default=".")

    args = parser.parse_args()

    root_dir = args.ROOT_DIR #"/mnt/data2/Ankita/perception_evaluation/"
    txt_file = args.TXT_FILE #"dataset/ortable_pcd.txt"
    mesh_path = args.MESH_PATH #"models_scaled/or_table/or_table.obj"
    resize_factor = 0.333333

    #Load OBJ Mesh
    mesh = o3d.io.read_triangle_mesh(f"{root_dir}/{mesh_path}")
    mesh_scaled = mesh.scale(0.97, center=mesh.get_center())
    mesh.compute_vertex_normals()
    
    with open(f"{root_dir}/{txt_file}", 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        path = line.strip()
        path_list = line.strip().split('/')

        if path_list[1][0:2] == '09' or path_list[1][0:2] == '10':
            #HALO (09_X and 10_X) folders
            json_file = f"{root_dir}/dataset/halo_calibration/ba_poses_to_world.json"
            subfolders = ['40034694L','40795974L','46026258L','47661457L']
        elif path_list[1][0:2] == '00':
            #PULM (00_X) folders
            json_file = f"{root_dir}/dataset/pulm_calibration/ba_poses_to_world.json"
            subfolders = ['41908851L', '44664489L', '45902703L', '46517772L']

        prediction_file = path.replace("pointclouds", "pred_poses").replace(".ply", ".json")
        transformation_matrix = extract_transformation(f"{root_dir}/{prediction_file}")

        for folder in subfolders:
            camera_intrinsics, camera_extrinsics = extract_intr_and_extr(json_file, folder)
            
            # Project mesh to 2D
            projected_vertices, visible_mask, depths = project_mesh_to_2d(
                mesh,
                transformation_matrix,
                camera_intrinsics,
                camera_extrinsics
            )
            
            # Create output images
            image_path = "/".join(path_list[0:2]) + "/images/" + folder + "/" + path_list[-1].replace(".ply", ".jpg")
            print(image_path)
            img = cv2.imread(f"{root_dir}/{image_path}")
            h, w, c = img.shape
            scaled_img = np.array(cv2.resize(img, (round(w * resize_factor), round(h * resize_factor))))
            image_wireframe = scaled_img.copy()
            
            # Render wireframe
            image_wireframe = render_mesh_on_image(
                image_wireframe,
                mesh,
                projected_vertices,
                visible_mask,
                depths,
                render_mode='wireframe',
                color=(0, 0, 255),
                thickness=2
            )
            output_image = cv2.addWeighted(scaled_img, 0.5, image_wireframe, 0.5, 0)
            print("rendered wireframe")
            
            # Save results
            os.makedirs(root_dir + "/".join(path_list[0:3])+"/visuals", exist_ok=True)
            wireframe_path = path.replace("pointclouds", "visuals").replace(".ply", f"_{folder}.png")
            print(f"{root_dir}/{wireframe_path}")
            cv2.imwrite(f"{root_dir}/{wireframe_path}", output_image)
            print("image saved")