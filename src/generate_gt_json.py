#!/usr/bin/env python3
"""
Script to convert 3D angles and translation values into a JSON file.
Stores filepath, frame, angles, location, rotation matrix, and translation vector.
"""

import json
import numpy as np
import argparse
import os


def euler_to_rotation_matrix(angles, order='xyz'):
    """
    Convert Euler angles to a 3x3 rotation matrix.
    
    Args:
        angles: tuple/list of (x, y, z) angles in degrees
        order: rotation order (default 'xyz')
    
    Returns:
        3x3 rotation matrix as a list
    """
    # Convert degrees to radians
    x, y, z = np.radians(angles)
    
    # Rotation matrix around X axis
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(x), -np.sin(x)],
        [0, np.sin(x), np.cos(x)]
    ])
    
    # Rotation matrix around Y axis
    Ry = np.array([
        [np.cos(y), 0, np.sin(y)],
        [0, 1, 0],
        [-np.sin(y), 0, np.cos(y)]
    ])
    
    # Rotation matrix around Z axis
    Rz = np.array([
        [np.cos(z), -np.sin(z), 0],
        [np.sin(z), np.cos(z), 0],
        [0, 0, 1]
    ])
    
    # Combine rotations based on order
    if order == 'xyz':
        R = Rz @ Ry @ Rx
    elif order == 'zyx':
        R = Rx @ Ry @ Rz
    else:
        R = Rz @ Ry @ Rx  # default to xyz
    
    return R.tolist()


def create_transform_json(filepath, frame, angles, translation, output_file, rotation_order='xyz'):
    """
    Create a JSON file with transformation data.
    
    Args:
        filepath: path to the file
        frame: frame number
        angles: tuple/list of (x, y, z) rotation angles in degrees
        translation: tuple/list of (x, y, z) translation values
        output_file: path to output JSON file
        rotation_order: order of rotation application (default 'xyz')
    """
    # Calculate rotation matrix
    rotation_matrix = euler_to_rotation_matrix(angles, rotation_order)
    
    # Create data structure
    data = {
        "filepath": filepath,
        "frame": frame,
        "angle": {
            "x": angles[0],
            "y": angles[1],
            "z": angles[2]
        },
        "location": {
            "x": translation[0],
            "y": translation[1],
            "z": translation[2]
        },
        "Rot": rotation_matrix,
        "t": [translation[0], translation[1], translation[2]]
    }
    
    # Write to JSON file
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"JSON file created: {output_file}")
    return data

if __name__ == "__main__":
    
    txt_file = "dataset/ortable_pcd.txt"
    with open(txt_file, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        path = line.strip().split('/')
        path[-2] = 'gt_poses'
        path[-1] = path[-1].replace(".ply", ".json")
        #print(path)

        filepath = "/".join(path[:-2])
        print(filepath)
        frame = path[-1][:-5]
        #print(frame)

        output_file = "/".join(path)
        #print(output_file)
        os.makedirs("/".join(path[:-1]), exist_ok=True)
        
        angle_x = 0.0
        angle_y = 0.0
        angle_z = 0.0
        
        trans_x = -1.5169
        trans_y = -0.88026
        trans_z = -0.11205
        
        create_transform_json(
            filepath=filepath,
            frame=frame,
            angles=(angle_x, angle_y, angle_z),
            translation=(trans_x, trans_y, trans_z),
            output_file=output_file
        )