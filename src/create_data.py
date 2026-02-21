import numpy as np
import cv2
from PIL import Image
import yaml
import json
from pathlib import Path
import re
import argparse
import os

def load_opencv_yaml(yaml_path):
    """
    Load YAML file with OpenCV format using cv2.FileStorage
    """
    fs = cv2.FileStorage(str(yaml_path), cv2.FILE_STORAGE_READ)
    
    data = {}
    
    # Get the root node
    root = fs.root()
    
    # Get all keys
    keys = root.keys()
    
    for key in keys:
        node = root.getNode(key)
        
        if node.isMap():
            # It's a matrix - read it
            mat = node.mat()
            if mat is not None:
                data[key] = mat
        elif node.isSeq():
            # It's a sequence (like 'names')
            seq_data = []
            for i in range(node.size()):
                item = node.at(i)
                seq_data.append(item.string())
            data[key] = seq_data
        elif node.isString():
            data[key] = node.string()
        elif node.isInt():
            data[key] = node.real()
        elif node.isReal():
            data[key] = node.real()
    
    fs.release()
    return data

def create_npz(data_dir, subfolders, mask_type, json_file=None, intrinsics_yml_file=None, extrinsics_yml_path=None,
                        file_dim=(1080, 1920), resize_factor=0.333333, output_dir='mayostand.npz', start_frame=0, end_frame=None):
    """
    Create mayostand.npz file from directory structure.
    
    Args:
        data_dir: Path to directory containing N subfolders
        extrinsics_yml_path: Path to YAML file with extrinsics
        output_path: Output npz file path
    """
    data_dir = Path(data_dir)
    
    # Load extrinsics YAML
    if extrinsics_yml_path is not None:
        extrinsics_data = load_opencv_yaml(extrinsics_yml_path)
    # Load intrinsics YAML
    if intrinsics_yml_file is not None:
        intrinsics_data = load_opencv_yaml(intrinsics_yml_file)
    #Load json file
    if json_file is not None:
        with open(json_file, 'r') as f:
            json_data = json.load(f)
    
    all_rgbs = []
    all_depths = []
    all_extrs = []
    all_intrs = []
    all_masks = []
    
    H, W = file_dim

    for folder in subfolders:
        print(f"Processing folder: {folder}")
        folder = Path(folder)
        
        # Get RGB and depth paths
        rgb_dir = data_dir / 'images' / folder 
        depth_dir = data_dir / 'depths' / folder
        mask_dir = data_dir / mask_type /'masks' / folder
        
        # Get sorted list of files
        rgb_files = sorted(rgb_dir.glob('*.jpg'), key=lambda x: int(x.stem))
        depth_files = sorted(depth_dir.glob('*.png'), key=lambda x: int(x.stem))
        mask_files = sorted(mask_dir.glob('*.png'), key=lambda x: int(x.stem))
        filenames = [x.stem for x in rgb_files]
        
        M = len(rgb_files)
        print(f"  Found {M} frames")
        
        # Read RGB images
        rgbs = []
        for rgb_file in rgb_files:
            #print(rgb_file)
            img = Image.open(rgb_file)
            img = img.resize((round(W * resize_factor), round(H * resize_factor)), Image.LANCZOS)
            rgb_array = np.array(img)  # Shape: [H, W, 3]
            rgbs.append(rgb_array)
        
        rgbs = np.stack(rgbs, axis=0)  # Shape: [M, H, W, 3]
        rgbs = np.transpose(rgbs, (0, 3, 1, 2))  # Shape: [M, 3, H, W]
        
        # Read depth maps
        depths = []
        for depth_file in depth_files:
            #print(depth_file)
            depth_img = Image.open(depth_file).convert('I')
            depth_img = depth_img.resize((round(W * resize_factor), round(H * resize_factor)), Image.NEAREST)
            depth_array = np.array(depth_img) / 1000.0  # Shape: [H, W]
            #print(f"Min: {depth_array.min()}, Max: {depth_array.max()}")
            depths.append(depth_array)
        
        depths = np.stack(depths, axis=0)  # Shape: [M, H, W]
        depths = np.expand_dims(depths, axis=1)  # Shape: [M, 1, H, W]
        
        # Read masks
        masks = []
        for mask_file in mask_files:
            #print(mask_file)
            mask_img = Image.open(mask_file).convert('L')  # Convert to grayscale
            mask_img = mask_img.resize((round(W * resize_factor), round(H * resize_factor)), Image.NEAREST)
            mask_array = np.array(mask_img)  # Shape: [H, W]
            # Convert: 0 stays 0, 255 becomes 1
            #print(np.unique(mask_array, return_counts=True))
            mask_array = (mask_array / 255.0).astype(np.float32)
            masks.append(mask_array)
        
        masks = np.stack(masks, axis=0)  # Shape: [M, H, W]
        masks = np.expand_dims(masks, axis=1)  # Shape: [M, 1, H, W]

        # Extract extrinsics and intrinsics from json file
        if json_file is not None:
            cam_data = json_data[str(folder)[:-1]]
            
            Rot = np.array(cam_data["R"]).reshape(3, 3)
            T = np.array(cam_data["t"]).reshape(3, 1)
            extr_3x4 = np.hstack([Rot, T])  # Shape: [3, 4]
            extrs = np.tile(extr_3x4[np.newaxis, :, :], (M, 1, 1))  # Shape: [M, 3, 4]
            
            K = np.array(cam_data["K"]).reshape(3, 3)
            K[:-1, :] = K[:-1, :] * resize_factor
            intrs = np.tile(K[np.newaxis, :, :], (M, 1, 1))  # Shape: [M, 3, 3]
        else:
            # Extract extrinsics for this camera
            rot_key = f'Rot_{folder}'
            t_key = f'T_{folder}'
            
            if rot_key in extrinsics_data and t_key in extrinsics_data:
                Rot = np.array(extrinsics_data[rot_key]).reshape(3, 3)
                T = np.array(extrinsics_data[t_key]).reshape(3, 1)
                # Create 3x4 extrinsics matrix
                extr_3x4 = np.hstack([Rot, T])  # Shape: [3, 4]
                # Repeat for all M frames
                extrs = np.tile(extr_3x4[np.newaxis, :, :], (M, 1, 1))  # Shape: [M, 3, 4]
            else:
                print(f"  Warning: Extrinsics not found")
                extrs = np.zeros((M, 3, 4))
            
            # Extract intrinsics for this camera
            K_key = f'K_{folder}'

            if K_key in intrinsics_data:
                # Extract 3x3 camera matrix
                K = np.array(intrinsics_data[K_key]).reshape(3, 3)
                K[:-1, :] = K[:-1, :] * resize_factor
                # Repeat for all M frames
                intrs = np.tile(K[np.newaxis, :, :], (M, 1, 1))  # Shape: [M, 3, 3]
            else:
                print(f"  Warning: Intrinsics file not found")
                intrs = np.eye(3)[np.newaxis, :, :].repeat(M, axis=0)  # Identity matrix fallback
        
        if end_frame == None:
            end_frame = len(rgbs)
        rgbs = rgbs[start_frame:end_frame, :, :, :] 
        depths = depths[start_frame:end_frame, :, :, :]   
        extrs = extrs[start_frame:end_frame, :, :]  
        intrs = intrs[start_frame:end_frame, :, :]   
        masks = masks[start_frame:end_frame, :, :, :] 
        all_rgbs.append(rgbs)
        all_depths.append(depths)
        all_extrs.append(extrs)
        all_intrs.append(intrs)
        all_masks.append(masks)
    
    # Stack all cameras
    rgbs_final = np.stack(all_rgbs, axis=0)  # Shape: [N, M, 3, H, W]
    depths_final = np.stack(all_depths, axis=0)  # Shape: [N, M, 1, H, W]
    extrs_final = np.stack(all_extrs, axis=0)  # Shape: [N, M, 3, 4]
    intrs_final = np.stack(all_intrs, axis=0)  # Shape: [N, M, 3, 3]
    masks_final = np.stack(all_masks, axis=0)  # Shape: [N, M, 1, H, W]  
    
    print(f"\nFinal shapes:")
    print(f"  rgbs: {rgbs_final.shape}")
    print(f"  depths: {depths_final.shape}")
    print(f"  extrs: {extrs_final.shape}")
    print(f"  intrs: {intrs_final.shape}")
    print(f"  masks: {masks_final.shape}")
    
    # Save to npz
    # np.savez_compressed(f"{output_dir}/{mask_type}.npz",
    #             rgbs=rgbs_final,
    #             depths=depths_final,
    #             extrs=extrs_final,
    #             intrs=intrs_final,
    #             masks=masks_final
    #             )
    for idx, filename in enumerate(filenames):
        np.savez_compressed(f"{output_dir}/{filename}.npz",
                        rgbs=rgbs_final[:,idx:idx+1,:,:,:],
                        depths=depths_final[:,idx:idx+1,:,:,:],
                        extrs=extrs_final[:,idx:idx+1,:,:],
                        intrs=intrs_final[:,idx:idx+1,:,:],
                        masks=masks_final[:,idx:idx+1,:,:,:]
                        )
    
    print(f"\nSaved to {output_dir}")
    return rgbs_final, depths_final, extrs_final, intrs_final, masks_final

def main():

    parser = argparse.ArgumentParser(description='Process data to create npz files with image, depth, extrinsic, intrinsic, and masks')

    parser.add_argument('DATA_DIR', help='Input data folder')
    parser.add_argument('DATA_TYPE',  choices=['pulm', 'halo'], help='pulm or halo')
    parser.add_argument('MASK_NAME', help='Name of mask folder')
    parser.add_argument('--ROOT_DIR', help='Root data directory path', default=".")
    parser.add_argument('--start_frame', '-s', type=int, default=0, help="start frame value")
    parser.add_argument('--end_frame', '-f', type=int, default=None, help="end frame value")

    args = parser.parse_args()

    ROOT_DIR = args.ROOT_DIR #'/mnt/data2/Ankita/perception_evaluation'
    DATA_DIR = f"{ROOT_DIR}/dataset/{args.DATA_DIR}"
    OUTPUT_DIR = f"{DATA_DIR}/{args.MASK_NAME}/frames"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.DATA_TYPE == 'pulm':
        #PULM (00_X) folders
        json_file = f"{ROOT_DIR}/dataset/pulm_calibration/ba_poses_to_world.json"
        extrinsics_file = f"{ROOT_DIR}/dataset/pulm_calibration/extri.yml"
        intrinsics_file = f"{ROOT_DIR}/dataset/pulm_calibration/intri.yml"
        subfolders = ['41908851L', '44664489L', '45902703L', '46517772L']
    elif args.DATA_TYPE == 'halo':
        #HALO (09_X and 10_X) folders
        json_file = f"{ROOT_DIR}/dataset/halo_calibration/ba_poses_to_world.json"
        extrinsics_file = f"{ROOT_DIR}/dataset/halo_calibration/extri.yml"
        intrinsics_file = f"{ROOT_DIR}/dataset/halo_calibration/intri.yml"
        subfolders = ['40034694L','40795974L','46026258L','47661457L']

    
    create_npz(data_dir = DATA_DIR, subfolders = subfolders, mask_type=args.MASK_NAME,
                #json_file = json_file, 
                intrinsics_yml_file=intrinsics_file, extrinsics_yml_path=extrinsics_file, 
                output_dir= OUTPUT_DIR, start_frame=args.start_frame, end_frame=args.end_frame)
    
    # Verify by loading
    files = Path(OUTPUT_DIR).glob('*.npz')
    for file in files:
        print(f"\n{file}")
        data = np.load(file)
        print("Loaded data keys:", list(data.keys()))
        print("rgbs shape:", data['rgbs'].shape)
        print("depths shape:", data['depths'].shape)
        print("extrs shape:", data['extrs'].shape)
        print(data['extrs'][0,0,:,:])
        print("intrs shape:", data['intrs'].shape)
        print(data['intrs'][0,0,:,:])
        print("masks shape:", data['masks'].shape)
        print(np.sum(data['masks'][:,0,:,:,:]))


# Usage example
if __name__ == "__main__":
    main()