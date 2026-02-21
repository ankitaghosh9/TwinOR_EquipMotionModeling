from pathlib import Path
import random
import argparse

def get_npz_paths(base_path, foldername, num_samples=10):
    """
    Get random .npz file paths matching the pattern using pathlib.
    
    Args:
        base_path: Base directory path
        foldername: The fixed folder name to search for
        num_samples: Number of random paths to extract
    
    Returns:
        List of randomly selected path strings
    """
    base = Path(base_path)
    
    # Use glob with the pattern
    npz_paths = list(base.glob(f"*/{foldername}/frames/*.npz"))
    
    # Convert to strings
    npz_paths_str = [str(path) for path in npz_paths]
    
    # Use random.sample to get random paths without replacement
    # If there are fewer paths than num_samples, return all of them
    num_to_sample = min(num_samples, len(npz_paths_str))
    random_paths = random.sample(npz_paths_str, num_to_sample)
    
    return random_paths

def create_bash_script(random_paths, base_path, output_script='run_demo.sh', output_txt='pointcloud_paths.txt'):
    """
    Create a bash script that runs demo.py for each path and save pointcloud paths to txt.
    
    Args:
        random_paths: List of npz file paths
        output_script: Name of the output bash script file
        output_txt: Name of the output text file for pointcloud paths
    """
    ply_paths = []
    
    with open(output_script, 'w') as f:
        # Write shebang
        f.write('#!/bin/bash\n\n')
        
        # Write commands for each path
        for npz_path in random_paths:
            # Convert npz_path to ply_path
            ply_path = npz_path.replace('/frames/', '/pointclouds/').replace('.npz', '.ply')
            ply_paths.append(ply_path)
            
            # Create the directory for ply_path if it doesn't exist
            ply_dir = str(Path(ply_path).parent)
            f.write(f'mkdir -p {ply_dir}\n')
            
            # Write the command
            cmd = (f'python generate_pcd.py --frame_path {npz_path} --save_pcd_path {ply_path}\n')
            f.write(cmd)
            f.write('\n')
    
    # Make the script executable
    Path(output_script).chmod(0o755)
    print(f"Bash script saved to: {output_script}")
    
    # Save pointcloud paths to text file
    with open(output_txt, 'w') as f:
        for ply_path in ply_paths:
            ply_path = ply_path.replace(base_path, "")
            f.write(f"{ply_path}\n")
    
    print(f"Pointcloud paths saved to: {output_txt}")
    
    return ply_paths


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process data to create npz files with image, depth, extrinsic, intrinsic, and masks')

    parser.add_argument('EQUIPMENT', help='Input data folder')
    parser.add_argument('--NUM_SAMPLES', help='Number of scenes to sample', default=10)
    parser.add_argument('--ROOT_DIR', help='Root data directory path', default=".")

    args = parser.parse_args()

    # Example usage:
    base_path = args.ROOT_DIR  #'/mnt/data2/Ankita/perception_evaluation/'
    foldername = args.EQUIPMENT  #'carm'
    num_samples = args.NUM_SAMPLES

    random_paths = get_npz_paths(f"{base_path}/dataset", foldername, num_samples=args.NUM_SAMPLES)

    print(f"Selected {len(random_paths)} random files:")
    for i, path in enumerate(random_paths, 1):
        print(f"{i}. {path}")

    # Create the bash script and save pointcloud paths
    ply_paths = create_bash_script(random_paths, base_path, 
                        output_script=f'{foldername}.sh', 
                        output_txt=f'{base_path}/{foldername}_pcd.txt')

    print(f"\nGenerated {len(ply_paths)} pointcloud paths")