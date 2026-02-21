# 6D Equipement Motion Modeling

This codebase contains the pipeline for equipment 6D pose estimation and evaluation for motion modeling in TwinOR.

## Data Folder Structure

The dataset should be organized as follows:

```
ROOT_DIR
|
├── dataset/
|    └── XX_X/
|        ├── images/
|        |     ├── {view_1}
|        |     ├── ...
|        |     └── {view_v}
|        |  
|        ├── depths/
|        |     ├── {view_1}
|        |     ├── ...
|        |     └── {view_v}
|        | 
|        └── {EQUIPMENT}/
|               └── masks/
|                   ├── {view_1}
|                   ├── ...
|                   └── {view_v}
|          
|
└── models_scaled/
    └── {EQUIPMENT}/
        ├── equipment.obj
        ├── equipment.mtl
        └── texture.jpg
```

Where:
- `XX_X` represents folder name for a sequence with scenes of a surgical procedure
- `images` contains RGB images with `v` subfolders of all the different views
- `depths` contains depth information  with `v` subfolders of all the different views
- `{EQUIPMENT}` contains equipment-specific data and will store future generated results
- `masks` contains mask with `v` subfolders of all the different views indicating where the equipement is located in the scene. These masks are generated using Segment Anything Model 2 (SAM2).

## Pipeline Overview

All scripts to execute are in `src` directory.

```bash
cd src
```

### 1. Data Preparation

Create NPZ files for point cloud generation:

```bash
python create_data.py DATA_DIR DATA_TYPE MASK_NAME --ROOT_DIR={root_dir_path}
```

**Parameters:**
- `DATA_DIR`: XX_X folder name
- `DATA_TYPE`: We use two types of OR: `pulm` for `00_X` folder and `halo` for `09_X` and `10_X` folder.
- `MASK_NAME`: Name of the `EQUIPMENT` 
- `--ROOT_DIR`: Root directory path (optional)

An example command line: `python create_data.py 00_0 pulm carm`

These npz files will get stored in `{EQUIPMENT}/frames/XXXXXX.npz`

### 2. Generate Point Cloud Batch Script

Create a bash script for automatic point cloud generation:

```bash
python create_bashscript_genpcd.py EQUIPMENT --NUM_SAMPLES=10 --ROOT_DIR={root_dir_path}
```

This script randomly selects `NUM_SAMPLES` .npz scene files generated across all sequences and generates a bash script `EQUIPMENT.sh` to automatically create point clouds. It also generates an `EQUIPMENT_pcd.txt` file in the `ROOT_DIR` with paths to the .ply pointclouds that will be created.

### 3. Point Cloud Generation

Execute the generated bash script:

```bash
./EQUIPMENT.sh
```

This runs `generate_pcd.py` for each selected scenes to create scene point cloud in `{EQUIPMENT}/poinclouds/XXXXXX.ply`

### 4. Ground Truth Preparation

Generate ground truth 6D pose JSON file and put them in `{EQUIPMENT}/gt_poses/XXXXXX.json`

`generate_gt_json.py` is a helper script to generate ground truth data in the correct format for evaluation.

### 5. Pose Prediction

Generate prediction JSON files:

```bash
python pose_registration.py EQUIPMENT_pcd.txt models_scaled/EQUIPMENT/equipment.obj --ROOT_DIR={root_dir_path}
```

Performs pose registration and outputs predictions in JSON format as `{EQUIPMENT}/pred_poses/XXXXXX.json`.

### 6. Evaluation

```bash
python calc_metric.py EQUIPMENT_pcd.txt models_scaled/EQUIPMENT/equipment.obj --ROOT_DIR={root_dir_path}
```

Computes the following metrics:
- Rotation error
- Translation error
- 3D mesh IoU
- 3D bounding box IoU

### 7. Visualization

```bash
python pose_proj_visualization.py EQUIPMENT_pcd.txt models_scaled/EQUIPMENT/equipment.obj --ROOT_DIR={root_dir_path}
```

Creates visualization results for qualitative assessment of pose estimation performance and stores as `{EQUIPMENT}/visuals/XXXXXX_{camera-sl}.png`.

## Workflow Summary

1. Prepare data with `create_data.py`
2. Generate batch script with `create_bashscript_genpcd.py`
3. Execute `./equipment.sh` to create point clouds
4. Generate ground truth with `generate_gt_json.py`
5. Run pose estimation with `pose_registration.py`
6. Evaluate results with `calc_metric.py`
7. Visualize outputs with `pose_proj_visualization.py`