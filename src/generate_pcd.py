import argparse
import os
import numpy as np
import torch
import open3d as o3d

def save_pcd(pts, path, pts_color=None):
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.clone().detach().cpu().numpy())
    if pts_color is not None:
        pcd.colors = o3d.utility.Vector3dVector(pts_color.clone().detach().cpu().numpy())
    o3d.io.write_point_cloud(path, pcd)

def remove_noise_open3d(points, method='global statistical'):
    """
    points: [N, 3] torch tensor or numpy array
    method: 'statistical' or 'radius'
    """
    # Convert to Open3D format
    num_points = points.shape[0]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.cpu().numpy())
    
    if method == 'global statistical':
        # Statistical outlier removal
        cleaned_pcd, inlier_idx = pcd.remove_statistical_outlier(
            nb_neighbors=num_points//2,
            std_ratio=2.0
        )
    elif method == 'local statistical':
        # Statistical outlier removal
        cleaned_pcd, inlier_idx = pcd.remove_statistical_outlier(
            nb_neighbors=20,
            std_ratio=1.5
        )
    elif method == 'global radius':
        # Radius outlier removal
        cleaned_pcd, inlier_idx = pcd.remove_radius_outlier(
            nb_points=num_points//2,
            radius=2.5
        )
    elif method == 'local radius':
        # Radius outlier removal
        cleaned_pcd, inlier_idx = pcd.remove_radius_outlier(
            nb_points=5,
            radius=0.05
        )
    
    # Convert back to torch
    cleaned_points = torch.tensor(np.asarray(cleaned_pcd.points))
    
    return cleaned_points.float(), inlier_idx

def to_homogeneous(x):
    return torch.cat([x, x.new_ones(x[..., :1].shape)], -1)


def from_homogeneous(x, assert_homogeneous_part_is_equal_to_1=False, eps=0.1):
    if assert_homogeneous_part_is_equal_to_1:
        assert torch.allclose(x[..., -1], x.new_ones(x[..., -1].shape), atol=eps)
    return x[..., :-1] / x[..., -1:]

def init_pointcloud_from_rgbd(
        fmaps: torch.Tensor,
        depths: torch.Tensor,
        intrs: torch.Tensor,
        extrs: torch.Tensor,
        depth_interp_mode='nearest',
        return_validity_mask=False,
        return_color_value=False,
):
    B, V, S, C, H, W = fmaps.shape
    assert fmaps.shape == (B, V, S, C, H, W)
    assert depths.shape == (B, V, S, 1, H, W)
    assert intrs.shape == (B, V, S, 3, 3)
    assert extrs.shape == (B, V, S, 3, 4)

    # Invert intrinsics and extrinsics
    intrs_inv = torch.inverse(intrs.float()).type(intrs.dtype)
    extrs_square = torch.eye(4).to(extrs.device)[None].repeat(B, V, S, 1, 1)
    extrs_square[:, :, :, :3, :] = extrs
    extrs_inv = torch.inverse(extrs_square.float()).type(extrs.dtype)
    assert intrs_inv.shape == (B, V, S, 3, 3)
    assert extrs_inv.shape == (B, V, S, 4, 4)

    # Pixel --> Camera --> World
    pixel_xy = torch.stack(torch.meshgrid(
        torch.arange(0, H),
        torch.arange(0, W),
        indexing="ij",
    )[::-1], dim=-1)
    pixel_xy = pixel_xy.to(device=fmaps.device, dtype=fmaps.dtype)
    pixel_xy_homo = to_homogeneous(pixel_xy)
    depthmap_camera_xyz = torch.einsum('BVSij,HWj->BVSHWi', intrs_inv, pixel_xy_homo)
    depthmap_camera_xyz = depthmap_camera_xyz * depths[..., 0, :, :, None]
    depthmap_camera_xyz_homo = to_homogeneous(depthmap_camera_xyz)
    depthmap_world_xyz_homo = torch.einsum('BVSij,BVSHWj->BVSHWi', extrs_inv, depthmap_camera_xyz_homo)
    depthmap_world_xyz = from_homogeneous(depthmap_world_xyz_homo)

    pointcloud_xyz = depthmap_world_xyz.permute(0, 2, 1, 3, 4, 5).reshape(B * S, V * H * W, 3)
    pointcloud_fvec = fmaps.permute(0, 2, 1, 4, 5, 3).reshape(B * S, V * H * W, C)

    return_params = (pointcloud_xyz, pointcloud_fvec)

    if return_validity_mask:
        pointcloud_valid_mask = depths.permute(0, 2, 1, 3, 4, 5).reshape(B * S, V * H * W) > 0
        return_params = (*return_params, pointcloud_valid_mask)

    if return_color_value:
        pointcloud_color = fmaps.permute(0, 2, 1, 4, 5, 3).reshape(B * S, V * H * W, 3)
        return_params = (*return_params, pointcloud_color)
    
    return return_params

def main():

    p = argparse.ArgumentParser()
    
    p.add_argument("--frame_path", default="data_sample.npz", help="Path to loading npz data")
    p.add_argument("--save_pcd_path", default=None, help="Path for saving .ply pointcloud of the object")
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sample = np.load(args.frame_path)
    rgbs = torch.from_numpy(sample["rgbs"]).float()
    depths = torch.from_numpy(sample["depths"]).float()
    intrs = torch.from_numpy(sample["intrs"]).float()
    extrs = torch.from_numpy(sample["extrs"]).float()
    masks = torch.from_numpy(sample["masks"]).float()

    sample_depths = depths * masks
    xyz, _, xyz_mask, xyz_color = init_pointcloud_from_rgbd(
        fmaps=rgbs[None],  # [1,V,T,1,H,W], uint8 0–255
        depths=sample_depths[None],  # [1,V,T,1,H,W]
        intrs=intrs[None],  # [1,V,T,3,3]
        extrs=extrs[None],  # [1,V,T,3,4]
        return_validity_mask=True,
        return_color_value=True
    )
    pts = xyz.clone()
    pts_color = xyz_color.clone()
    assert pts.numel() > 0, "No valid depth points to sample queries from."

    query_pool = []
    pool = pts[0][xyz_mask[0]]
    pool_color = pts_color[0][xyz_mask[0]]
    
    ####### Post processing steps #########
    # print("color range", torch.max(pool_color), torch.min(pool_color))
    # pool, kept_idx = remove_noise_open3d(pool, method="global radius")
    # pool_color = pool_color[kept_idx]
    pool, kept_idx = remove_noise_open3d(pool, method="local statistical")
    pool_color = pool_color[kept_idx]
    # pool, kept_idx = remove_noise_open3d(pool, method="local radius")
    # pool_color = pool_color[kept_idx]
    print("obj shape:", pool.shape, pool_color.shape)
    ######################################

    if args.save_pcd_path is not None:
        save_pcd(pool, args.save_pcd_path, pool_color/255.0)


if __name__ == "__main__":
    main()