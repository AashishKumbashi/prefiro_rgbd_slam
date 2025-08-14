import numpy as np

def accumulate(rel_poses):
    globals_ = [np.eye(4)]
    for T in rel_poses:
        globals_.append(globals_[-1] @ T)
    return globals_

def compute_path_length(rel_poses):
    return sum(np.linalg.norm(T[:3, 3]) for T in rel_poses)

def pixel_to_cam(u, v, depth_m, intrinsics):
    x = (u - intrinsics["cx"]) * depth_m / intrinsics["fx"]
    y = (v - intrinsics["cy"]) * depth_m / intrinsics["fy"]
    return np.array([x, y, depth_m], dtype=np.float32)

def make_3d_2d_corresp(kps1, kps2, matches, depth1, intrinsics):
    pts3d, pts2d = [], []
    h, w = depth1.shape
    for m in matches:
        u1, v1 = kps1[m.queryIdx].pt
        u2, v2 = kps2[m.trainIdx].pt
        ui, vi = int(round(u1)), int(round(v1))
        if not (0 <= ui < w and 0 <= vi < h):
            continue
        d = depth1[vi, ui]
        if d == 0:
            continue
        z = d / intrinsics["depth_scale"]
        if not (0.05 < z < 20):
            continue
        pts3d.append(pixel_to_cam(u1, v1, z, intrinsics))
        pts2d.append([u2, v2])
    return np.array(pts3d), np.array(pts2d)
