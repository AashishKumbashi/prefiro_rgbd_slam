import os
import yaml
import numpy as np
import re

def load_intrinsics(config_path):
    with open(config_path, 'r') as f:
        intrinsics = yaml.safe_load(f)

    K = np.array([
        [intrinsics['fx'], 0, intrinsics['cx']],
        [0, intrinsics['fy'], intrinsics['cy']],
        [0, 0, 1]
    ], dtype=np.float64)
    width, height = intrinsics['width'], intrinsics['height']
    dist_coeffs = np.array(intrinsics['distortion'], dtype=np.float64)
    depth_scale = intrinsics.get('depth_scale', 1000.0)
    return K, dist_coeffs, depth_scale, width, height

def get_sort_keys(filename):
    match = re.search(r'(\d{8})_(\d{6})_(\d+)', filename)
    if match:
        middle = int(match.group(2))
        last = int(match.group(3))
        return (middle, last)
    return (0,0)

def load_images(rgb_dir, depth_dir):
    rgb_files = [os.path.join(rgb_dir, f) for f in os.listdir(rgb_dir) if f.endswith(".png") or f.endswith(".jpg")]
    depth_files = [os.path.join(depth_dir, f) for f in os.listdir(depth_dir) if f.endswith(".png")]
    rgb_files.sort(key=lambda x: get_sort_keys(os.path.basename(x)))
    depth_files.sort(key=lambda x: get_sort_keys(os.path.basename(x)))
    assert len(rgb_files) == len(depth_files), "RGB and Depth counts do not match!"
    return list(zip(rgb_files, depth_files))