import numpy as np
import cv2
from .feature_extraction import match_features, compute_3d_points

def estimate_pose(prev_kp, prev_des, prev_depth, kp, des, K, depth_scale):
    # Match features
    matches = match_features(prev_des, des)
    if len(matches) < 6:
        return False, None

    # Compute 3D points from previous keypoints
    pts_3d = compute_3d_points(prev_kp, prev_depth, K, depth_scale)

    # Filter matches where 3D point exists
    valid_matches = [m for m in matches if pts_3d[m.queryIdx] is not None]
    if len(valid_matches) < 6:
        return False, None

    # Construct arrays of 3D points and 2D points
    pts_3d_array = np.array([pts_3d[m.queryIdx] for m in valid_matches], dtype=np.float32)
    pts_2d_array = np.array([kp[m.trainIdx].pt for m in valid_matches], dtype=np.float32)

    # Solve PnP with RANSAC
    _, rvec, tvec, _ = cv2.solvePnPRansac(
        pts_3d_array, pts_2d_array, K, None, flags=cv2.SOLVEPNP_ITERATIVE
    )
    # Restrict translation in Z direction
    tvec[2] = 0

    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R

    T[:3, 3] = tvec.flatten()

    return True, T