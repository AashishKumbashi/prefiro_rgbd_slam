import numpy as np
import cv2
from .feature_extraction import extract_and_match
from .trajectory import pixel_to_cam, make_3d_2d_corresp

def estimate_T(gray1, gray2, depth1, intrinsics):
    kps1, kps2, _, _, matches = extract_and_match(gray1, gray2)
    pts3d, pts2d = make_3d_2d_corresp(kps1, kps2, matches, depth1, intrinsics)
    if len(pts3d) < 12:
        return False, np.eye(4)

    K = np.array([[intrinsics["fx"], 0, intrinsics["cx"]],
                  [0, intrinsics["fy"], intrinsics["cy"]],
                  [0, 0, 1]])
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts3d, pts2d, K, None, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return False, np.eye(4)
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    return True, T
