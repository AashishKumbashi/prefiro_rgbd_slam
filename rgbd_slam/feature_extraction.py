import cv2
import numpy as np
import yaml


def extract_features(img):
    orb = cv2.ORB_create(800, fastThreshold=50)
    kp, des = orb.detectAndCompute(img, None)
    return kp, des

def match_features(des1, des2, ratio=0.75):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn_matches = bf.knnMatch(des1, des2, k=2)
    good_matches = []
    for m,n in knn_matches:
        if m.distance < ratio * n.distance:
            good_matches.append(m)
    # Sort by distance
    good_matches.sort(key=lambda x: x.distance)
    return good_matches

def compute_3d_points(kp, depth, K, depth_scale):
    # Extract pixel coordinates from keypoints
    uv = np.array([p.pt for p in kp], dtype=np.float32)  # shape: (N, 2)
    u = uv[:, 0].astype(np.int32)
    v = uv[:, 1].astype(np.int32)
    
    # Get depth values for those coordinates
    z = depth[v, u].astype(np.float32) / depth_scale

    # Mask invalid depths (z == 0)
    valid_mask = z > 0
    
    # Back-project to 3D
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]

    # Create Nx3 array of 3D points
    points_3d = np.stack((x, y, z), axis=-1)

    # Set invalid depths to NaN (instead of None for NumPy efficiency)
    points_3d[~valid_mask] = np.nan
    return points_3d  # shape: (N, 3)