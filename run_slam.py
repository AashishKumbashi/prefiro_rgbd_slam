import os
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import g2o
import time
from rgbd_slam.dataloder import load_intrinsics, load_images
from rgbd_slam.feature_extraction import extract_features
from rgbd_slam.pose_estimation import estimate_pose
from rgbd_slam.trajectory import compute_total_distance
from rgbd_slam.optimizer import create_vertex, create_edge

def main(args):
    K, dist_coeffs, depth_scale, width, height = load_intrinsics(args.config)
    pairs = load_images(args.rgb_dir, args.depth_dir)

    # Setup g2o optimizer
    solver = g2o.BlockSolverX(g2o.LinearSolverEigenX())
    algo = g2o.OptimizationAlgorithmLevenberg(solver)
    optimizer = g2o.SparseOptimizer()
    optimizer.set_algorithm(algo)

    traj = []

    # Add first vertex (fixed)
    create_vertex(optimizer, 0, np.eye(4), fixed=True)
    traj.append(np.zeros(3))

    prev_kp = prev_des = prev_depth = None

    # Incremental plotting
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title("RGB-D SLAM Incremental Trajectory")
    frame_times = []

    for i, (rgb_path, depth_path) in enumerate(pairs):
        start_time = time.time()

        rgb = cv2.imread(rgb_path)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        map1, map2 = cv2.initUndistortRectifyMap(K, dist_coeffs, None, K, (width, height), cv2.CV_16SC2)
        rgb_undistorted = cv2.remap(rgb, map1, map2, cv2.INTER_LINEAR)

        kp, des = extract_features(rgb_undistorted)
        cv2.imshow('Keypoints', cv2.drawKeypoints(rgb_undistorted, kp, None,
                                                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS))
        cv2.waitKey(1)

        if prev_kp is not None:
            success, T = estimate_pose(prev_kp, prev_des, prev_depth, kp, des, K, depth_scale)
            if success:
                create_vertex(optimizer, i, np.eye(4))
                create_edge(optimizer, i-1, i, T)
                # assumes T[:3,3] in world frame
                traj.append(traj[-1] + T[:3,3])  
            else:
                traj.append(traj[-1])
        else:
            traj.append(traj[-1])

        prev_kp, prev_des, prev_depth = kp, des, depth

        traj_np = np.array(traj)
        ax.clear()
        ax.plot(traj_np[:,0], traj_np[:,1], traj_np[:,2], 'b-')
        ax.scatter(traj_np[-1,0], traj_np[-1,1], traj_np[-1,2], c='r', s=30)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_title("RGB-D SLAM Incremental Trajectory")
        plt.draw(); plt.pause(0.001)

        frame_times.append(time.time() - start_time)

    plt.ioff()
    total_distance = compute_total_distance(traj)
    print(f"Total incremental distance: {total_distance:.3f} m")
    print(f"Total processing time: {sum(frame_times):.3f} s")
    plt.savefig(os.path.join(args.output_dir, 'trajectory.png'))
    # No loop closures
    # Back end scrip to be added to optimize the trajectory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RGB-D SLAM with g2o")
    parser.add_argument('--rgb_dir', required=True)
    parser.add_argument('--depth_dir', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output_dir', default='outputs')
    args = parser.parse_args()
    main(args)
