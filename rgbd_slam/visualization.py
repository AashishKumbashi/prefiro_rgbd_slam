import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os

def plot_trajectory(global_poses, save_path):
    traj = np.array([T[:3, 3] for T in global_poses])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(traj[:,0], traj[:,1], traj[:,2], linewidth=2)
    ax.scatter(traj[0,0], traj[0,1], traj[0,2], c='g', marker='o', label='start')
    ax.scatter(traj[-1,0], traj[-1,1], traj[-1,2], c='r', marker='^', label='end')
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend()
    max_range = np.ptp(traj, axis=0).max()
    mid = np.mean(traj, axis=0)
    for setter, m in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], mid):
        setter(m - max_range/2, m + max_range/2)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
