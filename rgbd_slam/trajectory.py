import numpy as np

def compute_total_distance(traj):
    distance = 0
    for i in range(1, len(traj)):
        distance += np.linalg.norm(traj[i] - traj[i-1])
    return distance

