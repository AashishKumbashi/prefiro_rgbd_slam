# rgbd_slam/optimizer.py
import g2o
import numpy as np

def create_vertex(optimizer, id, pose=np.eye(4), fixed=False):
    """
    Adds a VertexSE3 to the optimizer.
    :param optimizer: g2o optimizer
    :param id: vertex ID
    :param pose: 4x4 transformation matrix
    :param fixed: whether to fix this vertex
    """
    v = g2o.VertexSE3()
    v.set_id(id)
    se3 = g2o.SE3Quat(pose[:3,:3], pose[:3,3])
    v.set_estimate(g2o.Isometry3d(se3.matrix()))
    v.set_fixed(fixed)
    optimizer.add_vertex(v)
    return v

def create_edge(optimizer, id_from, id_to, transform, information=None):
    """
    Adds an EdgeSE3 between two vertices.
    :param optimizer: g2o optimizer
    :param id_from: source vertex ID
    :param id_to: target vertex ID
    :param transform: 4x4 relative pose from source to target
    :param information: 6x6 information matrix
    """
    e = g2o.EdgeSE3()
    e.set_vertex(0, optimizer.vertex(id_from))
    e.set_vertex(1, optimizer.vertex(id_to))
    se3 = g2o.SE3Quat(transform[:3,:3], transform[:3,3])
    e.set_measurement(g2o.Isometry3d(se3.matrix()))

    if information is None:
        information = np.eye(6)
        # balanced rotation and translation confidence
        information[:3,:3] *= 1.0  # rotation
        information[3:,3:] *= 1.0  # translation
    e.set_information(information)
    optimizer.add_edge(e)
    return e
