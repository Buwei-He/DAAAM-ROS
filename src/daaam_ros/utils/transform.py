#!/usr/bin/env python3

import numpy as np
from numpy.typing import NDArray
from geometry_msgs.msg import Transform


def transform_to_matrix(transform: Transform) -> NDArray[np.float64]:
    """Convert ROS Transform message to 4x4 transformation matrix."""
    rot = transform.rotation
    trans = transform.translation
    
    R = np.array([
        [1 - 2*rot.y*rot.y - 2*rot.z*rot.z, 2*rot.x*rot.y - 2*rot.z*rot.w, 2*rot.x*rot.z + 2*rot.y*rot.w],
        [2*rot.x*rot.y + 2*rot.z*rot.w, 1 - 2*rot.x*rot.x - 2*rot.z*rot.z, 2*rot.y*rot.z - 2*rot.x*rot.w],
        [2*rot.x*rot.z - 2*rot.y*rot.w, 2*rot.y*rot.z + 2*rot.x*rot.w, 1 - 2*rot.x*rot.x - 2*rot.y*rot.y]
    ])
    
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [trans.x, trans.y, trans.z]
    return T