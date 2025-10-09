
import torch
import math
import numpy as np
from typing import NamedTuple


def getProjectionMatrixShift(znear, zfar,  cx, cy, width, height, fovX, fovY):
    
    focal_x, focal_y = width / (2 * math.tan(fovX / 2)), height / (2 * math.tan(fovY / 2))
    
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    # the origin at center of image plane
    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    # shift the frame window due to the non-zero principle point offsets
    offset_x = cx - (width/2)
    offset_x = (offset_x/focal_x)*znear
    offset_y = cy - (height/2)
    offset_y = (offset_y/focal_y)*znear

    top = top + offset_y
    left = left + offset_x
    right = right + offset_x
    bottom = bottom + offset_y

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P
