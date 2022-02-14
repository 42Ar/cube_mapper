#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from mathutil import Rx, Rz
import numpy as np


# a simple pinhole camera model is used at the moment
in_size = (1080, 1920)
fov = np.array([1, in_size[0]/in_size[1]])  # half size of the field of view at distance of 1
out_size = 1500
camsys = "AMS131"
active_cams = [1, 2, 5]


# all cameras "look up" (along +Z) when R = I
camdata = {
    "AMS131/1": {
        "R": Rx(np.pi/2)
    },
    "AMS131/2": {
        "R": np.array([[ 0.21832521, -0.51721712,  0.82753885],
                       [-0.8463925 ,  0.32173762,  0.42438736],
                       [-0.48575079, -0.79307714, -0.36752525]])
    },
    "AMS131/5": {
        "R": np.array([[ 0.15831412,  0.29959825, -0.94083874],
                       [ 0.96277469,  0.16459091,  0.21441716],
                       [ 0.21909251, -0.93976099, -0.26238855]])
    }
}


#
#     +--+
#     |+Y|
#  +--+--+--+--+
#  |-X|+Z|+X|-Z|
#  +--+--+--+--+
#     |-Y|
#     +--+
#
faces = {
    "+Z": np.array([[ 1,  0,  0],
                    [ 0,  1,  0],
                    [ 0,  0,  1]]),
    "-Z": np.array([[-1,  0,  0],
                    [ 0,  1,  0],
                    [ 0,  0, -1]]),
    "+X": np.array([[ 0,  0,  1],
                    [ 0,  1,  0],
                    [-1,  0,  0]]),
    "-X": np.array([[ 0,  0, -1],
                    [ 0,  1,  0],
                    [ 1,  0,  0]]),
    "+Y": np.array([[ 1,  0,  0],
                    [ 0,  0,  1],
                    [ 0, -1,  0]]),
    "-Y": np.array([[ 1,  0,  0],
                    [ 0,  0, -1],
                    [ 0,  1,  0]])
}
