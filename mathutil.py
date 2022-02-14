import numpy as np


def Rx(phi):
    return np.array([[1,            0,           0],
                     [0,  np.cos(phi), np.sin(phi)],
                     [0, -np.sin(phi), np.cos(phi)]])


def Ry(phi):
    return np.array([[ np.cos(phi), 0, np.sin(phi)],
                     [           0, 1,           0],
                     [-np.sin(phi), 0, np.cos(phi)]])


def Rz(phi):
    return np.array([[ np.cos(phi), np.sin(phi), 0],
                     [-np.sin(phi), np.cos(phi), 0],
                     [           0,           0, 1]])
