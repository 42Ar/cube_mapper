import numpy as np
import cv2
from imgutil import read_img
from scipy.optimize import minimize
from mathutil import Rx, Ry, Rz


in_size = (1080//2, 1920//2)
fov_factor = 1
marks = [
    (1, 2, [((925, 1080//2 - 338), (1131 - 1920//2, 1080//2 - 383)),
            ((946, 1080//2 - 321), (1156 - 1920//2, 1080//2 - 375)),
            ((834, 1080//2 - 390), (1036 - 1920//2, 1080//2 - 398)),
            ((952, 1080//2 - 372), (1154 - 1920//2, 1080//2 - 428)),
            ((808, 1080//2 - 367), (1014 - 1920//2, 1080//2 - 364))]),
    (5, 1, [((891, 1080//2 - 450), (1007 - 1920//2, 1080//2 - 407)),
            ((950, 1080//2 - 331), (1075 - 1920//2, 1080//2 - 313)),
            ((842, 1080//2 - 380), (970 - 1920//2, 1080//2 - 316))])
]
camid_to_param_offset = {2: 0, 5: 3}


def to_vec(px, py):
    fov = fov_factor*np.array([1, in_size[0]/in_size[1]])
    x = fov[0]*(2*px/in_size[1] - 1)
    y = fov[1]*(2*py/in_size[0] - 1)
    vec = np.array([x, y, 1])
    return vec/np.linalg.norm(vec)


def gen_matrix(camid, params):
    if camid == 1:
        return Rx(np.pi/2)
    else:
        off = camid_to_param_offset[camid]
        return np.matmul(Rx(params[off + 2]), np.matmul(Ry(params[off + 1]), Rz(params[off])))


def calc_chi2(params):
    chi2 = 0
    for a, b, marker in marks:
        Ma = gen_matrix(a, params)
        Mb = gen_matrix(b, params)
        for m1, m2 in marker:
            diff = np.matmul(Ma, to_vec(*m1)) - np.matmul(Mb, to_vec(*m2))
            chi2 += np.sum(diff**2)
    return chi2


def show_markers():
    for a, b, marker in marks:
        a = read_img(a, in_size)
        b = read_img(b, in_size)
        for m1, m2 in marker:
            cv2.circle(a, m1, 5, (1, 0, 0))
            cv2.circle(b, m2, 5, (1, 0, 0))
        cv2.imshow("f", np.flip(np.hstack([a, b]), axis=0))
        while cv2.waitKey(0) != ord("q"):
            pass
        cv2.destroyAllWindows()


show_markers()
res = minimize(calc_chi2, x0=[0]*3*len(marks))
print(repr(gen_matrix(2, res.x)))
print(repr(gen_matrix(5, res.x)))

