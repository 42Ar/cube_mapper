import cv2
import numpy as np
from config import camdata, fov, out_size, in_size, faces, camsys, active_cams
from imgutil import read_img

def calculate_pixels(camconf, facemat):
    c = np.linspace(-1 + 1/out_size, 1 - 1/out_size, out_size)
    x, y = np.meshgrid(c, c)
    z = np.full((out_size, out_size), 1)
    M = np.matmul(np.linalg.inv(camconf["R"]), facemat)
    coords = np.einsum("ij,jkl", M, [x, y, z])
    x = coords[0]/coords[2]
    y = coords[1]/coords[2]
    px = np.round(in_size[1]*(x/fov[0] + 1)/2).astype(np.float32)
    py = np.round(in_size[0]*(y/fov[1] + 1)/2).astype(np.float32)
    py[coords[2] < 0] = 1e8  # make sure that those are out of range
    px[coords[2] < 0] = 1e8
    return px, py


precalc_rays = {}
for name, facemat in faces.items():
    cams = {}
    for camid in active_cams:
        px, py = calculate_pixels(camdata[f"{camsys}/{camid}"], facemat)
        if np.sum(np.all([px >= 0, py >= 0,
                          px < in_size[1], py < in_size[0]], axis=0)) == 0:
            continue
        cams[camid] = (px, py)
    precalc_rays[name] = cams

#%%

images = {}
for camid in active_cams:
    images[camid] = read_img(camid)
if len(images) == 0:
    print("no cameras available")

#%%

first_image = next(iter(images.values()))
output = np.zeros((out_size, 4*out_size, 3), dtype=first_image.dtype)
cnt = np.zeros((out_size, 4*out_size), dtype=np.uint8)
dst = np.zeros((out_size, out_size, 3), dtype=first_image.dtype)
faces = ["-X", "+Y", "+X", "-Y"]
for i, cube_side in enumerate(faces):
    for camid, (px, py) in precalc_rays[cube_side].items():
        try:
            if cube_side == "+X":
                px = np.flip(px.T, axis=1)
                py = np.flip(py.T, axis=1)
            elif cube_side == "-Y":
                px = np.flip(px)
                py = np.flip(py)
            if cube_side == "-X":
                px = np.flip(px.T, axis=0)
                py = np.flip(py.T, axis=0)
            cv2.remap(images[camid], px, py, cv2.INTER_LINEAR, dst)
            output[0:out_size, out_size*i:out_size*(i + 1)] += dst
            cnt[0:out_size, out_size*i:out_size*(i + 1)] += np.any(dst > 0, axis=2)
        except KeyError:
            pass # camera not available
            
    px, py = calculate_pixels(camdata["AMS131/1"], np.eye(3))
output //= np.repeat(cnt, 3).reshape(*output.shape)
res_scale = 3
output = cv2.flip(cv2.resize(output, (output.shape[1]//res_scale, output.shape[0]//res_scale)), 0)
for i, face in enumerate(faces):
    cv2.putText(output, face, (20 + out_size*i//res_scale, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                cv2.LINE_AA)
cv2.imshow('frame', output)

cv2.waitKey(0)
cv2.destroyAllWindows()