# -*- coding: utf-8 -*-

from glob import glob
from config import in_size
import cv2


def read_img(camid, in_size=in_size):
    files = glob(f"test_data/*{camid}.mp4")
    if len(files) == 0:
        print("file not found")
        return None
    vidcap = cv2.VideoCapture(files[0])    
    success, image = vidcap.read()
    if not success:
        print(f"failed to read camera image {files[0]}")
        return None
    if image.shape[:2] != in_size:
        image = cv2.resize(image, (in_size[1], in_size[0]))
    return cv2.flip(image, 0)