"""
This file is used to implement secondary functions.
"""
import numpy as np
import cv2

SIGMA=0.33

def read_image(image_path,size:tuple=(480,480)):
    img=cv2.imread(image_path,0)
    return cv2.resize(img,size)

def get_lower_upper_values(img:np.ndarray):
    v = np.median(img)
    lower = int(max(0, (1.0 - SIGMA) * v))
    upper = int(min(255, (1.0 + SIGMA) * v))
    return lower,upper