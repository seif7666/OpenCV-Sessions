import cv2
import numpy as np
from library import *

"""
Set all pixel values that are smaller than threshold to 0 and otherwise to 255
"""
def threshold_to_zero(img:np.ndarray, threshold:int):
    img[img > threshold]= 255
    img[img<=threshold]= 0
    return
"""
Same as above but returns different copy of image
"""
def threshold_to_zero_copy(img:np.ndarray, threshold:int):
    img= img.copy()
    img[img > threshold]= 255
    img[img<=threshold]= 0
    cv2.imshow(f'threshold{threshold}', img)
    return img

def sharpen_image(img):
    kernel= (1/9)*np.array([
        [-1,-2,-1],
        [-2,25,-2],
        [-1,-2,-1]
    ])
    sharpen= cv2.filter2D(img,-1,kernel)
    cv2.imshow('SharpenedImage',sharpen)
    return sharpen

def make4Images(images:list, final_size:tuple=(640,480)):
    for i in range(len(images)):
        cv2.imshow(f'ImagePart{i+1}',images[i])
        images[i]= cv2.resize(images[i], (final_size[0]//2,final_size[1]//2))
    upper_image= np.hstack((images[0],images[1]))
    bottom_image= np.hstack((images[2],images[3]))
    whole_image= np.vstack((upper_image,bottom_image))
    cv2.imshow('Whole Image', whole_image)

def detect_edges(img:np.ndarray):
    lower,upper=get_lower_upper_values(img)
    mask= cv2.Canny(img,lower,upper)
    cv2.imshow('Mask',mask)
    return mask
    


if __name__ =='__main__':
    img1= read_image('pictures/mercedes.jpeg', (640,360))
    img2= read_image('pictures/bmw.jpeg', (640,360))
    img3= read_image('pictures/audi.jpeg', (480,360))
    img4= read_image('pictures/volks.jpeg', (360,360))
    
    cv2.imshow('image',img1)
    threshold_to_zero_copy(img1,50)
    make4Images([img1,img2,img3,img4], (800,500))
    sharped_detect= detect_edges(sharpen_image(img1))
    detect_edges(img1)
    cv2.imshow('Sharpened Mask',sharped_detect)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


