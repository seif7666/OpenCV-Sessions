import cv2
import numpy as np


fld = cv2.ximgproc.createFastLineDetector()
"""
The array is colored
"""
def get_otsu_threshold(img:np.ndarray):
    gray_img= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return cv2.threshold(gray_img,100,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

def get_kmeans_on_image(img:np.ndarray):
    datapoints= img.reshape((-1,3)).astype(np.float32)
    # print(datapoints.shape)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,labels,centers= cv2.kmeans(datapoints,3, None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    labels=labels.reshape((img.shape[0],img.shape[1]))
    # print(labels.shape)
    images= []
    for i in range(3):
        output= img.copy()
        # print(output.shape)
        output[labels!=i]=np.uint8([0,0,0])
        images.append(output)
    return images


def detect_lines(img:np.ndarray):
    gray_img= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray_img= cv2.threshold(gray_img,30,255,cv2.THRESH_BINARY)[1]
    blurred= cv2.GaussianBlur(gray_img,(5,3),33)
    lines= fld.detect(blurred, 150)
    output_img= img.copy()
    if lines is None:return
    for line in lines:
        line= line[0].astype('int')
        cv2.line(output_img,(line[0],line[1]),(line[2],line[3]),(255,0,0),5)
    cv2.imshow('Line Detector', output_img)



def draw_contours(img:np.ndarray):
    processed= cv2.medianBlur(img,21)
    processed= get_otsu_threshold(img)
    processed= cv2.Canny(img,30,130)
    cv2.imshow('Canny',processed)
    contours, hierarchy = cv2.findContours(processed,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output_img= np.zeros_like(img,dtype=np.uint8)
    cv2.drawContours(output_img, contours, -1, (0, 255, 0), 3)
    
    cv2.imshow('Contours', output_img)




cap= cv2.VideoCapture(0)

while True:
    frame= cap.read()[1]

    cv2.imshow('Original',frame)
    detect_lines(frame)
    draw_contours(frame)
    if cv2.waitKey(1) == 27:
        break
cv2.destroyAllWindows()
cap.release()
