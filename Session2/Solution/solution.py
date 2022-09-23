import cv2
import numpy as np

def segment():
    

    def nothing(x):
            pass

    cv2.namedWindow('Bars')
    cv2.createTrackbar('HMin', 'Bars', 0, 255, nothing)
    cv2.createTrackbar('HMax', 'Bars', 0, 255, nothing)
    cv2.createTrackbar('SMin', 'Bars', 100, 255, nothing)
    cv2.createTrackbar('SMax', 'Bars', 255, 255, nothing)
    cv2.createTrackbar('VMin', 'Bars', 100, 255, nothing)
    cv2.createTrackbar('VMax', 'Bars', 255, 255, nothing)
    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()      

        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([(cv2.getTrackbarPos('HMin', 'Bars'), cv2.getTrackbarPos('SMin', 'Bars'), cv2.getTrackbarPos('VMin', 'Bars'))])
        upper = np.array([(cv2.getTrackbarPos('HMax', 'Bars'), cv2.getTrackbarPos('SMax', 'Bars'), cv2.getTrackbarPos('VMax', 'Bars'))])
        mask = cv2.inRange(frame_hsv, lower, upper)
        result = cv2.bitwise_and(frame, frame, mask=mask)
        k = cv2.waitKey(5)
        if k == 27:
            break
        cv2.imshow('HSV', result)
        cv2.imshow('Original',frame)
    cap.release()
    cv2.destroyAllWindows()


segment()