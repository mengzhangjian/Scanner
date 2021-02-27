import os
import cv2

cap = cv2.VideoCapture('2.mp4')

count  = 1
while True:
    ret, frame= cap.read()
    cv2.imshow('1', frame)
    cv2.waitKey(1)
    #if count % 10 == 0:
        #cv2.imwrite(str(count) + '.jpg', frame)
    count += 1
