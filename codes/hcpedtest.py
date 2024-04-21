import cv2
import numpy as np
import matplotlib.pyplot as plt

ped_cascade = cv2.CascadeClassifier("S:/Projects/pedestrianDetection/HaarCascade/haarcascade_fullbody.xml")

def detect_body(img):
    body_rect = ped_cascade.detectMultiScale(img, 1.3, 2)

    for (x,y,w,h) in body_rect:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 1)

    return img

img = cv2.imread('S:/Projects/pedestrianDetection/HaarCascade/Images/pedtest.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
body = detect_body(img)
cv2.imshow('body', body)

cv2.waitKey(0)
cv2.destroyAllWindows()