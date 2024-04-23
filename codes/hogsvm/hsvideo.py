import cv2
from imutils.object_detection import non_max_suppression
import numpy as np

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cap = cv2.VideoCapture("S:/Projects/pedestrianDetection/HaarCascade/Images/dronevideo.mp4")

while True:
    ret, img = cap.read()

    if type(img) == type(None):
        break

    (rects, weights) = hog.detectMultiScale(img, winStride=(4,4), padding=(8,8), scale=1.05)

    rects = np.array([[x,y,x+w,y+h] for (x,y,w,h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    for (xA,yA,xB,yB) in pick:
        cv2.rectangle(img, (xA,yA), (xB,yB), (0,255,0), 2)

    cv2.imshow("video", img)

    if cv2.waitKey(33) == 27:
        break

cv2.destroyAllWindows()