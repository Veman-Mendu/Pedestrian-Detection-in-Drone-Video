import cv2

cap = cv2.VideoCapture("S:/Projects/pedestrianDetection/HaarCascade/Images/dronevideo.mp4")

pedCascade = cv2.CascadeClassifier("S:/Projects/pedestrianDetection/HaarCascade/haarcascade_fullbody.xml")

while True:
    ret, img = cap.read()

    if type(img) == type(None):
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ped = pedCascade.detectMultiScale(gray)

    for (x,y,w,h) in ped:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)

    cv2.imshow('video', img)

    if cv2.waitKey(33) == 27:
        break

cv2.destroyAllWindows()