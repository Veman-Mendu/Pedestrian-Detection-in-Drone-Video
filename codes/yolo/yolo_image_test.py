import cv2
from ultralytics import YOLO
import numpy as np
from imutils.object_detection import non_max_suppression

model = YOLO("yolov9c.pt")

def predict(chosen_model, img, classes=[], conf=0.5):
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf)
    else:
        results = chosen_model.predict(img, conf=conf)
    
    return results

def predict_and_detect(chosen_model, img, classes=[], conf=0.5, rectangle_thickness=2, text_thickness=1):
    results = predict(chosen_model, img, classes, conf)

    for result in results:
        new_boxes = []
        for box in result.boxes:
            x = int(box.xyxy[0][0])
            y = int(box.xyxy[0][1])
            w = int(box.xyxy[0][2])
            h = int(box.xyxy[0][3])
            new_boxes.append((x,y,w,h))
        new_boxes = np.array(new_boxes)

        pick = non_max_suppression(new_boxes, probs=None, overlapThresh=0.65)

        for (xA,yA,xB,yB) in pick:
            cv2.rectangle(img, (xA,yA), (xB,yB), (0,255,0), 2)
    return img, results

image = cv2.imread("S:/Projects/pedestrianDetection/HaarCascade/Images/pedtest1.jpg")
result_img, _ = predict_and_detect(model, image, classes=[], conf=0.5)

cv2.imshow("Image", result_img)
cv2.waitKey(0)