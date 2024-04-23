# Object Detection
Here I have implemented few methods of object tracking possible. The object in the project is Pedestrians/ Humans

# Method 1 - Haar Cascade

1. The cascade file is downloaded from the following repository - https://github.com/anaustinbeing/haar-cascade-files/blob/master/haarcascade_fullbody.xml
2. The results with this haar cascade are not very satisfactory as they are not able to detect the pedestrians from a side angle.
3. Futher Work will be to try and develop a new cascade xml file for training with pedestrian data

# Method 2 - HOG+Linear SVM

1. Open CV provides a in-built HOG Descriptor that can detect pedestrians
2. The files related to this code can be found at hogsvm folder in the codes section where hstest.py is for a bunch of images stored in a folder and hsvideo.py is for a drone video.
3. The results are as follows
  a. The processing is taking long time
  b. The detection is very better when compared to Haar Cascade for still images but is not much different for the video

