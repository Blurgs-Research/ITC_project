
import torch
import cv2
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from playsound import playsound

#######

model=torch.hub.load('ultralytics/yolov5','yolov5s') # downloading the model 
model.classes=[0] #in coco data set car index is in 2
model.conf=0.20

###
model=torch.hub.load('ultralytics/yolov5','yolov5s') # downloading the model 
model.classes=[0] #in coco data set car index is in 2

# now we will use opecv for real time 

import cv2
# define a video capture object
vid = cv2.VideoCapture(0)
# os.mkdir("video_to_image1") 
# img_count = 1
while vid.isOpened():      
    ret, frame = vid.read()
    results=model(frame,size=320) 

    string_convert=(str(results))
    class_names=string_convert[20:27]
    print(class_names) # pring the class name, 
    # Display the resulting frame
    a=np.squeeze(results.render())
   
    # if the place is detected then paly the sound
    if class_names==" person":
        playsound('/home/blurgs/Downloads/human.mp3')
    else:
        pass
    cv2.imshow("video",a)

    cv2.waitKey(25) & 0xff == ord('q')
    # img_count += 1  
vid.release()
cv2.destroyAllWindows()

