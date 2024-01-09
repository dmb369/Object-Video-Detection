import cv2
from ultralytics import YOLO
import numpy as np

#import torch
#print(torch.backends.mps.is_available())   #checking whether the system has MPS (Metal Performance Shaders) 
                                            #Metal is Apple's API for programming Metal GPU
cap = cv2.VideoCapture("dogs.mp4")

model = YOLO("yolov8m.pt")

while True:

    ret, frame = cap.read()
    if not ret:
        break      #to avoid the error when there are no more frames

    results = model(frame, device="mps")   #accessing the mps to increase the GPU performance
    result = results[0] 
    bboxes = np.array(result.boxes.xyxy.cpu(), dtype='int')  #bounding boxes , also we want the datatype of the tensor as int

    # adding classes and the confidence score
    classes = np.array(result.boxes.cls.cpu(), dtype="int")

 
    for bbox, cls in zip(bboxes,classes):
        (x, y, x2, y2) = bbox
        cv2.rectangle(frame, (x,y), (x2,y2), (0,0,225), 2)  # bgr format + thickness level
        cv2.putText(frame, str(cls), (x, y-5), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 225), 2)
        #print("x", x, "y", y)
        
    # print(bboxes)

    cv2.imshow("Img", frame)
    key = cv2.waitKey(0)
    if key == 27:          #the Esc key is corresponded to number 27
        break


cap.release()    #to release the video
cv2.destroyAllWindows() 