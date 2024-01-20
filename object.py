#loading initial library
from ultralytics import YOLO
import cv2
# loding model
model = YOLO('yolov8n.pt')

# loading video
path = './WhatsApp Video 2024-01-19 at 17.35.44_326050cb'

# read frmaes
cap = cv2.VideoCapture(path)
ret = True
while ret:
    ret,frame = cap.read()
    if ret==True:
        # detecting objects from frame
        # track object
        result = model.track(frame, persist=True)

        #plot results
        frame_ = result[0].plot()

        # visulaize
        cv2.imshow('frame',frame_)
        if cv2.waitKey(1)==ord('q'):
            break

cap.release()
cv2.destroyAllWindows()