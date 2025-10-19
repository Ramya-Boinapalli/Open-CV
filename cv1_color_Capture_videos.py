import cv2 
import numpy


cap = cv2.VideoCapture(0)

while True:
     _,frame = cap.read()
     
     hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
     
     cv2.imshow("Original Frame",frame)
     cv2.imshow("HSV Frame",hsv)
     
     if cv2.waitKey(1) & 0xFF == ord('q'):
         break

cap.release()
cv2.destroyAllWindows()