import cv2 #import cv2
import numpy as np #import numpy
cap = cv2.VideoCapture(0)
while True:
        _, frame = cap.read()
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define range for green color in HSV
        lower = np.array([0, 42, 0]) #check  values=[0,0,0]& [1,1,1]
        upper= np.array([179, 255, 255])
        #every color expect white
        mask = cv2.inRange(hsv, lower, upper)
        # Invert mask to get white color
        result = cv2.bitwise_not(mask)
        
        cv2.imshow("Frame", frame)
        cv2.imshow("Result", result)
        
        key=cv2.waitKey(1) 
        if key==27:  # Press 'Esc' key to exit
            break

cap.release()
cv2.destroyAllWindows()