import cv2
import numpy as np
cap = cv2.VideoCapture(0)
while True:
        _, frame = cap.read()
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define range for green color in HSV
        lower_green = np.array([40, 100, 100])
        upper_green = np.array([102, 255, 255])
        
        # Create mask for green color
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Bitwise-AND mask and original image
        green_output = cv2.bitwise_and(frame, frame, mask=green_mask)
        
        cv2.imshow("Original Frame", frame)
        cv2.imshow("Green Color Mask", green_output)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()