import cv2
import numpy as np
cap = cv2.VideoCapture(0)
while True:
        _, frame = cap.read()
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define range for blue color in HSV
        lower_blue = np.array([94, 80, 2])
        upper_blue = np.array([126, 255, 255])
        
        # Create mask for blue color
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Bitwise-AND mask and original image
        blue_output = cv2.bitwise_and(frame, frame, mask=blue_mask)
        
        cv2.imshow("Original Frame", frame)
        cv2.imshow("Blue Color Mask", blue_output)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()