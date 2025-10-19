import cv2
import numpy as np

cap = cv2.VideoCapture(0)
while True:
     _, frame = cap.read()
     
     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
     
     # Define range for red color in HSV
     lower_red = np.array([161,155,84])
     upper_red = np.array([179,255,255])
     lower_red2 = np.array([0,155,84])
     upper_red2 = np.array([10,255,255])
     # Create masks for red color
     mask1 = cv2.inRange(hsv, lower_red, upper_red)
     mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
     red_mask = mask1 | mask2
     
     # Bitwise-AND mask and original image
     red_output = cv2.bitwise_and(frame, frame, mask=red_mask)
     
     cv2.imshow("Original Frame", frame)
     cv2.imshow("Red Color Mask", red_output)
     
     if cv2.waitKey(1) & 0xFF == ord('q'):
         break

cap.release()
cv2.destroyAllWindows()