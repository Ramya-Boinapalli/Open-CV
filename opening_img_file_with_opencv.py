import cv2 # Import OpenCV library

img1 = cv2.imread(r"C:\Users\Ramya\OneDrive\Desktop\rose.jpg")
img2 = cv2.imread(r"C:\Users\Ramya\OneDrive\Desktop\rose.jpg")
while True: # Infinite loop to display images 
    cv2.imshow('Image 1', img1) # Display first image
    cv2.imshow('Image 2', img2) # Display second image
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break # Exit loop on ESC key press

cv2.destroyAllWindows() # Close all OpenCV windows