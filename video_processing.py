import cv2

# Open webcam
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("❌ Error: Could not open camera.")
    exit()

# Get frame width and height
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"🎥 Camera Resolution: {width} x {height}")

# Start video loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to grab frame.")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Show both color and grayscale in one window
    combined = cv2.hconcat([frame, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)])
    cv2.imshow('Color (Left) | Grayscale (Right)', combined)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release and close
cap.release()
cv2.destroyAllWindows()
