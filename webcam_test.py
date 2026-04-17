import cv2

#  Open webcam
cap = cv2.VideoCapture(0)

# Check if webcam opened successfully
if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

print("Webcam opened successfully! Press 'q' to quit.")

#  Read frames in a loop
while True:
    ret, frame = cap.read()

    # If frame failed to read, stop
    if not ret:
        print("Failed to read frame")
        break

    # Show the frame
    cv2.imshow("Object Tracking", frame)

    #  Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#  Clean up
cap.release()
cv2.destroyAllWindows()
print("Webcam released. Done.")