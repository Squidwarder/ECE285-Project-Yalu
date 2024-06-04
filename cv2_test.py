import cv2

def capture_video():
    cap = cv2.VideoCapture(1) #! 0 is internal webcam, 1 works for usb cam

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Display the resulting frame
            cv2.imshow('Video Stream', frame)

            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('Video Stream', cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Stream stopped")

# Call the function to start capturing video
capture_video()
