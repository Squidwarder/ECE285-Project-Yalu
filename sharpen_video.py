import cv2
import numpy as np

"""

Initially tried to stabilize footage, but optical flow weren't tracking at all. The result was abysmal.
    
Feature detection: Detect good features to track in the video. You can use functions like cv2.goodFeaturesToTrack() for this1.
Optical flow: Track the detected features in the video using optical flow. You can use functions like cv2.calcOpticalFlowPyrLK() for this1.
Transformation matrix: Estimate the transformation (e.g., translation, rotation) between frames using the tracked features. 
You can use functions like cv2.estimateAffinePartial2D() or cv2.findHomography() for this1.
Warp frames: Apply the inverse of the estimated transformation to the frames to align them and thus stabilize the video. 
You can use functions like cv2.warpAffine() for this1.
    
"""


# The video feed is read in as a VideoCapture object
cap = cv2.VideoCapture(1)

# Define the sharpening kernel
kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])


# TODO: Other methods
# 1. unsharp masking
# 2. super resolution with cv2.dnn_superres.
# 3. histogram equalization & adaptive HE


# Create an SR object
sr = cv2.dnn_superres.DnnSuperResImpl_create()

# Read the desired model
path = "EDSR_x4.pb"  # replace with the path to your model
sr.readModel(path)

# Set the desired model and scale to get correct pre- and post-processing
sr.setModel("edsr", 4)  # replace "edsr" with your model's name, and 4 with the desired scale

while True:
    # Capture the video frame
    ret, frame = cap.read()
    
    if not ret:
        break

    # Apply the sharpening kernel to the frame using the filter2D function
    # same depth as source
    sharpened = cv2.filter2D(frame, -1, kernel)    
    
    # cv2.imshow('Original window', frame)
    # Display the sharpened frame
    cv2.imshow('Sharpened window', sharpened)        
    

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('Sharpened window', cv2.WND_PROP_VISIBLE) < 1:
        dnn_sharpen = sr.upsample(frame)
        cv2.imshow("DNN sharpened", dnn_sharpen)
        break

# When everything done, release the capture and destroy the window
cap.release()
cv2.destroyAllWindows()
