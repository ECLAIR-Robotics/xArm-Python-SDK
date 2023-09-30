# Scope: Use an OpenCV model to detect and send coordinates of the tops of two beakers

'''
Important things I need:
* The beakers to test code I write
* A good approach to detecting the beakers and getting the locations:
* 
'''

import cv2
import numpy as np

# Define a video capture object
vid = cv2.VideoCapture(0)

#### MAIN VIDEO CAPTURE LOOP ####
while(True):

    # Capture video frame by frame and get a grayscale version
    ret, frame = vid.read()
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Use grayscale version for Hough Transform
    '''
    QUESTION: HOW DO THE ARGS WORK, READ PAPER FOR PARAM1 AND PARAM2: https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=f8785ba3a56b5ce90fb264e82dacaca1ac641091
    '''
    circles = cv2.HoughCircles(gray_frame, cv2.HOUGH_GRADIENT, 1.0, 10, param1=60,param2=75,minRadius=0,maxRadius=50)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(frame, (x, y), r, (0, 255, 0), 4)

    # Display frame
    cv2.imshow('frame', frame)

    # press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


  
vid.release()
cv2.destroyAllWindows()