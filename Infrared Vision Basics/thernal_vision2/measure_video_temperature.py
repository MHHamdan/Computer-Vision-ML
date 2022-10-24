# import the necessary packages
import cv2
import numpy as np
import os
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True, help="path of the video sequence")
args = vars(ap.parse_args())

# create mouse global coordinates
x_mouse = 0
y_mouse = 0

# create thermal video fps variable (8 fps in this case)
fps = 8

# mouse events function
def mouse_events(event, x, y, flags, param):
    # mouse movement event
    if event == cv2.EVENT_MOUSEMOVE:

        # update mouse global coordinates
        global x_mouse
        global y_mouse

        x_mouse = x
        y_mouse = y

# set up mouse events and prepare the thermal frame display
gray16_frame = cv2.imread("lighter_gray16_image.tiff", cv2.IMREAD_ANYDEPTH)
cv2.imshow('gray8', gray16_frame)
cv2.setMouseCallback('gray8', mouse_events)

# loop over the thermal video frames
for image in sorted(os.listdir(args["video"])):

    # filter .tiff files (gray16 images)
    if image.endswith(".tiff"):
    
        # define the gray16 frame path
        file_path = os.path.join(args["video"], image)
        
        # open the gray16 frame
        gray16_frame = cv2.imread(file_path, cv2.IMREAD_ANYDEPTH)

        # calculate temperature
        temperature_pointer = gray16_frame[y_mouse, x_mouse]
        #temperature_pointer = (temperature_pointer / 100) - 273.15
        temperature_pointer = (temperature_pointer / 100) * 9 / 5 - 459.67

        # convert the gray16 frame into a gray8
        gray8_frame = np.zeros((120, 160), dtype=np.uint8)
        gray8_frame = cv2.normalize(gray16_frame, gray8_frame, 0, 255, cv2.NORM_MINMAX)
        gray8_frame = np.uint8(gray8_frame)
        
        # colorized the gray8 frame using OpenCV colormaps
        gray8_frame = cv2.applyColorMap(gray8_frame, cv2.COLORMAP_INFERNO)

        # write pointer
        cv2.circle(gray8_frame, (x_mouse, y_mouse), 2, (255, 255, 255), -1)

        # write temperature
        cv2.putText(gray8_frame, "{0:.1f} Fahrenheit".format(temperature_pointer), (x_mouse - 40, y_mouse - 15), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

        # show the thermal frame
        cv2.imshow("gray8", gray8_frame)

        # wait 125 ms: RGMVision ThermalCAM1 frames per second = 8
        cv2.waitKey(int((1 / fps) * 1000))


# do a bit of cleanup
cv2.destroyAllWindows()
