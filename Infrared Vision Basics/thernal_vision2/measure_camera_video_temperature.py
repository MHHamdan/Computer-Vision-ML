# import the necessary packages
import cv2
import numpy as np

# create mouse global coordinates
x_mouse = 0
y_mouse = 0

# mouse events function
def mouse_events(event, x, y, flags, param):
    # mouse movement event
    if event == cv2.EVENT_MOUSEMOVE:

        # update mouse global coordinates
        global x_mouse
        global y_mouse

        x_mouse = x
        y_mouse = y

# set up the thermal camera index (thermal_camera = cv2.VideoCapture(0, cv2.CAP_DSHOW) on Windows OS)
thermal_camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# set up the thermal camera resolution
thermal_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
thermal_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)

# set up the thermal camera to get the gray16 stream and raw data
thermal_camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('Y', '1', '6', ' '))
thermal_camera.set(cv2.CAP_PROP_CONVERT_RGB, 0)

# set up mouse events and prepare the thermal frame display
grabbed, frame_thermal = thermal_camera.read()
cv2.imshow('gray8', frame_thermal)
cv2.setMouseCallback('gray8', mouse_events)

# loop over the thermal camera frames
while True:

    # grab the frame from the thermal camera stream
    (grabbed, thermal_frame) = thermal_camera.read()

    # calculate temperature
    temperature_pointer = thermal_frame[y_mouse, x_mouse]
    # temperature_pointer = (temperature_pointer / 100) - 273.15
    temperature_pointer = (temperature_pointer / 100) * 9 / 5 - 459.67

    # convert the gray16 image into a gray8
    cv2.normalize(thermal_frame, thermal_frame, 0, 255, cv2.NORM_MINMAX)
    thermal_frame = np.uint8(thermal_frame)
  
    # colorized the gray8 image using OpenCV colormaps
    thermal_frame = cv2.applyColorMap(thermal_frame, cv2.COLORMAP_INFERNO)
    
    # write pointer
    cv2.circle(thermal_frame, (x_mouse, y_mouse), 2, (255, 255, 255), -1)

    # write temperature
    cv2.putText(thermal_frame, "{0:.1f} Fahrenheit".format(temperature_pointer), (x_mouse - 40, y_mouse - 15), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
    
    # show the thermal frame
    cv2.imshow('gray8', thermal_frame)
    cv2.waitKey(1)

# do a bit of cleanup
thermal_camera.release()
cv2.destroyAllWindows()
