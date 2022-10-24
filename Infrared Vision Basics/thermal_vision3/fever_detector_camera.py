# import the necessary packages
import cv2
import numpy as np

# set up the thermal camera index (thermal_camera = cv2.VideoCapture(0, cv2.CAP_DSHOW) on Windows OS)
thermal_camera = cv2.VideoCapture(0)

# set up the thermal camera resolution
thermal_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
thermal_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)

# set up the thermal camera to get the gray16 stream and raw data
thermal_camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('Y', '1', '6', ' '))
thermal_camera.set(cv2.CAP_PROP_CONVERT_RGB, 0)

# load the haar cascade face detector
haar_cascade_face = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")

# fever temperature threshold in Celsius or Fahrenheit
fever_temperature_threshold = 37.0
#fever_temperature_threshold = 99.0

# loop over the thermal camera frames
while True:

    # grab the frame from the thermal camera stream
    (grabbed, gray16_frame) = thermal_camera.read()

    # convert the gray16 image into a gray8
    gray8_frame = np.zeros((120, 160), dtype=np.uint8)
    gray8_frame = cv2.normalize(gray16_frame, gray8_frame, 0, 255, cv2.NORM_MINMAX)
    gray8_frame = np.uint8(gray8_frame)

    # color the gray8 image using OpenCV colormaps
    gray8_frame = cv2.applyColorMap(gray8_frame, cv2.COLORMAP_INFERNO)

    # detect faces in the input frame using the haar cascade face detector
    faces = haar_cascade_face.detectMultiScale(gray8_frame, scaleFactor=1.1, minNeighbors=5, minSize=(10, 10),
                                               flags=cv2.CASCADE_SCALE_IMAGE)
    # loop over the bounding boxes to measure their temperature
    for (x, y, w, h) in faces:

        # draw the rectangles
        cv2.rectangle(gray8_frame, (x, y), (x + w, y + h), (255, 255, 255), 1)

        # define the roi with a circle at the haar cascade origin coordinate

        # haar cascade center for the circle
        haar_cascade_circle_origin = x + w // 2, y + h // 2

        # circle radius
        radius = w // 4

        # get the 8 most significant bits of the gray16 image
        # (we follow this process because we can't extract a circle
        # roi in a gray16 image directly)
        gray16_high_byte = (np.right_shift(gray16_frame, 8)).astype('uint8')

        # get the 8 less significant bits of the gray16 image
        # (we follow this process because we can't extract a circle
        # roi in a gray16 image directly)
        gray16_low_byte = (np.left_shift(gray16_frame, 8) / 256).astype('uint16')

        # apply the mask to our 8 most significant bits
        mask = np.zeros_like(gray16_high_byte)
        cv2.circle(mask, haar_cascade_circle_origin, radius, (255, 255, 255), -1)
        gray16_high_byte = np.bitwise_and(gray16_high_byte, mask)

        # apply the mask to our 8 less significant bits
        mask = np.zeros_like(gray16_low_byte)
        cv2.circle(mask, haar_cascade_circle_origin, radius, (255, 255, 255), -1)
        gray16_low_byte = np.bitwise_and(gray16_low_byte, mask)

        # create/recompose our gray16 roi
        gray16_roi = np.array(gray16_high_byte, dtype=np.uint16)
        gray16_roi = gray16_roi * 256
        gray16_roi = gray16_roi | gray16_low_byte

        # estimate the face temperature by obtaining the higher value
        higher_temperature = np.amax(gray16_roi)

        # calculate the temperature
        higher_temperature = (higher_temperature / 100) - 273.15
        # higher_temperature = (higher_temperature / 100) * 9 / 5 - 273.15

        # write temperature value in gray8
        if higher_temperature < fever_temperature_threshold:

            # white text: normal temperature
            cv2.putText(gray8_frame, "{0:.1f} Celsius".format(higher_temperature), (x, y - 10), cv2.FONT_HERSHEY_PLAIN,
                        1, (255, 255, 255), 1)
        else:
            # - red text + red circle: fever temperature
            cv2.putText(gray8_frame, "{0:.1f} Celsius".format(higher_temperature), (x, y - 10), cv2.FONT_HERSHEY_PLAIN,
                        1, (0, 0, 255), 2)
            cv2.circle(gray8_frame, haar_cascade_circle_origin, radius, (0, 0, 255), 2)

    # show the temperature results
    cv2.imshow("final", gray8_frame)
    cv2.waitKey(1)
    
# do a bit of cleanup
thermal_camera.release()
cv2.destroyAllWindows()