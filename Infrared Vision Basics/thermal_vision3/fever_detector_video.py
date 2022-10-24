# import the necessary packages
import cv2
import numpy as np
import os
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True, help="path of the video sequence")
args = vars(ap.parse_args())

# load the haar cascade face detector
haar_cascade_face = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")

# create thermal video fps variable (8 fps in this case)
fps = 8

# loop over the thermal video frames to detect faces
for image in sorted(os.listdir(args["video"])):

    # filter .tiff files (gray16 images)
    if image.endswith(".tiff"):

        # define the gray16 frame path
        file_path = os.path.join(args["video"], image)

        # open the gray16 frame
        gray16_frame = cv2.imread(file_path, cv2.IMREAD_ANYDEPTH)

        # convert the gray16 image into a gray8
        gray8_frame = np.zeros((120, 160), dtype=np.uint8)
        gray8_frame = cv2.normalize(gray16_frame, gray8_frame, 0, 255, cv2.NORM_MINMAX)
        gray8_frame = np.uint8(gray8_frame)

        # color the gray8 image using OpenCV colormaps
        gray8_frame = cv2.applyColorMap(gray8_frame, cv2.COLORMAP_INFERNO)

        # detect faces in the input image using the haar cascade face detector
        faces = haar_cascade_face.detectMultiScale(gray8_frame, scaleFactor=1.1, minNeighbors=5, minSize=(10, 10),
                                                   flags=cv2.CASCADE_SCALE_IMAGE)
        # loop over the bounding boxes
        for (x, y, w, h) in faces:
            # draw the rectangles
            cv2.rectangle(gray8_frame, (x, y), (x + w, y + h), (255, 255, 255), 1)

        # show results
        cv2.imshow("gray8-face-detected", gray8_frame)

        # wait 125 ms: RGMVision ThermalCAM1 frames per second = 8
        cv2.waitKey(int((1 / fps) * 1000))

# fever temperature threshold in Celsius and Fahrenheit
fever_temperature_threshold = 37.0
#fever_temperature_threshold = 99.0

# loop over the thermal video frames to detect faces and measure their temperature
for image in sorted(os.listdir(args["video"])):

    # filter .tiff files (gray16 images)
    if image.endswith(".tiff"):

        # define the gray16 frame path
        file_path = os.path.join(args["video"], image)

        # open the gray16 frame
        gray16_frame = cv2.imread(file_path, cv2.IMREAD_ANYDEPTH)

        # convert the gray16 image into a gray8
        gray8_frame = np.zeros((120, 160), dtype=np.uint8)
        gray8_frame = cv2.normalize(gray16_frame, gray8_frame, 0, 255, cv2.NORM_MINMAX)
        gray8_frame = np.uint8(gray8_frame)

        # color the gray8 image using OpenCV colormaps
        gray8_frame = cv2.applyColorMap(gray8_frame, cv2.COLORMAP_INFERNO)

        # detect faces in the input image using the haar cascade face detector
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

            # apply the mask to our 8 most significant bits
            mask = np.zeros_like(gray16_high_byte)
            cv2.circle(mask, haar_cascade_circle_origin, radius, (255, 255, 255), -1)
            gray16_high_byte = np.bitwise_and(gray16_high_byte, mask)

            # get the 8 most significant bits of the gray16 image
            # (we follow this process because we can't extract a circle
            # roi in a gray16 image directly)
            gray16_low_byte = (np.left_shift(gray16_frame, 8) / 256).astype('uint16')

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
                cv2.putText(gray8_frame, "{0:.1f} Celsius".format(higher_temperature), (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN,
                            1, (255, 255, 255), 1)
            else:
                # - red text + red circle: fever temperature
                cv2.putText(gray8_frame, "{0:.1f} Celsius".format(higher_temperature), (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN,
                            1, (0, 0, 255), 2)
                cv2.circle(gray8_frame, haar_cascade_circle_origin, radius, (0, 0, 255), 2)

            # show results
            cv2.imshow("gray8-face-temperature", gray8_frame)

            # wait 125 ms: RGMVision ThermalCAM1 frames per second = 8
            cv2.waitKey(int((1 / fps) * 1000))
