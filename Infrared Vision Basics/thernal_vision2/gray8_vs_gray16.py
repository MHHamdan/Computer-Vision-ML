# import the necessary packages
import numpy as np
import cv2

# open the gray16 image
gray16_image = cv2.imread("lighter_gray16_image.tiff", cv2.IMREAD_ANYDEPTH)

# convert the gray16 image into a gray8
gray8_image = np.zeros((120,160), dtype=np.uint8)
gray8_image = cv2.normalize(gray16_image, gray8_image, 0, 255, cv2.NORM_MINMAX)
gray8_image = np.uint8(gray8_image)

# colorized the gray8 image using OpenCV colormaps
inferno_palette = cv2.applyColorMap(gray8_image, cv2.COLORMAP_INFERNO)
jet_palette = cv2.applyColorMap(gray8_image, cv2.COLORMAP_JET)
viridis_palette = cv2.applyColorMap(gray8_image, cv2.COLORMAP_VIRIDIS)

# show the different thermal color palettes
cv2.imshow("grayscale", gray8_image)
cv2.imshow("inferno", inferno_palette)
cv2.imshow("jet", jet_palette)
cv2.imshow("viridis", viridis_palette)
cv2.waitKey(0)