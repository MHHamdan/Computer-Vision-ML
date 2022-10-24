# import the necessary packages
import numpy as np
import cv2

# open the gray16 image
gray16_image = cv2.imread("lighter_gray16_image.tiff", cv2.IMREAD_ANYDEPTH)

# get the first gray16 value

# pixel coordinates
x = 90
y = 100

pixel_flame_gray16  = gray16_image [y, x]

# calculate temperature value in °C
#pixel_flame_gray16  = (pixel_flame_gray16 / 100) - 273.15

# calculate temperature value in °F
pixel_flame_gray16  = (pixel_flame_gray16 / 100) * 9 / 5 - 459.67

# convert the gray16 image into a gray8 to show the result
gray8_image = np.zeros((120,160), dtype=np.uint8)
gray8_image = cv2.normalize(gray16_image, gray8_image, 0, 255, cv2.NORM_MINMAX)
gray8_image = np.uint8(gray8_image)

# write pointer
cv2.circle(gray8_image, (x, y), 2, (0, 0, 0), -1)
cv2.circle(gray16_image, (x, y), 2, (0, 0, 0), -1)

# write temperature value in gray8 and gray16 image
cv2.putText(gray8_image,"{0:.1f} Fahrenheit".format(pixel_flame_gray16),(x - 80, y - 15), cv2.FONT_HERSHEY_PLAIN, 1,(255,0,0),2)
cv2.putText(gray16_image,"{0:.1f} Fahrenheit".format(pixel_flame_gray16),(x - 80, y - 15), cv2.FONT_HERSHEY_PLAIN, 1,(255,0,0),2)

# show result
cv2.imshow("gray8-fahrenheit", gray8_image)
cv2.imshow("gray16-fahrenheit", gray16_image)
cv2.waitKey(0)
