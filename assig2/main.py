import cv2
from matplotlib import pyplot as plt
import numpy as np
import math

def hough_lines(image_name):
    src = cv2.imread('assets/' + image_name + '.jpg')
    dst = cv2.Canny(src, 50, 200, None, 3)

    image_lines = cv2.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)
    
    if image_lines is not None:
        for i in range(0, len(image_lines)):
            rho = image_lines[i][0][0]
            theta = image_lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv2.line(dst, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)

    cv2.imshow("Source", src)
    cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", dst)

    cv2.waitKey()

if __name__ == '__main__':
    images =