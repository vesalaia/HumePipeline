"""
Orientation utilities
"""
from skimage.transform import hough_line, hough_line_peaks
from skimage.transform import rotate
from skimage.feature import canny
from skimage.io import imread
from skimage.color import rgb2gray
import cv2
import numpy as np
import math
def detect_rotation_angle(filename):
    """
    Detect rotation using hough transformation method
    Average of detected lines longer than 40 pixels
    """
    src = cv2.imread(cv2.samples.findFile(filename), cv2.IMREAD_GRAYSCALE)

    dst = cv2.Canny(src, 50, 200, None, 3)

    # Copy edges to the images that will display the results in BGR
    cdstP = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)

    lines = cv2.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)
    cnt = 0
    tot_angle = 0
    if lines is not None:
        for i in range(0, len(lines)):
            l = lines[i][0]
            deltaY = l[3] - l[1];
            deltaX = l[2] - l[0];
            if abs(deltaX) > 40:
                angle = math.atan2(deltaY, deltaX) * 180 / np.pi
                tot_angle += angle
                cnt += 1
    print("Average angle: {}".format(tot_angle / cnt))
    return tot_angle / cnt
