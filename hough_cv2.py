from dataset_canny import dataset
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

src = dataset[1][0]

canny = cv2.Canny(src, 50, 200, None, 3)

dst = src.copy()

lines = cv2.HoughLines(canny, 1, np.pi / 180, 150, None, 0, 0)

if lines is not None:
    for line in lines:
        rho = line[0][0]
        theta = line[0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        cv2.line(dst, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)

cv2.imshow("src", src)
cv2.imshow("canny", canny)
cv2.imshow("dst", dst)

k = cv2.waitKey(0)