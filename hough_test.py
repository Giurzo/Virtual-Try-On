from dataset import dataset
import cv2
import numpy as np

from hough_hough_space_implemented import *

import matplotlib.pyplot as plt


import time
start = time.time()

src = dataset[0][0]
canny = cv2.Canny(src, 50, 200, None, 3)
cv2.HoughLines(canny, 1, np.pi / 180, 150, None, 0, 0)

end = time.time()
print(f"Runtime of the program is {end - start}")

###

start = time.time()

src = dataset[0][0]
canny = cv2.Canny(src, 50, 200, None, 3)
hough_space(canny)

end = time.time()
print(f"Runtime of the program is {end - start}")

#hough_space_print_lines(canny)