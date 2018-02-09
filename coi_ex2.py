import cv2
import imutils
import numpy as np

padding = 0
img = cv2.imread('images/ax.jpeg')
bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
rows, cols = bw.shape

non_empty_columns = np.where(bw.max(axis=0) > 0)[0]
non_empty_rows = np.where(bw.max(axis=1) > 0)[0]
cropBox = (min(non_empty_rows) * (1 - padding),
            min(max(non_empty_rows) * (1 + padding), rows),
            min(non_empty_columns) * (1 - padding),
            min(max(non_empty_columns) * (1 + padding), cols))

cv2.imshow("ljepotica", img[cropBox[0]+1:cropBox[1]-1, cropBox[2]:cropBox[3]+1, :])
cv2.waitKey(0)