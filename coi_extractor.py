import cv2
import imutils
import numpy as np


slika0= "images/ogavna.jpg"
slika1 = "images/aw.jpg"
slika2 = "images/ae.jpeg"
slika3 = "images/ai.jpeg"
slika4 = "images/as.jpg"
slika5 = "images/at.jpg"
slika6 = "images/az.jpeg"
slika7 = "images/bf.jpeg"
slika8 = "images/bm.jpg"
slika9 = "images/bz.jpeg"  # nosite se u kurac
slika10 = "images/ao.jpeg"  # jebem vam mater
slika11 = "images/bs.jpeg"
slika12 = "images/crna2.jpg"


image = cv2.imread(slika12)  # read image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # image 2 grayscale


# Blurring
gray = cv2.GaussianBlur(gray, (5, 5), 0)
# gray = cv2.medianBlur(gray, (5, 5))
# gray = cv2.blur(gray, (21, 21))
# gray = cv2.bilateralFilter(gray, 9, 41, 41)

# Thresholding
# _, gray = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 10)


# Laplacian
# gray = cv2.Laplacian(gray, cv2.CV_64F)
# gray = np.uint8(np.absolute(gray))


# Sobel
# sobelX = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
# sobelY = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
# sobelZ = cv2.bitwise_or(sobelX, sobelY)
# gray = sobelX
# gray = np.uint8(np.absolute(gray))


# Canny
gray = imutils.auto_canny(gray)
# gray = cv2.Canny(gray, 25, 150)


# Contours
# (_, contours, _) = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# contours = sorted(contours, key=cv2.contourArea, reverse=True)[0]
# print(len(contours))
# cv2.drawContours(image, contours, -1, (0, 0, 255), 3)


img_sw = imutils.resize(image, height=800)
cv2.imshow("Image", img_sw)
edg_sw = imutils.resize(gray, height=800)
cv2.imshow("Edged", edg_sw)
cv2.waitKey(0)

# cv2.reduce() # TODO
