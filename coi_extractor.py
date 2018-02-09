import cv2
import imutils


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


image = cv2.imread(slika10)

# convert the image to grayscale, blur it, and find edges
# in the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
# edged = cv2.adaptiveThreshold(gray, 250, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 3)
# gray = cv2.Canny(gray, 0, 3)
gray = cv2.bilateralFilter(gray, 9, 41, 41)
# _, gray = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 20)
# gray = imutils.auto_canny(gray)
# gray = cv2.blur(gray, (21, 21))

# find the contours in the edged image, keeping only the
# largest ones, and initialize the screen contour
# (_, contours, _) = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# contours = sorted(contours, key=cv2.contourArea, reverse=True)[0]
# print(len(contours))
#
# cv2.drawContours(image, contours, -1, (0, 0, 255), 3)
img_sw = imutils.resize(image, height=800)
cv2.imshow("Image", img_sw)
edg_sw = imutils.resize(gray, height=800)
cv2.imshow("Edged", edg_sw)
cv2.waitKey(0)

cv2.reduce()