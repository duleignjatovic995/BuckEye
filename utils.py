import cv2
import glob

import imutils
import numpy as np


def remove_black_edges(img):
    padding = 0
    # img = cv2.imread(image)
    bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rows, cols = bw.shape

    non_empty_columns = np.where(bw.max(axis=0) > 0)[0]
    non_empty_rows = np.where(bw.max(axis=1) > 0)[0]
    crop_box = (min(non_empty_rows) * (1 - padding),
                min(max(non_empty_rows) * (1 + padding), rows),
                min(non_empty_columns) * (1 - padding),
                min(max(non_empty_columns) * (1 + padding), cols))

    final = img[crop_box[0] + 20:crop_box[1] - 20, crop_box[2]:crop_box[3] + 1, :]
    return final


def crop_redundant(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, th2 = cv2.threshold(gray, 8, 255, cv2.THRESH_BINARY)
    _, contours, hierarchy = cv2.findContours(th2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(contour) for contour in contours]
    max_index = np.argmax(areas)
    cnt = contours[max_index]
    x, y, w, h = cv2.boundingRect(cnt)

    # Ensure bounding rect should be at least 16:9 or taller
    if w / h > 16 / 9:
        # increase top and bottom margin
        newHeight = w / 16 * 9
        y = y - (newHeight - h) / 2
        h = newHeight

    crop = image[y:y + h, x:x + w]
    return crop


def check_rotation(img):
    rows, cols, _ = img.shape

    if rows / cols > 4 / 3:
        print("Vertikala ekstremna")
    elif cols / rows > 4 / 3:
        print("horizontala ekstremna")
    elif rows / cols <= 4 / 3 or cols / rows <= 4 / 3:
        print("U granicama")

        # M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
        # dst = cv2.warpAffine(img, M, (cols, rows))


def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    face_rects = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in face_rects:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)


def remove_black_edges_v0():
    # 'images/bs.jpeg'
    # "images/crna1.jpg"
    image = cv2.imread("images/crna3.jpg")

    proc = remove_black_edges(image)
    proc = imutils.resize(proc, height=400)

    image = imutils.resize(image, height=400)
    show = np.concatenate((image, proc), axis=1)
    cv2.imshow("ljepotica", show)
    # cv2.imshow("ljepotica", proc)
    # cv2.imshow("kurotica", image)
    cv2.waitKey(0)


def remove_black_edges1():
    for filename in glob.iglob("images/*"):
        image = cv2.imread(filename)
        proc = remove_black_edges(image.copy())
        # detect_faces(proc)
        check_rotation(proc)
        proc = imutils.resize(proc, height=400)
        image = imutils.resize(image, height=400)
        show = np.concatenate((image, proc), axis=1)
        cv2.imshow("ljepotica", show)
        # cv2.imshow("ljepotica", proc)
        # cv2.imshow("kurotica", image)
        print('%s' % filename)
        cv2.waitKey(0)
        # input(">>")


if __name__ == "__main__":
    remove_black_edges1()
