import cv2
import imutils
import numpy as np

padding = 0


def remove_black_edges(img):
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


def test_remove_black_edges():
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


if __name__ == "__main__":
    test_remove_black_edges()