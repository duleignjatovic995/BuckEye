import cv2
import numpy as np
import glob
import imutils


def remove_black_edges(img):
    # img = cv2.imread(image)
    bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rows, cols = bw.shape

    non_empty_columns = np.where(bw.max(axis=0) < 255)[0]
    non_empty_rows = np.where(bw.max(axis=1) < 255)[0]
    crop_box = (min(non_empty_rows),
                min(max(non_empty_rows), rows),
                min(non_empty_columns),
                min(max(non_empty_columns), cols))

    final = img[crop_box[0] + 20:crop_box[1] - 20, crop_box[2]:crop_box[3] + 1, :]
    return final


def jebi_mu_majku_u_picku(image):
    gray = cv2.threshold(image.copy(), 34, 255, cv2.THRESH_BINARY)[1]
    cv2.imshow("sisaj kurac", gray)
    cv2.waitKey(0)


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


if __name__ == "__main__":
    image = cv2.imread("images1/aa.jpg")
    cropped = crop_redundant(image)
    showed = imutils.resize(cropped.copy(), height=500)
    oridjidji = imutils.resize(image.copy(), height=500)
    cv2.imshow("slika1", showed)
    cv2.imshow("slika2", oridjidji)
    cv2.waitKey(0)

    # image = cv2.imread("budalabela.jpg")
    # image = cv2.imread("images/bf.jpeg")
    # cv2.imshow("jebem", image)
    # proc = remove_black_edges(image.copy())
    # jebi_mu_majku_u_picku(image)
    # for filename in glob.iglob("images/*"):
    #     image = cv2.imread(filename)
    #     proc = remove_black_edges(image.copy())
    #     proc = imutils.resize(proc, height=400)
    #     image = imutils.resize(image, height=400)
    #     show = np.concatenate((image, proc), axis=1)
    #     print('%s' % filename)
    #     cv2.imshow("ljepotica", show)
    #     # cv2.imshow("ljepotica", proc)
    #     # cv2.imshow("kurotica", image)
    #
    #     cv2.waitKey(0)
