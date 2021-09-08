import os
import cv2 as cv
import matplotlib.pyplot as plt


def calculateAndSaveHistogram(img, dst_filename):
    plt.hist(img.ravel(), bins=256, range=[0, 256])
    plt.savefig(dst_filename, bbox_inches='tight')


def calculateAndShowHistogram(img):
    plt.hist(img.ravel(), bins=256, range=[0, 256])
    plt.show()


def histeqAndSave(img, dst_name):
    dst_img = cv.equalizeHist(src=img)
    cv.imwrite(dst_name, dst_img)


def histEqAllImages(src_dir, dst_dir):
    for img_name in os.listdir(src_dir):
        src_name = os.path.join(src_dir, img_name)
        img = cv.imread(os.path.join(src_dir, img_name), 0)
        dst_name = os.path.join(dst_dir, img_name)
        dst_img = cv.equalizeHist(img)
        cv.imwrite(dst_name, dst_img)


def superDirectorySize(directory):
    total_size = 0
    for dir in os.listdir(directory):
        total_size += directorySize(os.path.join(directory, dir))

    return total_size


def directorySize(directory):
    return len(os.listdir(directory))