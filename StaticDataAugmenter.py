import os
import random
import cv2 as cv
import numpy as np


def augmentImage(img, num, dst_dir):
    dst = randomRotation(img, -90, 90)
    dst = randomTranslation(dst, 100, 100)
    r = random.randint(0, 1)
    if r == 1:
        cv.flip(dst, random.randint(0, 1))
    dst = randomScaling(dst, 0.7, 1.2)

    name = "retina" + str(num) + ".jpg"
    name = os.path.join(dst_dir, name)
    cv.imwrite(name, dst)


def testfunction(img, width_shift, height_shift):
    dst = translateImage(img, width_shift, height_shift)

    cv.imshow('img',dst)
    cv.waitKey(0)
    cv.destroyAllWindows()


def rotateImage(img, angle):
    rows, cols = img.shape[0], img.shape[1]

    M = cv.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    dst = cv.warpAffine(img, M, (cols, rows))

    return dst


def scaleImage(img, factor):
    #height, width = img.shape[:2]
    dst = cv.resize(img,None,fx=factor, fy=factor, interpolation = cv.INTER_CUBIC)

    return dst


def randomScaling(img, min_factor, max_factor):
    r = random.randint(10*min_factor, 10*max_factor)
    r /= 10
    return scaleImage(img, r)


def randomRotation(img, min_angle, max_angle):
    ra = random.randint(min_angle, max_angle)

    return rotateImage(img, ra)


def translateImage(img, width_shift, height_shift):
    rows, cols = img.shape[0], img.shape[1]

    M = np.float32([[1, 0, width_shift], [0, 1, height_shift]])
    dst = cv.warpAffine(img, M, (cols, rows))

    return dst


def randomTranslation(img, width_shift_range, height_shift_range):
    rw = random.randint(-width_shift_range / 2, width_shift_range / 2)
    rh = random.randint(-height_shift_range / 2, height_shift_range / 2)

    return translateImage(img, rw, rh)


def generateStaticAugmentation(src_dir, dst_dir):
    orig_size = directory_size(dst_dir)
    index = directory_size(dst_dir)
    while directory_size(dst_dir) < directory_size(src_dir):
        r = random.randint(0, orig_size)
        name = "retina" + str(r) + ".jpg"
        img = cv.imread(os.path.join(dst_dir, name))
        augmentImage(img, index, dst_dir)
        index += 1


def super_directory_size(directory):
    total_size = 0
    for dir in os.listdir(directory):
        total_size += directory_size(os.path.join(directory, dir))

    return total_size


def directory_size(directory):
    return len(os.listdir(directory))


def rename_files(src_dir):
    i = 0
    for file in os.listdir(src_dir):
        name = "retina" + str(i) + ".jpg"
        orig_name = os.path.join(src_dir, file)
        dst = os.path.join(src_dir, name)
        os.rename(src=orig_name, dst=dst)
        i += 1


def threshold_all_images(source_directory, destination_directory, threshold):
    for img in os.listdir(source_directory):
        img_name = os.path.join(source_directory, img)
        img2 = cv.imread(img_name)
        retval, dst = cv.threshold(img2, threshold, 255, cv.THRESH_BINARY)
        name = os.path.join(destination_directory, img)
        cv.imwrite(name, dst)
