import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def remove_clumps(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = cv2.findContours(morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    min_area = 1000
    clump_regions = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            clump_regions.append(contour)

    mask = np.zeros_like(image, dtype=np.uint8)
    cv2.drawContours(mask, clump_regions, -1, (255, 255, 255), thickness=cv2.FILLED)

    result = cv2.bitwise_and(image, mask)
    gray_result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    mask = gray_result == 0
    image[mask] = [255, 255, 255]

    output_path = "./static/uploads/no_clumps.png"

    cv2.imwrite(output_path, image)

    return output_path

def black_and_white(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    output_path = image_path[:-4] + "_black_and_white.png"

    cv2.imwrite(output_path, binary_image)

    return output_path

def calculate_areas(pixels_to_unit):
    image = cv2.imread('static/uploads/no_clumps.png')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)

    areas = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        areas.append(int(area) / int(pixels_to_unit))

    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    labeled_img[label_hue == 0] = 0

    # cv2.imshow('Edges without Clumps', thresh)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return areas

def find_scale(image):
    pixels_per_cm = 0
    return pixels_per_cm



# no_clumps = find_clumps(image)
# areas = calculate_areas(no_clumps)

# cv2.imshow('Edges without Clumps', no_clumps)
# cv2.waitKey(0)
# cv2.destroyAllWindows()