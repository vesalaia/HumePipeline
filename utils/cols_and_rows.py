"""
Column and row detection based on Hough Transformation
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import hough_line, hough_line_peaks
from skimage.transform import rotate
from skimage.feature import canny
from skimage.io import imread
from skimage.color import rgb2gray

def detectColumns(img_path, alpha, beta, graph=True):
    #    image = rgb2gray(imread(img_path))
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.convertScaleAbs(gray, alpha, beta)
    image = cv2.addWeighted(image, alpha, np.zeros(image.shape, image.dtype), 0, beta)
    #    image = ~image
    height, width = image.shape

    edges = canny(image)
    # Classic straight-line Hough transform
    tested_angles = np.deg2rad(np.arange(170.0, 190.0))
    h, theta, d = hough_line(edges, theta=tested_angles)

    #  Generating figure 1
    if graph:
        fig, axes = plt.subplots(1, 2, figsize=(20, 20))
        ax = axes.ravel()

        ax[0].imshow(image, cmap="gray")
        ax[0].set_title('Input image')
        ax[0].set_axis_off()

        ax[1].imshow(edges, cmap="gray")
    origin = np.array((0, image.shape[1]))
    #    print(origin)
    #    print(edges.shape)
    borders = []
    borders.append([0, 0, 0, height])
    borders.append([width, 0, width, height])
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):

        y0 = (dist - origin[0] * np.cos(angle)) / np.sin(angle)
        y1 = (dist - origin[1] * np.cos(angle)) / np.sin(angle)
        x0 = origin[0]
        x1 = origin[1]
        A = (0, 0)
        B = (width, 0)
        C = (x0, y0)
        D = (x1, y1)
        E = (0, height)
        F = (width, height)
        x_upper, y_upper = [int(x) for x in line_intersection((A, B), (C, D))]
        x_lower, y_lower = [int(x) for x in line_intersection((C, D), (E, F))]
        borders.append([x_upper, y_upper, x_lower, y_lower])
        if graph: plt.plot(origin, (y0, y1), "r")
    if graph:
        ax[1].set_xlim(origin)
        ax[1].set_ylim((edges.shape[0], 0))
    return borders


def detectRows(img_path, alpha, beta, graph=True):
    #    image = rgb2gray(imread(img_path))
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.convertScaleAbs(gray, alpha, beta)
    image = cv2.addWeighted(image, alpha, np.zeros(image.shape, image.dtype), 0, beta)
    #    image = ~image
    height, width = image.shape

    edges = canny(image)
    # Classic straight-line Hough transform
    tested_angles = np.deg2rad(np.arange(85.0, 95.0))
    h, theta, d = hough_line(edges, theta=tested_angles)

    #  Generating figure 1
    if graph:
        fig, axes = plt.subplots(1, 2, figsize=(20, 20))
        ax = axes.ravel()

        ax[0].imshow(image, cmap="gray")
        ax[0].set_title('Input image')
        ax[0].set_axis_off()

        ax[1].imshow(edges, cmap="gray")
    origin = np.array((0, image.shape[1]))
    #    print(origin)
    #    print(edges.shape)
    borders = []
    borders.append([0, 0, width, 0])
    borders.append([0, height, width, height])
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):

        y0 = (dist - origin[0] * np.cos(angle)) / np.sin(angle)
        y1 = (dist - origin[1] * np.cos(angle)) / np.sin(angle)
        x0 = origin[0]
        x1 = origin[1]
        A = (0, 0)
        B = (0, height)
        C = (x0, y0)
        D = (x1, y1)
        E = (width, 0)
        F = (width, height)
        x_upper, y_upper = [int(x) for x in line_intersection((A, B), (C, D))]
        x_lower, y_lower = [int(x) for x in line_intersection((C, D), (E, F))]
        borders.append([x_upper, y_upper, x_lower, y_lower])
        if graph: plt.plot(origin, (y0, y1), "r")
    if graph:
        ax[1].set_xlim(origin)
        ax[1].set_ylim((edges.shape[0], 0))
    return borders