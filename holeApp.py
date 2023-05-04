import numpy as np
import cv2
import math
import random
from matplotlib import pyplot as plt

def rescaleImage(img, scale):
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)

    dimension = (width, height)

    return cv2.resize(img, dimension, interpolation=cv2.INTER_AREA) 

def avg_circles(circles, b):
    avg_x=0
    avg_y=0
    avg_r=0
    for i in range(b):
        #optional - average for multiple circles (can happen when a gauge is at a slight angle)
        avg_x = avg_x + circles[0][i][0]
        avg_y = avg_y + circles[0][i][1]
        avg_r = avg_r + circles[0][i][2]
    avg_x = int(avg_x/(b))
    avg_y = int(avg_y/(b))
    avg_r = int(avg_r/(b))
    return avg_x, avg_y, avg_r

image = cv2.imread('S00001_0.png')
imageRGB = rescaleImage(image, 0.28)
imgGray = cv2.cvtColor(imageRGB, cv2.COLOR_BGR2GRAY)
height, width = imageRGB.shape[:2]
# print(imageRGB.shape[:2])
# blur = cv2.GaussianBlur(imgGray, (3, 3), cv2.BORDER_DEFAULT)
edges = cv2.Laplacian(imgGray, cv2.CV_64F, ksize=3)
print(edges)
cv2.imshow('imgColor', edges)
cv2.waitKey(0)  

# blur = cv2.GaussianBlur(imgGray, (3, 3), cv2.BORDER_DEFAULT)
# canny = cv2.Canny(blur, 125, 175)
# print(canny)
# imgColor = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
# gray = cv2.cvtColor(imgColor, cv2.COLOR_BGR2GRAY)
# circles = cv2.HoughCircles(imgGray, cv2.HOUGH_GRADIENT, 1.5, 100, np.array([]), 50, 30, int(height * 0.45), int(height*0.68))
# # print(circles[0][0][0], circles[0][0][1])
# # print(circles)
# cases = 35 - 35
# # a, b, c = circles.shape
# # x,y,r = avg_circles(circles, b)
# # print(x, y, r)random.randint(0, 254)
# x = int(circles[0][0][0])
# y = int(circles[0][0][1])
# r = int(circles[0][0][2])
# # cv2.circle(imageRGB, (x, y), 254 - cases, (0, 154, 255), 1, cv2.LINE_AA) # for 
# cv2.circle(imageRGB, (x, y),r, (0, 154, 255), 1, cv2.LINE_AA)
# # grayCanny = cv2.cvtColor(imgColor, cv2.COLOR_BGR2GRAY)
# cv2.imshow('imgColor', imageRGB)
# cv2.waitKey(0)  
# # cv2.imshow('imgGray', grayCanny)
