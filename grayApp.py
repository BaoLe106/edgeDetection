import numpy as np
import cv2
from matplotlib import pyplot as plt

def rescaleImage(img):
    scale = 0.2 
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)

    dimension = (width, height)

    return cv2.resize(img, dimension, interpolation=cv2.INTER_AREA) 

image = cv2.imread('10k/S00028_1.png')

imgTemp = rescaleImage(image)
img = cv2.cvtColor(imgTemp, cv2.COLOR_BGR2GRAY)
# height, width = imgTemp.shape[:-1] 

t = cv2.imread('10k/S00028_object1.png')
templateTemp = rescaleImage(t)
template = cv2.cvtColor(templateTemp, cv2.COLOR_BGR2GRAY)
h, w = templateTemp.shape[:-1]

res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
print(res)
threshold = .6
loc = np.where(res >= threshold)
for pt in zip(*loc[::-1]):  # Switch columns and rows
    cv2.rectangle(imgTemp, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 1)


cv2.imshow('result', imgTemp)
cv2.waitKey(0)