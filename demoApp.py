import numpy as np
import cv2
import math
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

def calvec(x1, y1, x2, y2):
    v = []
    v.append(x2 - x1)
    v.append(y2 - y1)
    return v
    
def calcos(x1, y1, x2, y2):
    a = x1*x2 + y1*y2
    b = np.sqrt(x1*x1 + y1*y1) * np.sqrt(x2*x2 + y2*y2)
    if (a > b):
        return 0
    else:
        return (math.acos(a / b) / np.pi) * 180
    

MIN_MATCH_COUNT = 4


template = cv2.imread('10k/s.png')
# template = cv2.imread('10k/sss.png')
templateRGB = rescaleImage(template, 0.6)
templateGray = cv2.cvtColor(templateRGB, cv2.COLOR_BGR2GRAY)

image = cv2.imread('10k/S00043_0.png')
imageRGB = rescaleImage(image, 0.2)
imgGray = cv2.cvtColor(imageRGB, cv2.COLOR_BGR2GRAY)
height, width = imageRGB.shape[:2]
# cv2.imshow('img', imageRGB)
# cv2.waitKey(0)
# blur = cv2.GaussianBlur(imgGray, (3, 3), cv2.BORDER_DEFAULT)
# imgCanny = cv2.Canny(blur, 125, 175)
# cv2.imshow('img', imgCanny)
# cv2.waitKey(0)

blur = cv2.GaussianBlur(imgGray, (5, 5), cv2.BORDER_DEFAULT)
canny = cv2.Canny(blur, 125, 175)

# Convert the gray scaled image to BGR image
imgColor = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
grayCanny = cv2.cvtColor(imgColor, cv2.COLOR_BGR2GRAY)

circles = cv2.HoughCircles(imgGray, cv2.HOUGH_GRADIENT, 1, 20, np.array([]), 150, 50, int(height*0.35), int(height*0.48)) #find all the circles in the image
a, b, c = circles.shape
x,y,r = avg_circles(circles, b) #calculate the center and radius of the circle
cv2.circle(imageRGB, (x, y), r, (0, 255, 0), 3, cv2.LINE_AA)  # draw circle
cv2.circle(imageRGB, (x, y), 2, (0, 255, 0), 2, cv2.LINE_AA)  # draw center of the circle


# Initiate SIFT detector
sift = cv2.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(templateGray, None)
kp2, des2 = sift.detectAndCompute(imgGray, None)

# print(kp1, des1, kp2, des2)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1,des2,k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

if len(good) >= MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    h,w = templateRGB.shape[:-1]
    print(h, w)
    # ptsTest = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ])
    # print(ptsTest)
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
    # print(dst[2][0][0], dst[2][0][1])
    dst = np.int32(dst)
    centerX = (dst[0][0][0] + dst[2][0][0]) // 2
    centerY = (dst[0][0][1] + dst[2][0][1]) // 2
    # print(np.int32(dst))
    print(dst)
    img2 = cv2.polylines(imageRGB,[np.int32(dst)],True,255,3, cv2.LINE_AA)

else:
    # print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
    print("Not enough matches are found")
    matchesMask = None

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

img3 = cv2.drawMatches(templateRGB, kp1, imageRGB, kp2, good, None, **draw_params)

vec = calvec(x, y, centerX, centerY)
deg = calcos(vec[0], vec[1], 500, 0)
print(deg)
plt.imshow(img3, 'gray'),plt.show()
