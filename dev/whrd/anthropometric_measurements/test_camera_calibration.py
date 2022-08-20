# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import glob

# Path with images
path = "/Users/davidshatwell/dev/whrd/Chessboard/*.png"

# Defining the dimensions of checkerboard
CHECKERBOARD = (7,7)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Creating vector to store vectors of 3D points for each checkerboard image
objpoints = []
# Creating vector to store vectors of 2D points for each checkerboard image
imgpoints = []

# Defining the world coordinates for 3D points
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None

# Extracting path of individual image stored in a given directory
images = glob.glob(path)
for fname in images:
    img = cv2.imread(fname)
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    # Find the chess board corners
    # If desired number of corners are found in the image then ret = true
    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH + 
             cv2.CALIB_CB_FAST_CHECK + 
             cv2.CALIB_CB_NORMALIZE_IMAGE)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, flags)
    
    # If desired number of corner are detected, we refine the pixel 
    # coordinates and display them on the images of checker board
    if ret:
        objpoints.append(objp)
        
        # refining pixel coordinates for given 2d points.
        corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
        imgpoints.append(corners2)
        
        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        
    # cv2.imshow('img',img)
    # cv2.waitKey(0)

# cv2.destroyAllWindows()

h,w = img.shape[:2]

# Performing camera calibration by passing the value of known 3D points 
# (objpoints) and corresponding pixel coordinates of the detected corners 
# (imgpoints)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("Camera matrix : \n")
print(mtx)
print("dist : \n")
print(dist)
print("rvecs : \n")
print(rvecs)
print("tvecs : \n")
print(tvecs)

# Undistort image
path = "/Users/davidshatwell/dev/whrd/AnthopometricMeasurements/images/inferior_amayatest1.png"
test_image = cv2.imread(path)
test_image = cv2.rotate(test_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
dst = cv2.remap(test_image, mapx, mapy, cv2.INTER_LINEAR)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]

# Show undistorted image using OpenCV and MATLAB

plt.figure()
plt.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
plt.show()

plt.figure()
plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
plt.show()

path = "/Users/davidshatwell/dev/whrd/AnthopometricMeasurements/images/inferior_amayatest1_corrected.png"
matlab_image = cv2.imread(path)

plt.figure()
plt.imshow(cv2.cvtColor(matlab_image, cv2.COLOR_BGR2RGB))
plt.show()

