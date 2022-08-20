#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 12:02:32 2021

@author: David Shatwell
@company: Work & Health
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt


def plot(image, color, sz=(10,10)):
    plt.figure(figsize=sz)
    if color == 'c':
        plt.imshow(image)
        plt.show()
    elif color == 'm':
        plt.imshow(image, cmap='gray')
        plt.show()
    else:
        print("Choose 'c' for color or 'm' for grayscale")
    


class PoseModel:
    """
    Class used to detect the pose of one person in a color image

    Attributes
    ----------
    root : str
        Root directory containing the pose models
    num_points : int
        Number of points detected by the deep learning model
    net : cv2.dnn
        Deep neural network used to estimate the pose

    Methods
    -------
    detect_points(image, threshold=0.2)
        Detects the pose of a person in a RGB image
    mark_image(image)
        Draws red circles around the pose's keypoints
    """
    
    # Path to load weights
    root = './models'
    
    
    def __init__(self, mode='coco'):
        """
        Parameters
        ----------
        mode : str
            Name of the deep learning model
        """
        
        proto_file = f'{self.root}/{mode}/pose_deploy_linevec.prototxt'
        
        if mode == 'coco':
            weights_file = f'{self.root}/{mode}/pose_iter_440000.caffemodel'
            self.num_points = 18
        elif mode == 'mpi':
            weights_file = f'{self.root}/{mode}/pose_iter_160000.caffemodel'
            self.num_points = 15
        else:
            weights_file, self.num_points = None
            
        self.net = cv2.dnn.readNetFromCaffe(proto_file, weights_file)
        
    def detect_points(self, image, threshold=0.2):
        """ Detects the pose of a person in a RGB image
        
        Parameters
        ----------
        image : numpy.array
            BGR image of a person.
        threshold : double, optional
            Threshold used to detect the exact location of the pose points, 
            given the confidence maps returned by the neural network. The 
            default is 0.2.
            
        Returns
        -------
        points : list of tuples of ints
            List with coordinates of the pose's keypoints
        """
        
        in_width = 368
        in_height = 368
        in_blob = cv2.dnn.blobFromImage(image, 1.0 / 255, 
                                        (in_width, in_height), (0, 0, 0), 
                                        swapRB=False, crop=False)
        self.net.setInput(in_blob)
        maps = self.net.forward()

        # Empty list to store the detected keypoints
        self.points = []

        for i in range(self.num_points):
            # Confidence map corresponding to body's part
            prob_map = maps[0, i, :, :]

            # Finf global masima of the probmap
            _, prob, _, point = cv2.minMaxLoc(prob_map)

            # Scale the point to fit on the original image
            x = (image.shape[1] * point[0]) / maps.shape[3]
            y = (image.shape[0] * point[1]) / maps.shape[2]

            if prob > threshold : 
                # Add the point to the list if the probability is greater than 
                # the threshold
                self.points.append((int(x), int(y)))
            else:
                self.points.append(None)
                
        return self.points
    
    def mark_image(self, image):
        """ Draws red circles around the pose's keypoints
        
        Parameters
        ----------
        image : numpy.array
            BGR image of a person.

        Returns
        -------
        marked : numpy.array
            Marked BGR image.
        """
        
        marked = np.copy(image)
        for i in range(self.num_points):
            if self.points[i] is not None:
                cv2.circle(marked, (int(self.points[i][0]), 
                                    int(self.points[i][1])), 
                           8, (0, 255, 255), thickness=-1, 
                           lineType=cv2.FILLED)
                cv2.putText(marked, f'{i}', (int(self.points[i][0]), 
                                             int(self.points[i][1])), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, 
                            lineType=cv2.LINE_AA)
        return marked
    

class AnthropometricModel:
    """
    Class used to estimate the perimter of the anthropometric measurements of 
    a person using a color image
    
    The anthropometric measurements are:
        1. Neck
        2. Torax
        3. Abdomen
        4. Waist
        5. Hips
    
    The units of the perimeters are given in centimeters (cm).

    Attributes
    ----------
    image : numpy.array
        BGR image (1280x720x3) of a person.
    perimeters : dictionary of floats
        Estimated perimeter of the anthropometric measurements.

    Methods
    -------
    set_model_inputs(front_up, front_down, lateral)
        Store the necesary images used to detect the neck, torax, abdomen, 
        waist and hips of the person
    estimate_perimeters()
        Estimates the anthropometric measurements
    display()
        Display projection measurements used to estimate the perimeters
    """
    
    def __init__(self):
        """Loads MPI and COCO models"""
        
        # Init detection models
        self._mpi = PoseModel("mpi")
        self._coco = PoseModel("coco")
        self._px2cm = np.genfromtxt('distorsion.csv', delimiter=',')
        self.verbose = False
        self.debug = {
            "find_keypoints": False,
            "validate": False,
            "segment": False,
            "find_peaks": False,
            "estimate_neck_perimeter": False,
            "estimate_torax_perimeter": False,
            "estimate_abdomen_perimeter": False,
            "estimate_hips_perimeter": False,
            "estimate_wasit_perimeter": False,
        }
        
        flat_mm_per_px = 35/39
        mm_per_px_correction = 1/1.17471056
        self.mm_per_px = flat_mm_per_px * mm_per_px_correction
        
        self.projections_px = {}
        self.projections_mm = {}
        
    def set_debug_flags(self, debug_flags):
        self.debug = debug_flags
        if debug_flags:
            print()
    
    def set_model_inputs(self, image, age, weight, height):
        """ Store the necesary images and data used to detect the neck, torax, 
        abdomen, waist and hip of the person
        
        Parameters
        ----------
        image : numpy.array
            Frontal BGR image (1280x720x3) of person.
        """
        
        # Store images
        self.image = image
        self.marked_image = image
        self.age = age
        self.weight = weight
        self.height = height
        
    def _find_keypoints(self):
        """ Detects the X and Y coordinates for points of interest

        Returns
        -------
        keypoints : dict of tuple of ints
            Dictionay of the coordinates of the anthropometry keypoints with 
            the following keys:
                - neck
                - torax
                - abdomen
                - waist
                - hips
            
            All keypoints use matrix convention (x: height, y: width)
        """
        
        if self.debug["find_keypoints"]:
            print()
            print("PHASE 1: FIND KEYPOINTS")
            print("-----------------------")
        
        # Generate body model points for image with arms up
        mpi = self._mpi.detect_points(self.image, threshold=0.10)
        coco = self._coco.detect_points(self.image, threshold=0.10)

        if self.debug["find_keypoints"]:
            num_mpi_points = sum(point is not None for point in mpi)
            num_coco_points = sum(point is not None for point in coco)
            print(f"MPI: found {num_mpi_points}/{len(mpi)} points")
            print(f"COCO: found {num_coco_points}/{len(coco)} points")
            marked_mpi_image = self._mpi.mark_image(self.image)
            marked_coco_image = self._coco.mark_image(self.image)
            f, ax = plt.subplots(1,2)
            ax[0].imshow(cv2.cvtColor(marked_mpi_image, cv2.COLOR_BGR2RGB))
            ax[0].title.set_text('MPI')
            ax[1].imshow(cv2.cvtColor(marked_coco_image, cv2.COLOR_BGR2RGB))
            ax[1].title.set_text('COCO')
        
        # Validate keypoints
        pose = self._validate(mpi, coco)
        
        # Find neck, torax and abdomen coordinates
        self.keypoints = {
            'neck': pose['n'],
            'torax': (int(0.35*pose['rs'][0]+0.35*pose['ls'][0]+0.3*pose['c'][0]), 
                      int(pose['c'][1])),
            'abdomen': (None, None),
            'waist': (int(0.25*pose['rh'][0]+0.25*pose['lh'][0]+0.5*pose['c'][0]), 
                      int(pose['c'][1])),
            'hips': (None, None),
        }

        return self.keypoints
    
    def _mark_image(self, image):
        """ Marks the location of the keypoints
        
        Parameters
        ----------
        image : numpy.array
            BGR image of the person

        Returns
        -------
        marked : numpy.array
            BGR image with circles centered on the keypoints
        """

        marked = np.copy(image)
                
        for key in self.keypoints.keys():
            c = self.keypoints[key]
            cv2.circle(marked, center=(c[1], c[0]), radius=8, 
                       color=(0,255,255), thickness=-1, lineType=cv2.FILLED)
        return marked
    
    def _validate(self, mpi, coco):
        """Validates the pose coordinates given by the MPI and COCO models

        Parameters
        ----------
        mpi : list of tuples of ints
            List of points according to the MPI pose model
        coco : list of tuples of ints
            List of points according to the COCO pose model

        Returns
        -------
        val_points : dict
            Validated pose points used to find the neck, torax and abdomen
            Consider the folloing keys:
                - rs: right shoulder
                - ls: left shoulder
                - rh: right hip
                - lh: left hip
                - c:  center
                - n:  neck
        """
        
        if self.debug["validate"]:
            print()
            print("PHASE 2: VALIDATE KEYPOINTS")
            print("---------------------------")
        
        # Init point list
        points = []

        # Define dict with body parts
        body_points = {
            2: "right shoulder",
            5: "left shoulder",
            8: "right hip",
            11: "left hip"
        }

        # Loop through body parts and validate coordinates
        errors = np.ones((len(body_points), 1), dtype=bool)
        for i, key in enumerate(body_points.keys()):
            if coco[key] is not None:
                points.append((coco[key][1], coco[key][0]))
            elif mpi[key] is not None:
                points.append((mpi[key][1], mpi[key][0]))
            else:
                errors[i] = False
                points.append(None)
                if self.debug["validate"]:
                    print(f"Could not find {body_points[key]} in MPI or COCO models")
        
        # Missing body part flags
        hip_missing = ((errors[0] and errors[1] 
                        and errors[2] and not errors[3]) or 
                       (errors[0] and errors[1] 
                        and not errors[2] and errors[3]))
        
        shoulder_missing = ((errors[0] and not errors[1] and 
                             errors[2] and errors[3]) or 
                            (not errors[0] and errors[1] and 
                             errors[2] and errors[3]))

        hip_and_shoulder_missing = ((errors[0] and not errors[1] 
                                     and not errors[2] and errors[3]) or 
                                    (not errors[0] and errors[1] 
                                     and errors[2] and not errors[3]))
        
        # Validate center
        # Body center point (equivalent to mpi[14])
        error_flag = False
        if np.sum(errors) == 4:
            x_mean = int((points[0][0]+points[1][0]+points[2][0]+ 
                          points[3][0])/4)
            y_mean = int((points[0][1]+points[1][1]+points[2][1]+
                          points[3][1])/4)
            points.append((x_mean, y_mean))
            if self.debug["validate"]:
                print("No critical points missing , calculating body center using both shoulders and hips")

        elif hip_missing:
            if points[2] is not None: # If right hip is not missing
                x_mean = int((points[0][0]+points[1][0]+2*points[2][0])/4)
                points[3] = (points[2][0], points[1][1])
                if self.debug["validate"]:
                    print("Left hip is missing, using right hip and both shoulders to estimate body center")
            else:                     # If left hip is not missing
                x_mean = int((points[0][0]+points[1][0]+2*points[3][0])/4)
                points[2] = (points[3][0], points[0][1])
                if self.debug["validate"]:
                    print("Right hip is missing, using left hip and both shoulders to estimate body center")
            y_mean = int((points[0][1]+points[1][1])/2)
            points.append((x_mean, y_mean))

        elif shoulder_missing:
            if points[0] is not None:
                x_mean = int((2*points[0][0]+points[2][0]+points[3][0])/4)
                points[1] = (points[0][0], points[3][1])
                if self.debug["validate"]:
                    print("Left shoulder is missing, using right shoulder and both hips to estimate body center")
            else:
                x_mean = int((2*points[1][0]+points[2][0]+points[3][0])/4)
                points[0] = (points[1][0], points[2][1])
                if self.debug["validate"]:
                    print("Right shoulder is missing, using left shoulder and both hips to estimate body center")
            y_mean = int((points[2][1]+points[3][1])/2)
            points.append((x_mean, y_mean))

        elif hip_and_shoulder_missing:
            if points[0] is not None and points[3] is not None:
                x_mean = int((points[0][0]+points[3][0])/2)
                if self.debug["validate"]:
                    print("Left shoulder and right hip are missing, using right shoulder and left hip to estimate body center")
            else:
                x_mean = int((points[1][0]+points[2][0])/2)
                if self.debug["validate"]:
                    print("Right shoulder and left hip are missing, using left shoulder and right hip to estimate body center")
            y_mean = int(((points[0] is not None)*points[0][1]+
                          (points[1] is not None)*points[1][1]+
                          (points[2] is not None)*points[2][1]+
                          (points[3] is not None)*points[3][1])/2)
            points.append((x_mean, y_mean))

        elif mpi[14] is not None:
            points.append((mpi[14][1], mpi[14][0]))
            if self.debug["validate"]:
                    print("Three or more critical points could not be found. Estimating body center using point 14 of MPI.")

        elif coco[1] is not None:
            if self.debug["validate"]:
                print("Three or more critical points could not be found. Trying to estimate body center using the neck and the point with high frequency change that occurs between the skin and the pants.")
                print("  [WARNING] This estimation may not be accurate")
            start_point = coco[1]
            black_line = np.sum(self.image[start_point[1]:, start_point[0]], axis=1)
            vline = self.image[start_point[1]:, start_point[0], 0]
            dvline = np.gradient(vline)
            sigma = np.std(dvline)
            sigma_point = np.where(np.abs(dvline) >= 5*sigma)[0][0]
            black_point = np.where(black_line < 30)[0][0]
            if black_point < sigma_point:
                end_point = (start_point[0], start_point[1]+black_point+70)
                if self.debug["validate"]:
                    print("  Using black line")
            else:
                end_point = (start_point[0], start_point[1]+sigma_point)
                if self.debug["validate"]:
                    print("  Using high-frequency line")
                
            temp = (int(0.5*(start_point[1]+end_point[1])), start_point[0])
            points.append(temp)
            points[2] = (end_point[1], end_point[0])
            points[3] = (end_point[1], end_point[0])
            
            if self.debug["validate"]:
                n = np.linspace(start_point[1], 
                                self.image.shape[0], 
                                self.image.shape[0] - start_point[1])
                f = plt.figure(figsize=(10,8))
                ax = f.subplots(3,1)
                ax[0].plot(n, vline)
                ax[0].title.set_text('vline')
                ax[0].grid()
                ax[1].plot(n, dvline)
                ax[1].plot(n, 5*sigma*np.ones(dvline.shape))
                ax[1].title.set_text('vline derivative')
                ax[1].grid()
                ax[2].plot(n, black_line)
                ax[2].plot(n, 30*np.ones(black_line.shape))
                ax[2].title.set_text('black line')
                ax[2].grid()
                    
        else:
            points.append((0, 0))
            error_flag = True
            print("[ERROR] Body center not found. Algorithm cannot determine the anthopometric measurements of the person.")
        
        # Validate neck
        # Image dimensions
        height, width, _ = self.image.shape
        
        # Find neck
        if coco[1] is not None:
            NECK_OFFSET = 80
            points.append((coco[1][1] - NECK_OFFSET, coco[1][0]))
        elif mpi[1] is not None:
            points.append((mpi[1][1], mpi[1][0]))
        else:
            print("[ERROR] Neck not found")
            points.append((0, 0))
            error_flag = True
        
        # Results
        val_points = {
            "rs": points[0],
            "ls": points[1],
            "rh": points[2],
            "lh": points[3],
            "c": points[4],
            "n": points[5],
        }    
        
        if self.debug["validate"] and not error_flag:
            
            # Image to display
            debug_image = self.image.copy()
            
            # Point properties
            radius = 8
            point_color = (0, 255, 255)
            thickness = -1
            
            # Text properties
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 2
            text_color = (0, 0, 255)
            thickness = 3
            
            # Draw points in image
            for key in val_points.keys():
                point = (val_points[key][1], val_points[key][0])
                cv2.circle(debug_image, point, radius, point_color, 
                           thickness=thickness, lineType=cv2.FILLED)
                cv2.putText(debug_image, key, point, font, fontScale, 
                            text_color, thickness=thickness, 
                            lineType=cv2.LINE_AA)
            
            # Display image
            plt.figure()
            plt.imshow(cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB))
            plt.show()
        
        return val_points
    
    def _segment_image(self, T1=0.08, T2=30):
        """ Creates mask with skin pixels
        
        Parameters
        ----------
        image : numpy.array
            BGR image of the person to segment
        T1 : float, optional
            Threshold used for firtst segmentation using HSV color space. The 
            default is 0.08.
        T2 : float, optional
            Threshold used for second segmentation using Mahalabobis distance. 
            The default is 30.

        Returns
        -------
        mask : numpy.array
            Mask with skin pixels.

        """
        
        if self.debug["segment"]:
            print()
            print("PHASE 3: SEGMENT IMAGE")
            print("----------------------")
        
        # FIRST PART
        # Resize image for faster processing
        original_height, original_width, _ = self.image.shape
        target_size = (int(original_width / 4), int(original_height / 4))
        image_rsz = cv2.resize(self.image, target_size)
        
        # RGB to HSV, keep only hue
        hsv = cv2.cvtColor(image_rsz, cv2.COLOR_BGR2HSV)
        hue = (hsv[:,:,0] / 255.0)
        
        # Binary mask
        mask1 = (hue < T1).astype('uint8')
        mask2 = (hue >= (1 - T1)).astype('uint8')
        hmask = cv2.bitwise_or(mask1, mask2)
        
        # Masking
        height, width, depth = image_rsz.shape
        X = np.zeros((np.sum(hmask), depth))
        for i in range(depth):
            X[:,i] = hsv[:,:,i].flatten()[np.argwhere(hmask.flatten())].squeeze()
            
        if self.debug["segment"]:
            # Plot mask using only hue threshold
            f = plt.figure(figsize=(14,6))
            ax = f.subplots(1,4)
            
            ax[0].imshow(mask1, cmap="gray")
            ax[0].title.set_text(f"mask1 = (T1 < {T1})")
            
            ax[1].imshow(mask2, cmap="gray")
            ax[1].title.set_text(f"mask2 = (T1 >= {1-T1})")
            
            ax[2].imshow(hmask, cmap="gray")
            ax[2].title.set_text("mask = mask1 OR mask2")
            
            hue_masked_image = cv2.bitwise_and(image_rsz, image_rsz, mask=hmask)
            ax[3].imshow(cv2.cvtColor(hue_masked_image, cv2.COLOR_BGR2RGB))
            ax[3].title.set_text("Hue-masked image")

        # Center data
        mean = np.mean(X, axis=0)
        Xc = X - mean
        
        # Covariance matrix
        C = np.dot(Xc.T, Xc) / height / width
        
        # Mahalanobis distance
        mean = np.reshape(mean, (1,1,3))
        D = np.zeros((height, width))
        for h in range(height):
            for w in range(width):
                temp = np.reshape(hsv[h,w,:] - mean, (3, 1))
                D[h,w] = np.dot(temp.T, np.dot(np.linalg.inv(C), temp))
                
        # New Mask
        premask = (D < T2).astype('uint8')
        premask = cv2.resize(premask, (original_width, original_height))
        
        # SECOND PART
        # Find contours and keep largest areas
        _, contours, _ = cv2.findContours(premask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        # Find largest contour
        contours = sorted(contours, key=cv2.contourArea)            
        mask = np.zeros_like(premask)
        cv2.drawContours(mask, [contours[-1]], -1, 255, cv2.FILLED, 1) 
        
        if self.debug["segment"]:
            f = plt.figure(figsize=(14,10))
            ax = f.subplots(2,3)
            
            ax[0,0].imshow(hmask, cmap="gray")
            ax[0,0].title.set_text("Hue mask")
            
            ax[0,1].imshow(premask, cmap="gray")
            ax[0,1].title.set_text("Mahalanobis mask")
            
            ax[0,2].imshow(mask, cmap="gray")
            ax[0,2].title.set_text("Largest contour")
            
            ax[1,0].imshow(cv2.cvtColor(hue_masked_image, cv2.COLOR_BGR2RGB))
            ax[1,0].title.set_text("Hue-masked image")
            
            mahalanobis_masked_image = cv2.bitwise_and(self.image, self.image, mask=premask)
            ax[1,1].imshow(cv2.cvtColor(mahalanobis_masked_image, cv2.COLOR_BGR2RGB))
            ax[1,1].title.set_text("Mahalanobis-masked image")
            
            masked_image = cv2.bitwise_and(self.image, self.image, mask=mask)
            ax[1,2].imshow(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))
            ax[1,2].title.set_text("Largest countour-masked image")
        
        return mask
    
    def _find_peaks(self, x, threshold=None):
        if threshold is None:
            threshold = 3*np.std(x)
        
        outliers = np.squeeze(np.argwhere(x > threshold))
        left_outliers = [outlier for outlier in outliers if outlier <= len(x) // 2]
        right_outliers = [outlier for outlier in outliers if outlier > len(x) // 2]
        
        if len(left_outliers) > 0:
            first_peak = left_outliers[0]
            for outlier in left_outliers:
                if np.sum(x[outlier-3:outlier]) == 0:
                    first_peak = outlier
        else:
            return None
        
        if len(right_outliers) > 0:
            last_peak = right_outliers[-1]
            for outlier in np.flip(right_outliers):
                if np.sum(x[outlier+1:outlier+4]) == 0:
                    last_peak = outlier
        else:
            return None
        
        return (first_peak, last_peak)
    
    
    def _mm_per_px(self, x):
        return 1.320765353142718 - 1.516822026614581*10**(-4) * x
    

    def estimate_perimeters(self):
        """ Estimates the neck, torax, abdomen and waist perimeters
        
        Returns
        -------
        perimeters : dictionary of floats
            Dictionary of estimated perimeters with the following keys:
              - 'neck': neck perimeter
              - 'torax': torax perimeter
              - 'abdomen': abdomen perimeter
              - 'waist': waist perimeter
              - 'hips': hips perimeter
              - 'neck_val': 1 if the neck perimeter is within the acceptable 
                            range, else 0
              - 'torax_val': 1 if the torax  perimeter is within the 
                             acceptable range, else 0
              - 'abdomen_val': 1 if the abdomen  perimeter is within the 
                               acceptable range, else 0
              - 'waist_val': 1 if the waist perimeter is within the acceptable
                             range, else 0
              - 'hips_val': 1 if the hips perimeter is within the acceptable 
                            range, else 0
        """
        
        # Find model keypoints
        self._find_keypoints()
        
        # Edge detection
        self._mask = self._segment_image()
        masked = cv2.bitwise_and(self.image, self.image, mask=self._mask)
        
        # Constants
        CORRECTION_OFFSET = 20
        HORIZONTAL_OFFSET = 30
        DIFF_THRESHOLD = -30
        NUM_PIXELS_BELOW_THRESHOLD = 10
        MIN_NUM_PIX = 3
        
        # ------------- 
        #  DETECT NECK 
        # ------------- 
        
        if self.debug["estimate_neck_perimeter"]:
            print()
            print("PHASE 4: DETECT NECK")
            print("--------------------")
        
        # Choose point of interest
        p = self.keypoints['neck']

        H_RANGE = 150   # Horizontal range for ROI
        V_RANGE = 50    # Vertical range for ROI

        roi = self._mask[(p[0]-V_RANGE):(p[0]+V_RANGE), p[1]-H_RANGE:p[1]+H_RANGE]
        
        # MORPHOLOGY
        kernel = np.ones((2,2), np.uint8)
        opening = cv2.morphologyEx(roi.astype(np.uint8), cv2.MORPH_OPEN, 
                                   kernel, iterations=2)
        dilation = cv2.dilate(opening, kernel, iterations=1)
        
        if self.debug["estimate_neck_perimeter"]:
            f = plt.figure(figsize=(10,10))
            ax = f.subplots(1,3)
            
            ax[0].imshow(roi, cmap="gray")
            ax[0].title.set_text("ROI")
            
            ax[1].imshow(opening, cmap="gray")
            ax[1].title.set_text("ROI with opening")
            
            ax[2].imshow(dilation, cmap="gray")
            ax[2].title.set_text("ROI with opening and dilation ")
            
            f.show()
        
        # Find neck in ROI
        lengths = np.zeros((roi.shape[0], 1), dtype=int)
        reg_edges = np.zeros((roi.shape[0], 2), dtype=int)
        for i in range(roi.shape[0]):
            # Find edges
            edges = np.argwhere(dilation[i,:] > 0)

            # Check for valid egdes
            num_left = np.sum(edges < H_RANGE) > 0
            num_right = np.sum(edges > H_RANGE) > 0
            if num_left and num_right:
                lengths[i] = edges[-1,:] - edges[0,:]
                reg_edges[i,0] = edges[0,:]
                reg_edges[i,1] = edges[-1,:]
            else:
                lengths[i] = 2 * H_RANGE + 1
                reg_edges[i,0] = 0
                reg_edges[i,1] = 2 * H_RANGE + 1
                        
        # Find neck length
        imin = np.argmin(lengths)
        
        # Find min neck distance and location
        min_length_rltv_loc = imin  # Relative location
        min_length_loc = p[0] - V_RANGE + min_length_rltv_loc  # Absolute location

        # Modify marked image
        start = reg_edges[min_length_rltv_loc, 0]
        end = reg_edges[min_length_rltv_loc, 1]

        start_point = (self.keypoints['neck'][1] - H_RANGE + start, min_length_loc)
        end_point = (self.keypoints['neck'][1] - H_RANGE + end, min_length_loc)
        color = (0, 255, 255)
        thickness = 2
        self.marked_image = cv2.line(self.marked_image, start_point, 
                                     end_point, color, thickness)
        
        neck_length_px = end - start
        neck_length_mm = np.round(neck_length_px * self.mm_per_px)
        # neck_length_mm = np.round(neck_length_px * self._mm_per_px(min_length_loc))
        
        # Record result
        self.projections_px["neck"] = neck_length_px
        self.projections_mm["neck"] = neck_length_mm
        
        if self.debug["estimate_neck_perimeter"]:
            print("Found neck projection at coordinates:")
            print(f"  P1 = ({start_point[0]},{start_point[1]})")
            print(f"  P2 = ({end_point[0]},{end_point[1]})")
            print(f"Neck projection length (px): {neck_length_px}")
            print(f"Neck projection length (mm): {neck_length_mm}")

        # --------------
        #  DETECT TORAX
        # --------------
        
        if self.debug["estimate_torax_perimeter"]:
            print()
            print("PHASE 4: DETECT TORAX")
            print("---------------------")

        p = self.keypoints['torax'][0]
        
        num_lines = 30
        line_range = np.linspace(p-num_lines//2, p+num_lines//2-1, num_lines).astype(int)
        lengths = np.zeros_like(line_range).astype(float)
        for idx, m in enumerate(line_range):
            valid_line_pixels = np.argwhere(self._mask[m,:])
            start = valid_line_pixels[0][0] + CORRECTION_OFFSET
            end = valid_line_pixels[-1][0] - CORRECTION_OFFSET
        
            line = masked[m,start:end,:].squeeze().astype(float)
            dline = line[1:,:] - line[:-1,:]
            peaks = self._find_peaks(np.abs(dline[:,2]))
            if peaks is None:
                lengths[idx] = np.NAN
            else:
                lengths[idx] = peaks[1] - peaks[0]
        
        # Remove NaN values
        lengths = lengths[~np.isnan(lengths)]
        
        n, bins = np.histogram(lengths, bins=10)
        
        largest_bin_idx = np.argmax(n)
        lower_bound = bins[largest_bin_idx]
        upper_bound = bins[largest_bin_idx + 1]
        
        # Find location of most frequent length
        for m in np.flip(line_range):
            valid_line_pixels = np.argwhere(self._mask[m,:])
            start = valid_line_pixels[0][0] + CORRECTION_OFFSET
            end = valid_line_pixels[-1][0] - CORRECTION_OFFSET
        
            line = masked[m,start:end,:].squeeze().astype(float)
            dline = line[1:,:] - line[:-1,:]
            peaks = self._find_peaks(np.abs(dline[:,2]))
            if peaks is None:
                continue
            length = peaks[1] - peaks[0]
            if (length > lower_bound) and (length < upper_bound):
                break
        
        pt1 = (start + peaks[0], m)
        pt2 = (start + peaks[1], m)
        color = (255,0,0)
        thickness = 5
        self.marked_image = cv2.line(self.marked_image, pt1, pt2, color, 
                                     thickness)
        
        # Adjust torax location
        self.keypoints['torax'] = (m, start + (peaks[0] + peaks[1]) // 2)
        
        torax_length_px = peaks[1] - peaks[0]
        torax_length_mm = np.round(torax_length_px * self.mm_per_px)
        # torax_length_mm = np.round(torax_length_px * self._mm_per_px(m))
        self._mm_per_px(min_length_loc)
        
        # Record result
        self.projections_px["torax"] = torax_length_px
        self.projections_mm["torax"] = torax_length_mm
        
        if self.debug["estimate_torax_perimeter"]:
            print("Found torax projection at coordinates:")
            print(f"  P1 = ({pt1[0]},{pt1[1]})")
            print(f"  P2 = ({pt2[0]},{pt2[1]})")
            print(f"Torax projection length (px): {torax_length_px}")
            print(f"Torax projection length (mm): {torax_length_mm}")
        
        if self.debug["estimate_torax_perimeter"]:
            
            edges_x1 = [peaks[0], peaks[0]]
            edges_x2 = [peaks[1], peaks[1]]
            
            f = plt.figure(figsize=(7,7))
            ax = f.subplots(2,1)
            
            ax[0].plot(line[:,2])  # Plot red line
            edges_y = [np.min(line), np.max(line)]
            ax[0].plot(edges_x1, edges_y, "r--")
            ax[0].plot(edges_x2, edges_y, "r--")
            ax[0].title.set_text("Torax: red color line")
            
            ax[1].plot(dline)  # Plot red line
            edges_y = [np.min(dline), np.max(dline)]
            ax[1].plot(edges_x1, edges_y, "r--")
            ax[1].plot(edges_x2, edges_y, "r--")
            ax[1].title.set_text("Torax: red color line derivative")
            
            f.show()
        
        # --------------
        #  DETECT WAIST
        # --------------
        
        # Point of interest (matrix notation)
        p = self.keypoints['waist']
        
        # Generate ROI from point
        ROI_HEIGHT = 100
        RED_CHANNEL = 0
        xrange = range(p[0] - ROI_HEIGHT // 2, p[0] + ROI_HEIGHT // 2)
        roi = masked[xrange,:,RED_CHANNEL];
        
        # Loop through ROI lines and find navel
        line_range = np.linspace(p[0] - ROI_HEIGHT // 2, 
                                 p[0] + ROI_HEIGHT // 2, 
                                 ROI_HEIGHT).astype(int)
        
        # Find navel (ombligo en español)
        navel_counter = 0

        for m in line_range:
            
            valid_line_pixels = np.argwhere(self._mask[m,:])
            start = valid_line_pixels[0][0] + CORRECTION_OFFSET
            end = valid_line_pixels[-1][0] - CORRECTION_OFFSET
            
            line = masked[m,start:end,:].squeeze().astype(float)
            dline = line[1:,:] - line[:-1,:]
            peaks = self._find_peaks(np.abs(dline[:,2]))
            
            if peaks is None:
                if navel_counter > 0:
                    navel_counter -= 1
                continue
            
            body_line = line[peaks[0]+CORRECTION_OFFSET:peaks[1]-CORRECTION_OFFSET,RED_CHANNEL]
            N = body_line.shape[0]
            peak_center = len(body_line) // 2
            valid_line = body_line[peak_center-HORIZONTAL_OFFSET:peak_center+HORIZONTAL_OFFSET]
            invalid_line = np.concatenate([body_line[0:peak_center-HORIZONTAL_OFFSET],
                                           body_line[peak_center+HORIZONTAL_OFFSET:]])
            invalid_line = np.expand_dims(invalid_line, 1)
            left = np.expand_dims(np.arange(0, peak_center-HORIZONTAL_OFFSET), 1)
            right = np.expand_dims(np.arange(peak_center+HORIZONTAL_OFFSET, N), 1)
            invalid_line_idx = np.concatenate((left, right))
            
            # Regression
            x = invalid_line_idx
            X = np.concatenate((np.ones((x.shape[0],1)), x, np.multiply(x,x)), axis=1)
            y = invalid_line
            beta = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
            
            x = np.expand_dims(np.arange(N), 1)
            Xnew = np.concatenate((np.ones((N, 1)), x, np.multiply(x, x)), axis=1)
            y_hat = np.dot(Xnew, beta)
            y_test = np.expand_dims(body_line, 1)
            
            diff = y_test - y_hat
            pix_below_threshold = diff < DIFF_THRESHOLD
            
            if np.sum(pix_below_threshold) >= NUM_PIXELS_BELOW_THRESHOLD:
                navel_counter += 1
            else:
                if navel_counter > 0:
                    navel_counter -= 1
            
            if navel_counter >= MIN_NUM_PIX:
                break
            
        # Display navel location
        pt1 = (start + peaks[0], m)
        pt2 = (start + peaks[1], m)
        color = (255,0,0)
        thickness = 5
        self.marked_image = cv2.line(self.marked_image, pt1, pt2, color, 
                                     thickness)
        
        # Adjust waist location
        self.keypoints['waist'] = (m, start + (peaks[0] + peaks[1]) // 2)
        
        waist_length_px = peaks[1] - peaks[0]
        waist_length_mm = np.round(waist_length_px * self.mm_per_px)
        # waist_length_mm = np.round(waist_length_px * self._mm_per_px(m))
        
        # Record result
        self.projections_px["waist"] = waist_length_px
        self.projections_mm["waist"] = waist_length_mm
        
        if self.debug["estimate_waist_perimeter"]:
            print("Found waist projection at coordinates:")
            print(f"  P1 = ({pt1[0]},{pt1[1]})")
            print(f"  P2 = ({pt2[0]},{pt2[1]})")
            print(f"Waist projection length (px): {waist_length_px}")
            print(f"Waist projection length (mm): {waist_length_mm}")
        
        
        # ----------------
        #  DETECT ABDOMEN
        # ----------------
        
        if self.debug["estimate_torax_perimeter"]:
            print()
            print("PHASE 6: DETECT ABDOMEN")
            print("-----------------------")
        
        # Point of interest (matrix notation)
        p = int(0.7 * self.keypoints['waist'][0] + 0.3 * self.keypoints['torax'][0])
        
        num_lines = 20
        line_range = np.linspace(p-num_lines//2, p+num_lines//2-1, num_lines).astype(int)
        lengths = np.zeros_like(line_range).astype(float)
        for idx, m in enumerate(line_range):
            valid_line_pixels = np.argwhere(self._mask[m,:])
            start = valid_line_pixels[0][0] + CORRECTION_OFFSET
            end = valid_line_pixels[-1][0] - CORRECTION_OFFSET
        
            line = masked[m,start:end,:].squeeze().astype(float)
            dline = line[1:,:] - line[:-1,:]
            peaks = self._find_peaks(np.abs(dline[:,2]))
            if peaks is None:
                lengths[idx] = np.NAN
            else:
                lengths[idx] = peaks[1] - peaks[0]
        
        # Remove NaN values
        lengths = lengths[~np.isnan(lengths)]
        
        n, bins = np.histogram(lengths, bins=10)
        
        largest_bin_idx = np.argmax(n)
        lower_bound = bins[largest_bin_idx]
        upper_bound = bins[largest_bin_idx + 1]
        
        # Find location of most frequent length
        for m in np.flip(line_range):
            valid_line_pixels = np.argwhere(self._mask[m,:])
            start = valid_line_pixels[0][0] + CORRECTION_OFFSET
            end = valid_line_pixels[-1][0] - CORRECTION_OFFSET
        
            line = masked[m,start:end,:].squeeze().astype(float)
            dline = line[1:,:] - line[:-1,:]
            peaks = self._find_peaks(np.abs(dline[:,2]))
            if peaks is None:
                continue
            length = peaks[1] - peaks[0]
            if (length > lower_bound) and (length < upper_bound):
                print(f"Location: {m}")
                break
        
        pt1 = (start + peaks[0], m)
        pt2 = (start + peaks[1], m)
        color = (255,0,0)
        thickness = 5
        self.marked_image = cv2.line(self.marked_image, pt1, pt2, color, 
                                     thickness)
        
        # Adjust abdomen location
        self.keypoints['abdomen'] = (m, start + (peaks[0] + peaks[1]) // 2)
        
        abdomen_length_px = peaks[1] - peaks[0]
        abdomen_length_mm = np.round(abdomen_length_px * self.mm_per_px)
        # abdomen_length_mm = np.round(abdomen_length_px * self._mm_per_px(m))
        
        # Record result
        self.projections_px["abdomen"] = abdomen_length_px
        self.projections_mm["abdomen"] = abdomen_length_mm
        
        if self.debug["estimate_abdomen_perimeter"]:
            print("Found abdomen projection at coordinates:")
            print(f"  P1 = ({pt1[0]},{pt1[1]})")
            print(f"  P2 = ({pt2[0]},{pt2[1]})")
            print(f"Abdomen projection length (px): {abdomen_length_px}")
            print(f"Abdomen projection length (mm): {abdomen_length_mm}")
        
        if self.debug["estimate_abdomen_perimeter"]:
            
            edges_x1 = [peaks[0], peaks[0]]
            edges_x2 = [peaks[1], peaks[1]]
            
            f = plt.figure(figsize=(7,7))
            ax = f.subplots(2,1)
            
            ax[0].plot(line[:,2])  # Plot red line
            edges_y = [np.min(line), np.max(line)]
            ax[0].plot(edges_x1, edges_y, "r--")
            ax[0].plot(edges_x2, edges_y, "r--")
            ax[0].title.set_text("Abdomen: red color line")
            
            ax[1].plot(dline)  # Plot red line
            edges_y = [np.min(dline), np.max(dline)]
            ax[1].plot(edges_x1, edges_y, "r--")
            ax[1].plot(edges_x2, edges_y, "r--")
            ax[1].title.set_text("Abdomen: red color line derivative")
            
            f.show()  

        # -------------
        #  DETECT HIPS
        # -------------
        
        if self.debug["estimate_hips_perimeter"]:
            print()
            print("PHASE 6: DETECT HIPS")
            print("--------------------")
        
        abdomen = self.keypoints['abdomen']
        
        tolerance = 40  # pixels
        height = 400  # pixels
        
        px = abdomen[0]
        py1 = (abdomen[1] - abdomen_length_px // 2 - tolerance).astype(int)
        py2 = (abdomen[1] + abdomen_length_px // 2 + tolerance).astype(int)
        
        # Create ROI
        roi = self._mask[px:px+height, py1:py2]
        
        # Find convex hull
        _, contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        hull = []
        for i in range(len(contours)):
            hull.append(cv2.convexHull(contours[i], False))         
        mask = np.zeros_like(roi)
        hull = cv2.drawContours(mask, hull, -1, (1,1,1), -1)
        
        if self.debug["estimate_hips_perimeter"]:
            plt.figure()
            plt.imshow(hull, cmap="gray")
            plt.title("ROI used to find pants")
            plt.show()
        
        # Count number of rows in ROI
        num_rows = mask.shape[0]
        
        # Loop through rows and find extremes
        length = []
        start = []
        end = []
        for i in range(num_rows):
            row = mask[i,:]
            
            # Find start and end of mask
            mask_idx = np.argwhere(row > 0)
            if sum(mask_idx) > 0:
                start.append(mask_idx[0])
                end.append(mask_idx[-1])
            
                # Find length in pixels
                length.append(end[-1] - start[-1])
        
        # Convert length list to numpy array
        length = np.array(length)
        
        # Moving average
        w = 5
        smoothed_length = np.convolve(length.squeeze(), np.ones((w,)), 'valid') / w
        
        # Find derivative of lengths
        dlength = np.abs(np.gradient(smoothed_length, axis=0))
        
        # Find point where gradients diverge
        CMA = 0  # Cummulative moving average
        for n, x in enumerate(dlength):
            CMA = (x + n * CMA) / (n + 1)
            if x > 2 * CMA:
                break
        n = n - 1
            
        # Find hips location
        self.keypoints['hips'] = (n + px, py1 + (start[n] + end[n]) // 2)
        
        # Find lengths in px and mm
        hips_length_px = (end[n] - start[n])[0]
        hips_length_mm = np.round(hips_length_px * self.mm_per_px)
        # hips_length_mm = np.round(hips_length_px * self._mm_per_px(n + px))
        
        # Record result
        self.projections_px["hips"] = hips_length_px
        self.projections_mm["hips"] = hips_length_mm
        
        pt1 = (py1 + start[n], n + px)
        pt2 = (py1 + end[n], n + px)
        color = (255,0,0)
        thickness = 5
        self.marked_image = cv2.line(self.marked_image, pt1, pt2, color, 
                                     thickness)
        
        if self.debug["estimate_hips_perimeter"]:
            plt.figure()
            plt.plot(length)
            plt.title("Lengths")
            plt.show()
            
            plt.figure()
            plt.plot(dlength)
            plt.title("Lengths derivative")
            plt.show()
        
        
        # ----------------------
        #  SAVE POINTS TO IMAGE
        # ----------------------
        
        # Mark keypoints
        for key in self.keypoints.keys():
            p = self.keypoints[key]
            
            cv2.circle(self.marked_image, (p[1], p[0]), 8, (0, 255, 255), 
                       thickness=-1, lineType=cv2.FILLED)
            
        # Code to show final keypoints
        if self.debug["final_keypoints"]:
            
            # Image to display
            debug_image = self.image.copy()
            
            # Point properties
            radius = 8
            point_color = (0, 255, 255)
            thickness = -1
            
            # Text properties
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 2
            text_color = (0, 0, 255)
            thickness = 3
            
            for key in self.keypoints.keys():
                point = (self.keypoints[key][1], self.keypoints[key][0])
                cv2.circle(debug_image, point, radius, point_color, 
                           thickness=thickness, lineType=cv2.FILLED)
                cv2.putText(debug_image, key, point, font, fontScale, 
                            text_color, thickness=thickness, 
                            lineType=cv2.LINE_AA)
            
            # Display image
            plt.figure()
            plt.imshow(cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB))
            plt.show()
        
        # TODO: Train regression

        # ----------------
        #  RECORD RESULTS
        # ----------------

        # valid_neck = int(
        #     neck_length > 8 and neck_length < 18
        # )

        # valid_torax = int(
        #     torax_length > 28 and torax_length < 60        
        # )

        # valid_abdomen = int(
        #     abdomen_length > 28 and abdomen_length < 60
        # )
        
        # valid_waist = int(
        #     waist_length > 28 and waist_length < 60
        # )

        # # Linear regression
        # neck_coeffs = np.array([[6.477987966405332, 0.003458641757394,
        #                          -0.096883095200424, 10.519134173476433,
        #                          1.078880416875089, -1.204495344242136,
        #                          1.568410840692458, 0]])
        
        # torax_coeffs = np.array([[23.197497327561805, 0.092502873732669,
        #                           0.377559157827259, 10.540784650251016,
        #                           4.050238106122616, 0.542385550050988,
        #                           0.233353407961089, 0]])
        
        # abdomen_coeffs = np.array([[40.892324374260700, 0.180166254998366,
        #                             0.268643019318576, -18.978626682933616,
        #                             2.390703054972615, 0.172208819345683,
        #                             1.605754253800410, 0]])
        
        # waist_coeffs = np.array([[0, 0, 0, 0, 0, 0, 0, 1]])
        
        # x = np.array([[1.0, self.age, self.weight, self.height,
        #                neck_length, torax_length, abdomen_length,
        #                waist_length]]).T

        # self.perimeters = {
        #     "neck": int(np.dot(neck_coeffs, x)[0][0]),
        #     "torax": int(np.dot(torax_coeffs, x)[0][0]),
        #     "abdomen": int(np.dot(abdomen_coeffs, x)[0][0]),
        #     "waist": int(np.dot(waist_coeffs, x)[0][0]),
        #     "neck_val": valid_neck,
        #     "torax_val": valid_torax,
        #     "abdomen_val": valid_abdomen,
        #     "wasit_val": valid_waist,
        # }
        
        # return self.perimeters
    
    def display(self):
        """Display projection measurements used to estimate the perimeters"""
        plt.figure()
        plt.imshow(cv2.cvtColor(self.marked_image, cv2.COLOR_BGR2RGB))
        plt.show()
        
        
#%% Test module

plt.close('all')

# Select image
path = '/Users/davidshatwell/dev/whrd/AnthopometricMeasurements/images/inferior_amayatest1_corrected.png'
# image = cv2.rotate(cv2.imread(path), cv2.ROTATE_90_COUNTERCLOCKWISE)
image = cv2.imread(path)
age = 23
weight = 80
height = 1.70

# Fit image and other inputs to model
model = AnthropometricModel()
debug_flags = {
    "find_keypoints": 1,
    "validate": 1,
    "segment": 1,
    "find_peaks": 1,
    "estimate_neck_perimeter": 1,
    "estimate_torax_perimeter": 1,
    "estimate_abdomen_perimeter": 1,
    "estimate_waist_perimeter": 1,
    "estimate_hips_perimeter": 1,
    "final_keypoints": 1,
}
model.set_debug_flags(debug_flags)
model.set_model_inputs(image, age, weight, height)
model.estimate_perimeters()
model.display()
print(model.projections_mm)
