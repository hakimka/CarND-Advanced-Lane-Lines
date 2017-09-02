## Writeup 

### This document addresses solution points for Advanced Lane Finding project.

---

**Advanced Lane Finding Project main steps**

During the implementation of this project the following steps were taken:

* A separate script calculates camera calibration matrix and distortion coefficients for a given set of chessboard images.
* Ever frame in the video undergoes a distortion correction.
* Following distortion correction, a binary filter thresholds the undistorted image.
* The binary thresholded images go through a perspective transformation to obtain "birds-eye view".
* Lane pixels are detected and fit to find the lane boundary.
* Using line fit coefficients, we determine the curvature of the lane.
* Vehicle position with respect to center of the lane is calculated.
* To visualize the results, the detected lane boundaries are mapped back onto the original image.
* The final output displays of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./test_images/test1.jpg "Undistorted"
[image2]: ./output_images/test1.jpg "Road Transformed"
[image3]: ./output_images/binarized.jpg "Binary Example"
[image4]: ./output_images/warped.jpg "Warp Example"
[image5]: ./output_images/lanes.jpg "Fit Visual"
[image6]: ./output_images/sampleVideoFrame.jpg "Output"
[video1]: ./project_video.mp4 "Video"



---

### Camera Calibration

#### 1. Camera matrix and distortion coefficients. 

The code for camera calibration is located in "./calibration.ipynb".  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  

	objpoints=[] #3D points in real world
	imgpoints=[] #2D points in image	

	#prepare obj points (0,0,0), (1,0,0) (2,0,0)....(9,6,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)
    
Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

First I converted image to gray scale. 

 	# convert to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


	# rvecs, tvecs - camera position in the world, rotation and translation pos
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. The calibration results are stored in a pickle file, which is later used for rectifying the images. 

	calPickle = open ('calib.p', 'wb')
	pickle.dump([ret, mtx, dist, rvecs, tvecs], calPickle)
	calPickle.close()

I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. An example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Color transforms, gradients or other methods to create a thresholded binary image.  .

I used a combination of color and gradient thresholds to generate a binary image (thresholding in `processImage.ipynb`, cell #2 ***def binarize_image(img)***).  Here's an example of my output for this step.  

![alt text][image3]

#### 3. Perspective transformation.

To perform perspective transformation, I came up with the following number :


	src = np.float32(
	    [  [690 ,450],
	       [920 ,600],               
	       [380 , 600 ],
	       [595 , 450]
	    ]
	    )
	
	dst = np.float32(
	    [  [1050 , 50  ],
	       [1050 , 700 ],               
	       [150 , 700 ],
	       [150 , 50  ]
	     
	    ]
	    )

	M = cv2.getPerspectiveTransform(src, dst)
	img_size = (int(img.shape[1]), int(img.shape[0]-10))
	calPickle = open ('pTrans.p', 'wb')
	pickle.dump([M, img_size, src, dst], calPickle)
	calPickle.close()

The perspective transformation matrix and the image size values have been stored in a pickle file (***calibrateCamera.ipynb***, last cell) 

 In the proessImage file the perspective transformation matrix been loaded and utlized as follows:

   	ptransPickle = open ('pTrans.p', 'rb')
	M, img_size, srcTrans, dstTrans = pickle.load( ptransPickle)
	# projec the image orthogonal above view
    binary_warped = cv2.warpPerspective(combined_binary, M, img_size, flags=cv2.INTER_LINEAR)



I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Lane-line pixels identification  and polynomial positions fit.

To fit my lane lines from the warped image, I used 2nd order polynomial listed in the function ***def LanesStartingPoint*** and ***def GetLineIndicators***. First, I histogrammed the bottom part of the image to locate the line starting points:

	binary_warped = img 
    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
   
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

Once the starting points of the lanes been identified, I sliced the image in the strips and positioned small window blocks starting from each left and right lane positions respectively. 

	for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        #cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        #cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

The last step was to fit points obtained in the steps above on the image with 2nd degree polynomials. The function ***def FitPolynomials*** does exactly that. 

  
![alt text][image5]

#### 5. The radius of curvature of the lane and the position of the vehicle with respect to center.

Once the fitting polynomials are established. I use conversion for x,y pixels on the image to the world coordinates. 

	# Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

I obtain the x, y in the world
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)

Calculate radius in with world using the following formula:
    # Calculate the new radii of curvature
    
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / (2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / (2*right_fit_cr[0])

All this takes place in cell #5 of the *** processImage.ipynb*** file. The function is called "**processImage**" 

#### 6. An example image of the result plotted back down onto the road.

I implemented plotting lane overlay in the proccessImage function cell #5 of the *** processImage.ipynb*** file.    Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./videoOut/project_video.mp4)

---

### Discussion

#### 1. Problems / issues you faced during this project. 

Where will your pipeline likely fail?  What could you do to make it more robust?

I used the pipeline suggested during the video lessons. The approach of binarizing and thresholding works on the images that are not too busy with the scenery. This approach does not perform well on the video where the road curves or has reflection from the wind shield. The 

On the image with a lot of brightness or varying contrast, it is worth exploring the overall ambiance of the frame and make the binarization thresholding dynamic.

On the road that curves the window slicing needs to be more delicate, not the hardcoded 10 window sliced per frame. I think the slicing number needs to monitor the curvature of the road and dynamically adjust based what's been seen in the recent past. 

  
