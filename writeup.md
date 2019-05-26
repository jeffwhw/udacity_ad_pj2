## Writeup Template

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistort.png "Undistorted"
[image2]: ./output_images/edge_detected.png "Edge Detected"
[image3]: ./output_images/top-down.png "Top Down View"
[image4]: ./output_images/linefit.png "Line detected and fit"
[image5]: ./output_images/blended.jpg "Warp back to original view and blended over"
[video1]: ./output_images/project_video_first_run.mp4 "First Run"
[video2]: ./output_images/project_video_second_run.mp4 "Second Run"

[image6]: ./output_images/undistort-example.png "Image Calibration"
[image7]: ./output_images/warp.png "Perspective Change"

## Description of the work

### Summary

[Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

I have started with invididual steps such as calibration, perspective transform etc. Each individual function is tested using the given sample images. After testing them all, I combined all the code in a single file: pipeline.py. I built first a test run which combines all these steps together. After the test run is working, I started with video. 

The first trial on video was successful and recorded in "./output_images/project_video_first_run.mp4". The results seem to be very reasonable except that the detection encounters some disturtance when the line is missing for some time or there is strong shadow on the ground. Therefore, I built a second run which adds smoothing function and sanity check. These mechanisms filtered out the unreasonable results and smooth the output. The second run achieved a much better result in the above mentioned scenarios. It is recored in "./output_images/project_video_second_run.mp4". 


### Camera Calibration

Function "calibrate_prepare" calculates the distortion matrix (mtx and dist). It collects all objpoints and imgpoints from all images in "camera_cal" directory, using "findChessboardCorners()" function of OpenCV. Remind that objpoints are the same for all calibration images, the code simply repeat the objpoints every round. 

After collecting of objpoints and imgpoints, the points will be concatenated and given to "calibrateCamera" function of OpenCV to calculate the distoring matrix. This matrix will be used later to correct the distored raw images. 

An example result on chess board image is shown below:

![alt text][image6]

### Pipeline (single images)

The pipeline contains several steps: 


#### 1. The raw image is un-distored 

Un-distortion function is given by "undistor_image()" in "pipeline.py". It takes the undistortion matrix calculated by "calibrate_prepare" before. An example of output is given below:

![alt text][image1]

#### 2. The algorithm detectes the edge using a combination of Sobel threshold and S chanel threshold

Function "detc_edge()" provides the edge detection. An example of the output is given below: 

![alt text][image2]

#### 3. The image is changed to top-down view 

The "perspective_change()" function in "pipeline.py" provides the perspective changing function. The source and destination points are hard-coded. 

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 264, 670      | 264, 670      | 
| 579, 460      | 264, 100      |
| 703, 460      | 1042, 100     |
| 1042, 670     | 1042, 670     |

The points are manually collected on "test_images/straing_lines1.jpg". The following pictures shows the source and destination points on the sample image: 

![alt text][image7]

Following picture gives an example of the real captured frame after perspective change (before that the undistortion and edge detection): 

![alt text][image3]


#### 4. Find the left and right lines 

Line finding algorithm 

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
