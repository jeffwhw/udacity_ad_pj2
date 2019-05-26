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
[image3]: ./output_images/topdown.png "Top Down View"
[image4]: ./output_images/linefit.png "Line detected and fit"
[image5]: ./output_images/blended.png "Warp back to original view and blended over"
[video1]: ./output_images/project_video_first_run.mp4 "First Run"
[video2]: ./output_images/project_video_second_run.mp4 "Second Run"

[image6]: ./output_images/undistort-example.png "Image Calibration"
[image7]: ./output_images/warp.png "Perspective Change"
[image7]: ./output_images/pipeline.png "Perspective Change"

## Description of the work

### Summary

[Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

I have started with invididual steps such as calibration, perspective transform etc. Each individual function is tested using the given sample images. After testing them all, I combined all the code in a single file: pipeline.py. I built first a test run (line 273 to line 311) in "pipeline.py"), which combines all these steps together. After the test run is working, I started with video. 

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

Line finding algorithm (in "find_lanes()" function) has several steps built-in: 
1) it searchs for points of left and right lines using histogram and interactive windows. This step outputs "leftx", "lefty", "rightx" and "righty" in "find_lanes()". 
2) after the points are collected for both lines, it fits the points into a 2nd order polinomial equation, i.e. ax**2 + bx + c. This step outputs "left_fit" and "right_fit" in "find_lanes()". 
3) the algorithm then recalculates the points using the polinomial. This step outputs "ploty", "left_fitx" and "right_fitx" in "find_lanes()". 
4) the algorithm will caculate the radius of curverad and distance to the line, for both left and right lines. This function is given in "measure_curvature_pixels()". 
5) finally, the algorithm checks the sanity of the found lines. This function is given by "sanity_check". If checks succeed, found lines will be stored into two new Line objects, with "detected" value set to "True"; otherwise, algorithm gives two empty lines with "detected" value set to "False". 

"plot_detected_lines" will paint the found lines and line points on the top-down view image calculated in the last step. The green area points the discovered lane betwen left and right lines. Red and blue dots illustrate the original found line points. An example is given below: 

![alt text][image4]

#### 5. The image with discovered lane painted will be warped back to original perspective of camera and blended over the original image

Warping-back is easily implemented by reverse using of "perspective_change()", simply exchaging the source and destination points. 

Blending over original image can be achieved using "addWeighted()" from OpenCV.

Finally, texts about the radius of curvature and distance to left lines are put on the outputted image. An example is given below: 

![alt text][image5]

#### Summary

We can take an overview of the complete pipleline again in the combined picture: 

![alt text][image8]

### Pipeline (video)

#### 1. First Trial

My first trial starts with simply packing the pipeline mentioned in last section into a function called "process_image()". The result is stored in [link to my first trial video](./output_images/project_video_first_run.mp4). 

The results seem to be very reasonable except that the detection encounters some disturtance when the line is missing for some time or there is strong shadow on the ground.

#### 2. Smoothing and Line Tracing 

After first trial, I started to think the optimization. The basic idea is to build a simple LP filter, which outputs the average of last 5 results. Also, I planned to use sanity check to remove "bad" results. 

To take concrete steps, I built the "LineTracer" class. The class provides two major function: "newLine()" and "getAvg()". 

"newLine()" function takes the newly found lines in current iteration. It will take only "good" result which passed the sanity check. 

"getAvg()" function returns a pair of "artificial" lines which averages over the last 5 iterations. Concretely, it will calculate the average polinomial coefficiencies, the distances to left/right lines and the radius of curvature. After that, it calculates the newly fitted points and return the artificial lines. 

"LineTracer" class allows me to simply exchange the lines returned by "find_lanes()" by the artificial lines given by "LineTracer" class. These lines are passed to the "plot_detected_lines()" and "put_text_on_img()" functions. Therefore, the result image will be changed to the filterred lines. 

The result is stored then in [link to my second trial video](./output_images/project_video_second_run.mp4). We can observe that in difficult situation such as missing line marker and strong shadow, the algorithm is much more robust and gives reasonable estimation based on previous experiences. 

---

### Discussion

I have not yet optimized line finding algorithm based on sanity check result. Currently, searching points is started from scratch every time. This could be optimized later by searching only a small region based on the result of last search and sanity check. Nevertheless, it does not impact the performance of the algorithm so I passed over this feature for now. 

Also, I will consider improve the smoothing algorithm. Currently, it simply drops the "bad" result. However, in situation where the results are "bad" for a long while and recovered to good again, this algorithm will store some very old valid result in its circular buffer and user will notice strange fitting at very beginning after recovery. This might be improved by recording also the bad results but using onlz the good results for line finding. 