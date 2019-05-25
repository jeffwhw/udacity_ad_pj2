#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os
from PIL import Image

test_distort = False
test_edge_detc = False
test_perspective = False
test_fitpoly = False
test_curve = False

os.chdir('F:\\code\\udacity_carnd\\Project2\\udacity_ad_pj2')

#%% prepare the calibration data based on calibration images
def calibrate_prepare(images):

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Step through the list and search for chessboard corners
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        print (img.shape)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            #img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
            #cv2.imshow('img',img)
            #cv2.waitKey(500)

    np.concatenate(objpoints)
    np.concatenate(imgpoints)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1:], None, None)

    return mtx, dist 

images = glob.glob('camera_cal/calibration*.jpg')
mtx, dist = calibrate_prepare(images)

#%%  undistort the image using calibration data

def undistort_image(raw, mtx, dist):
    undist = cv2.undistort(raw, mtx, dist, None, mtx)
    
    return undist

def detc_edge(img, s_thresh=(170, 255), sx_thresh=(22, 100)):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]

    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # combined binary
    combined = np.zeros_like(s_channel)
    combined[ ((s_binary == 1) | (sxbinary == 1))] = 1

    # Stack each channel
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    return color_binary, combined

#change perspective
def perspective_change(img, reverse = False):

    if reverse == False: 
        src = np.float32([(264,670), (579,460), (703,460), (1042,670) ])
        dest = np.float32([(264,670), (264,100), (1042,100), (1042,670)])
    else: 
        dest = np.float32([(264,670), (579,460), (703,460), (1042,670) ])
        src = np.float32([(264,670), (264,100), (1042,100), (1042,670)])

    M = cv2.getPerspectiveTransform(src, dest)
    warped = cv2.warpPerspective(img, M, (img.shape[1],img.shape[0]), flags=cv2.INTER_LINEAR)
    
    return warped, M

from line_class import SingleLine

# find line points and fit to 2D polynomial
def find_lanes(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        # Find the four below boundaries of the window ###
        win_xleft_low = leftx_current-margin  # Update this
        win_xleft_high = leftx_current+margin  # Update this
        win_xright_low = rightx_current-margin  # Update this
        win_xright_high = rightx_current+margin  # Update this
        
        # Draw the windows on the visualization image
        #cv2.rectangle(out_img,(win_xleft_low,win_y_low), (win_xleft_high,win_y_high),(0,255,0), 2) 
        #cv2.rectangle(out_img,(win_xright_low,win_y_low), (win_xright_high,win_y_high),(0,255,0), 2)        
        
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & \
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & \
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window 
        # (`right` or `leftx_current`) on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        
    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each 
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Draw the fit line plain
    ploty = np.linspace(0, out_img.shape[0]-1, out_img.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    ploty = ploty.astype(int)
    left_fitx = left_fitx.astype(int)
    right_fitx = right_fitx.astype(int)

    # calculate curverad and distance
    middle_x = int(out_img.shape[1]/2)
    max_y = np.max(out_img.shape[0])
    left_curverad, right_curverad, left_dist, right_dist = \
        measure_curvature_pixels(max_y, middle_x, left_fit, right_fit)
    #print(left_curverad, right_curverad, left_dist, right_dist)
    
    # plot the plane on the detected lines
    for i in range(0,len(ploty)):
        cv2.line(out_img, (left_fitx[i]+10,ploty[i]), (right_fitx[i]-10,ploty[i]), \
            (0,100,0), 2)

    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # output all results to new line objects
    left_line = SingleLine(True, left_fit, left_fitx, ploty, left_curverad, left_dist, leftx, lefty)
    right_line = SingleLine(True, right_fit, right_fitx, ploty, right_curverad, right_dist, rightx, righty)

    return out_img, [left_line, right_line]

#Calculates the curvature of polynomial functions in meters.
def measure_curvature_pixels(max_y, middle_x, left_fit_cr, right_fit_cr):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Implement the calculation of R_curve (radius of curvature)
    left_curverad = (1 + (2*left_fit_cr[0]*max_y*ym_per_pix + left_fit_cr[1])**2)**1.5  \
        / np.abs(2*left_fit_cr[0])  
    right_curverad = (1 + (2*right_fit_cr[0]*max_y*ym_per_pix + right_fit_cr[1])**2)**1.5  \
        / np.abs(2*right_fit_cr[0])  
    
    left_dist = np.abs(middle_x-left_fit_cr[0])*xm_per_pix
    right_dist = np.abs(middle_x-right_fit_cr[0])*xm_per_pix

    return left_curverad, right_curverad, left_dist, right_dist

#%%

# read raw images and undistort the image
raw = mpimg.imread('test_images/test3.jpg')

img_undist = undistort_image(raw, mtx, dist)

# detect edges
color_binary, img_edges = detc_edge(img_undist)

# transform perspective to top-down
img_topdown, perspective_M = perspective_change(img_edges)

# find lines and calculate curvature
img_linefit, [left_line, right_line] = find_lanes(img_topdown)
print(vars(left_line))

# back-warp the line plotting
img_warpback, perspective_M = perspective_change(img_linefit, reverse=True)

# blend back-warped line plotting with undistored original image
img_blended = cv2.addWeighted(img_undist, 0.7, img_warpback, 1.0, 0)

# output intermediate steps 
f, ((ax1,ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(3, 2, figsize=(24, 27))
f.tight_layout()

ax1.imshow(raw)
ax2.imshow(img_undist)
ax3.imshow(img_edges, cmap='gray')
ax4.imshow(img_topdown, cmap='gray')
ax5.imshow(img_linefit)
ax6.imshow(img_blended)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

#%%

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

def process_image(raw):
    img_undist = undistort_image(raw, mtx, dist)

    # detect edges
    color_binary, img_edges = detc_edge(img_undist)

    # transform perspective to top-down
    img_topdown, perspective_M = perspective_change(img_edges)

    # find lines and calculate curvature
    img_linefit, [left_line, right_line] = find_lanes(img_topdown)
    
    # back-warp the line plotting
    img_warpback, perspective_M = perspective_change(img_linefit, reverse=True)

    # blend back-warped line plotting with undistored original image
    img_blended = cv2.addWeighted(img_undist, 0.7, img_warpback, 1.0, 0)

    return img_blended

white_output = 'output_images/project_video.mp4'
clip1 = VideoFileClip("project_video.mp4").subclip(0,2)
# clip1 = VideoFileClip("project_video.mp4")
clip1.reader.close()
clip1.audio.reader.close_proc()
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
get_ipython().run_line_magic('time', 'white_clip.write_videofile(white_output, audio=False)')


#%%


