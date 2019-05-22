#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os

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

if test_distort: 
    raw = cv2.imread('camera_cal/calibration12.jpg')
    img_undist = undistort_image(raw, mtx, dist)

    # plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(raw)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(img_undist)
    ax2.set_title('Undistorted Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

# detect the line edges using a combination of absolute sobel threshold 
# and S channel threshold

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

if test_edge_detc:
    # Read in an image
    images = glob.glob('test_images/test*.jpg')
    i = 0
    ret_list = []
    img_list = []
    for fname in images:
        image = mpimg.imread(fname)

        ksize = 3

        # gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh = (10,100))
        # grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh = (10,100))
        # mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(15, 100))
        # dir_binary = dir_thres(image, sobel_kernel=ksize, thresh=(0.7, 1.2))

        img_edges = np.zeros_like(dir_binary)
        #img_edges[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
        #img_edges[ ((mag_binary == 1) & (dir_binary == 1))] = 1
        color_binary, img_edges = detc_edge(image)

        img_list.append(image)
        ret_list.append(img_edges)
        i = i+1
    # Run the function

    # Plot the result 
    f, ((ax1,ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(3, 2, figsize=(24, 27))
    f.tight_layout()
    ax1.imshow(img_list[0])
    ax2.imshow(img_list[1])
    ax3.imshow(img_list[2])
    ax4.imshow(img_list[3])
    ax5.imshow(img_list[4])
    ax6.imshow(img_list[5])
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)


    f, ((ax1,ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(3, 2, figsize=(24, 27))
    f.tight_layout()
    ax1.imshow(ret_list[0], cmap='gray')
    ax2.imshow(ret_list[1], cmap='gray')
    ax3.imshow(ret_list[2], cmap='gray')
    ax4.imshow(ret_list[3], cmap='gray')
    ax5.imshow(ret_list[4], cmap='gray')
    ax6.imshow(ret_list[5], cmap='gray')
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

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

if test_perspective:
    raw = mpimg.imread('test_images/straight_lines1.jpg')
    img_topdown, perspective_M = perspective_change(raw, 0, 0, 0, 0)

    # plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(raw)
    ax1.set_title('Original Image', fontsize=50)
    ax1.plot([264,579], [670, 460], color='r', linestyle='-', linewidth=2)
    ax1.plot([579,703], [460, 460], color='r', linestyle='-', linewidth=2)
    ax1.plot([703,1042], [460, 670], color='r', linestyle='-', linewidth=2)
    ax1.plot([1042,264], [670, 670], color='r', linestyle='-', linewidth=2)
    ax2.imshow(img_topdown)
    ax2.set_title('Unwarped Image', fontsize=50)
    ax2.plot([264,264], [670, 100], color='r', linestyle='-', linewidth=2)
    ax2.plot([264,1042], [100, 100], color='r', linestyle='-', linewidth=2)
    ax2.plot([1042,1042], [100, 670], color='r', linestyle='-', linewidth=2)
    ax2.plot([1042,264], [670, 670], color='r', linestyle='-', linewidth=2)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

# find line points and fit to 2D polynomial
def find_lanes(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    #out_img = np.zeros_like(binary_warped)
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
        ### TO-DO: Find the four below boundaries of the window ###
        win_xleft_low = leftx_current-margin  # Update this
        win_xleft_high = leftx_current+margin  # Update this
        win_xright_low = rightx_current-margin  # Update this
        win_xright_high = rightx_current+margin  # Update this
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low), (win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low), (win_xright_high,win_y_high),(0,255,0), 2)        
        
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
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Fit a second order polynomial to each 
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    return out_img, left_fit, right_fit


def draw_fit_lines(img, left_fit, right_fit, ax):
    # Generate x and y values for plotting
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    # Plots the left and right polynomials on the lane lines
    ax.plot(left_fitx, ploty, color='yellow', linewidth=8)
    ax.plot(right_fitx, ploty, color='yellow', linewidth=8)

    return

if test_fitpoly:     
    binary_warped = mpimg.imread('F:\\code\\udacity_carnd\\Project2\\CarND-Advanced-Lane-Lines\\warped-example.jpg')
    out_img = find_lanes(binary_warped)
    plt.imshow(out_img)

#Calculates the curvature of polynomial functions in meters.
def measure_curvature_pixels(max_y, left_fit_cr, right_fit_cr):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    #xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Implement the calculation of R_curve (radius of curvature)
    left_curverad = (1 + (2*left_fit_cr[0]*max_y*ym_per_pix + left_fit_cr[1])**2)**1.5  \
        / np.abs(2*left_fit_cr[0])  
    right_curverad = (1 + (2*right_fit_cr[0]*max_y*ym_per_pix + right_fit_cr[1])**2)**1.5  \
        / np.abs(2*right_fit_cr[0])  
    
    return left_curverad, right_curverad

#%%

raw = mpimg.imread('test_images/test3.jpg')

img_undist = undistort_image(raw, mtx, dist)

img_topdown, perspective_M = perspective_change(img_edges)

color_binary, img_edges = detc_edge(img_undist)

img_linefit, left_fit, right_fit = find_lanes(img_topdown)

left_curverad, right_curverad = measure_curvature_pixels(np.max(img_linefit.shape[0]), left_fit, right_fit)
print(left_curverad, right_curverad)

img_warpback, perspective_M = perspective_change(img_linefit, reverse=True)

f, ((ax1,ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(3, 2, figsize=(24, 27))
f.tight_layout()

ax1.imshow(raw)
ax2.imshow(img_undist)
ax3.imshow(img_edges, cmap='gray')
ax4.imshow(img_topdown, cmap='gray')
ax5.imshow(img_topdown, cmap='gray')
draw_fit_lines(img_topdown, left_fit, right_fit, ax5)
ax6.imshow(img_warpback, cmap='gray')
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)


#plt.savefig('top-down.png')



#%%
