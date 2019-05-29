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

#os.chdir('F:\\code\\udacity_carnd\\Project2\\udacity_ad_pj2')
os.chdir('D:\\Code\\udacity\\udacity_ad_pj2')

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

def search_histogram(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    nwindows = 9 # Choose the number of sliding windows    
    margin = 100 # Set the width of the windows +/- margin    
    minpix = 50 # Set minimum number of pixels found to recenter window

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

    return leftx, lefty, rightx, righty

def search_around_poly(binary_warped, left_fit, right_fit):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 100

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Set the area of search based on activated x-values within the +/- margin of our polynomial function
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
                    left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
                    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
                    right_fit[1]*nonzeroy + right_fit[2] + margin)))
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty
   

# find line points and fit to 2D polynomial
def find_lanes(binary_warped, pre_lines):

    if pre_lines == None:
        leftx, lefty, rightx, righty = search_histogram(binary_warped)
    else: 
        if pre_lines[0].detected == False or pre_lines[1].detected == False:
            leftx, lefty, rightx, righty = search_histogram(binary_warped)
        else:
            leftx, lefty, rightx, righty = search_around_poly(binary_warped, pre_lines[0].fit, pre_lines[1].fit)        
    
    # Fit a second order polynomial to each 
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
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
    left_curverad, right_curverad = measure_curvature(leftx, lefty, rightx, righty, max_y)
    left_dist, right_dist = measure_distance_to_line(middle_x, left_fitx, right_fitx)
    #print(left_curverad, right_curverad, left_dist, right_dist)

    # output all results to new line objects
    left_line = SingleLine(True, left_fit, left_fitx, ploty, left_curverad, left_dist, leftx, lefty)
    right_line = SingleLine(True, right_fit, right_fitx, ploty, right_curverad, right_dist, rightx, righty)

    #sanity check
    if sanity_check(left_line, right_line) == False:
        left_line, right_line = generate_empty_lines()

    return out_img, [left_line, right_line]

def generate_empty_lines(): 
    left_line = SingleLine(False, None, None, None, None, None, None, None)
    right_line = SingleLine(False, None, None, None, None, None, None, None)

    return left_line, right_line

def plot_detected_lines(out_img, left_line, right_line):
    # plot the plane on the detected lines
    for i in range(0,len(left_line.fity)):
        cv2.line(out_img, (left_line.fitx[i]+10,left_line.fity[i]), \
            (right_line.fitx[i]-10,right_line.fity[i]), (0,100,0), 2)

    # Colors in the left and right lane regions
    out_img[left_line.ally, left_line.allx] = [255, 0, 0]
    out_img[right_line.ally, right_line.allx] = [0, 0, 255]

def put_text_on_img(img, curv, dist2center):
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(img,'Radius of curvature: {:2.2f} km'.format(curv),(200,100), font, 2,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(img,'Dist. to center: {:2.2f} m'.format(dist2center),(200,180), font, 2,(255,255,255),2,cv2.LINE_AA)

def rmse(value1, value2): 
    return np.sqrt(np.mean((value1-value2)**2))

def sanity_check(left_line, right_line):
    check1 = rmse(left_line.curvature, right_line.curvature) 
    check2 = rmse(left_line.dist2line, right_line.dist2line) 
    check3 = rmse(left_line.fitx, right_line.fitx) 
    
    #print([check1, check2, check3])
    
    if check1 < 10000 and check2 < 10 and check3 < 5000: 
        return True
    else:
        print("Sanity check failed")
        print([check1, check2, check3])
        return False

#Calculates the curvature of polynomial functions in meters.
def measure_curvature(leftx, lefty, rightx, righty, max_y):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # convert fit in pixel to fit in meters
    leftx_real = leftx * xm_per_pix
    rightx_real = leftx * xm_per_pix
    lefty_real = lefty * ym_per_pix
    righty_real = lefty * ym_per_pix

    # fit the points in meters in new polinomial 
    left_fit_cr = np.polyfit(lefty_real, leftx_real, 2)
    right_fit_cr = np.polyfit(righty_real, rightx_real, 2)
    y_cr = max_y * ym_per_pix

    # Implement the calculation of R_curve (radius of curvature)
    left_cur = (1 + (2*left_fit_cr[0]*y_cr + left_fit_cr[1])**2)**1.5 / np.abs(2*left_fit_cr[0])  
    right_cur = (1 + (2*right_fit_cr[0]*y_cr + right_fit_cr[1])**2)**1.5 / np.abs(2*right_fit_cr[0])  

    return left_cur, right_cur

def measure_distance_to_line(middle_x, left_fitx, right_fitx):
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    left_dist = np.abs(middle_x-left_fitx[-1])*xm_per_pix
    right_dist = np.abs(middle_x-right_fitx[-1])*xm_per_pix

    return left_dist, right_dist

def measure_deviation_left(left_dist, right_dist):
    mid = (left_dist + right_dist)/2
    return mid-left_dist

#%%

# read raw images and undistort the image
raw = mpimg.imread('test_images/test3.jpg')

img_undist = undistort_image(raw, mtx, dist)

# detect edges
color_binary, img_edges = detc_edge(img_undist)

# transform perspective to top-down
img_topdown, perspective_M = perspective_change(img_edges)

# find lines and calculate curvature
left_line, right_line = generate_empty_lines()
img_linefit, [left_line, right_line] = find_lanes(img_topdown, [left_line, right_line])
plot_detected_lines(img_linefit, left_line, right_line)
#print(vars(left_line))

# back-warp the line plotting
img_warpback, perspective_M = perspective_change(img_linefit, reverse=True)

# blend back-warped line plotting with undistored original image
img_blended = cv2.addWeighted(img_undist, 0.7, img_warpback, 1.0, 0)

# put text on image
put_text_on_img(img_blended, (left_line.curvature+right_line.curvature)/2000, \
    measure_deviation_left(left_line.dist2line, right_line.dist2line))

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
from moviepy.editor import ImageSequenceClip
from moviepy.editor import ipython_display
from IPython.display import HTML
from ring_buffer import RingBuffer

# Define a class to receive the characteristics of each line detection
class LineTracer():
    def __init__(self):
        self.last5_left = RingBuffer(5)
        self.last5_right = RingBuffer(5)
        self.avg_left_fit = 0
        self.avg_right_fit = 0
        self.avg_left_curv = 0 
        self.avg_right_curv = 0 
        self.avg_left_dist = 0
        self.avg_right_dist = 0

    def getLast(self):
        if len(self.last5_left.get()) > 0 and len(self.last5_right.get()) > 0:
            return [self.last5_left.get()[-1], self.last5_right.get()[-1]]
        else:
            return generate_empty_lines()

    def getAvg(self):
        left_list = self.last5_left.get()
        right_list = self.last5_right.get()
        num_lines = len(left_list)
        #print(num_lines)
        ploty = self.last5_left.get()[-1].fity

        sum_curvl = 0
        sum_curvr = 0
        sum_distl = 0
        sum_distr = 0
        sum_fitl = [0, 0, 0]
        sum_fitr = [0, 0, 0]

        if num_lines > 0:
            for left in left_list:
                sum_curvl += left.curvature
                sum_distl += left.dist2line
                sum_fitl = np.add(sum_fitl, left.fit)
            for right in right_list:
                sum_curvr += right.curvature
                sum_distr += right.dist2line
                sum_fitr = np.add(sum_fitr, right.fit)

        curvl = sum_curvl / num_lines
        curvr = sum_curvr / num_lines
        distl = sum_distl / num_lines
        distr = sum_distr / num_lines
        left_fit = np.divide(sum_fitl, num_lines)
        right_fit = np.divide(sum_fitr, num_lines)
        #print(left_fit)

        # Draw the fit line plain
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        left_fitx = left_fitx.astype(int)
        right_fitx = right_fitx.astype(int)

        leftl = SingleLine(True,left_fit, left_fitx,ploty,curvl,distl,\
            self.last5_left.get()[-1].allx, self.last5_left.get()[-1].ally)
        rightl = SingleLine(True,right_fit, right_fitx,ploty,curvr,distr,\
            self.last5_right.get()[-1].allx, self.last5_right.get()[-1].ally)

        return leftl, rightl

    def newLine(self, new_left, new_right):
        if new_left.detected == False or new_right.detected == False:
            #self.last5_left.append([False])
            #self.last5_right.append([False])
            pass
        else:
            self.last5_left.append(new_left)
            self.last5_right.append(new_right)

        left_list = self.last5_left.get()
        right_list = self.last5_right.get()
        num_lines = len(left_list)

        if num_lines > 0:
            for left in left_list:
                #self.avg_left_fit = np.add(self.avg_left_fit, left.fit)
                self.avg_left_curv += left.curvature
                self.avg_left_dist += left.dist2line
            for right in right_list:
                #self.avg_right_fit = np.add(self.avg_right_fit, right.fit)
                self.avg_right_curv += right.curvature
                self.avg_right_dist += right.dist2line
            
            #self.avg_left_fit = np.divide(self.avg_left_fit, num_lines)
            #self.avg_right_fit = np.divide(self.avg_right_fit, num_lines)
            self.avg_left_curv /= num_lines
            self.avg_right_curv /= num_lines
            self.avg_left_dist /= num_lines
            self.avg_right_dist /= num_lines
    

def process_image(raw, pre_lines):
    img_undist = undistort_image(raw, mtx, dist)

    # detect edges
    color_binary, img_edges = detc_edge(img_undist)

    # transform perspective to top-down
    img_topdown, perspective_M = perspective_change(img_edges)

    # find lines and calculate curvature
    img_linefit, [left_line, right_line] = find_lanes(img_topdown, pre_lines)
    tracer.newLine(left_line, right_line)
    leftl, rightl = tracer.getAvg()
    plot_detected_lines(img_linefit, leftl, rightl)

    # back-warp the line plotting
    img_warpback, perspective_M = perspective_change(img_linefit, reverse=True)

    # blend back-warped line plotting with undistored original image
    img_blended = cv2.addWeighted(img_undist, 0.7, img_warpback, 1.0, 0)
    
    # put text on image
    put_text_on_img(img_blended, (leftl.curvature+rightl.curvature)/2000, \
        measure_deviation_left(leftl.dist2line, rightl.dist2line))

    return img_blended

# create line tracer
tracer = LineTracer()

#in_clip = VideoFileClip("project_video.mp4").subclip(0,2)
in_clip = VideoFileClip("project_video.mp4")
#in_clip = VideoFileClip("challenge_video.mp4")
#out_clip = in_clip.fl_image(process_image) #NOTE: this function expects color images!!
new_frames = []
for frame in in_clip.iter_frames():
    new_frame = process_image(frame, tracer.getLast())
    new_frames.append(new_frame)
out_clip = ImageSequenceClip(new_frames, fps = 25)
out_clip.write_videofile('output_images/project_video.mp4')
#ipython_display(out_clip)



#%%


