import numpy as np

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None  

    def rmse(self, value1, value2): 
        return np.sqrt(np.mean((value1-value2)**2))

    def sanity_check(self, new_fit, new_curvature, new_linebase): 
        check_result = True

        if self.current_fit != [np.array([False])] and rmse(new_fit, self.current_fit) > 100: 
            check_result = False

        if self.radius_of_curvature != None and rmse(new_curvature, self.radius_of_curvature) > 10: 
            check_result = False

        if self.line_base_pos != None and rmse(new_linebase, self.line_base_pos) > 10:
            check_result = False

        return check_result



