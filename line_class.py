import numpy as np

class SingleLine():
    def __init__(self, detected, fit, fitx, fity, curv, dist, allx, ally):
        self.detected = detected  
        # polynomial coefficients for the most recent fit
        self.fit = fit
        # fitted X value
        self.fitx = fitx
        self.fity = fity 
        #radius of curvature of the line in some units
        self.curvature = curv 
        #distance in meters of vehicle center from the line
        self.dist2line = dist 
        #x values for detected line pixels
        self.allx = allx  
        self.ally = ally  



