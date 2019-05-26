#%%
# import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def perspective_change(img):

    src = np.float32([(264,670), (579,460), (703,460), (1042,670) ])
    dest = np.float32([(264,670), (264,100), (1042,100), (1042,670)])

    M = cv2.getPerspectiveTransform(src, dest)
    warped = cv2.warpPerspective(img, M, (img.shape[1],img.shape[0]), flags=cv2.INTER_LINEAR)
    
    return warped, M


raw = mpimg.imread('test_images/straight_lines1.jpg')
top_down, perspective_M = perspective_change(raw)

# plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(raw)
ax1.set_title('Original Image', fontsize=50)
ax1.plot([264,579], [670, 460], color='r', linestyle='-', linewidth=2)
ax1.plot([579,703], [460, 460], color='r', linestyle='-', linewidth=2)
ax1.plot([703,1042], [460, 670], color='r', linestyle='-', linewidth=2)
ax1.plot([1042,264], [670, 670], color='r', linestyle='-', linewidth=2)
ax2.imshow(top_down)
ax2.set_title('Unwarped Image', fontsize=50)
ax2.plot([264,264], [670, 100], color='r', linestyle='-', linewidth=2)
ax2.plot([264,1042], [100, 100], color='r', linestyle='-', linewidth=2)
ax2.plot([1042,1042], [100, 670], color='r', linestyle='-', linewidth=2)
ax2.plot([1042,264], [670, 670], color='r', linestyle='-', linewidth=2)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)


#%%
