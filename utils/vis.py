import cv2
import numpy as np
import sys

def visualize_disparity(disp):
    disp = np.minimum(256, disp*2).astype(np.uint8)
    color_disp = cv2.applyColorMap(disp, cv2.COLORMAP_JET)
    cv2.imshow('disp', color_disp)
    cv2.waitKey(0)

if __name__ == '__main__':
    disp = np.load(sys.argv[1])
    visualize_disparity(disp)
