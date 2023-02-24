import os
import sys
import numpy as np
import cv2
from PIL import Image

sys.path.insert(1, 'ImagePyramid')

from pyramid import pyramids, pyramids_method 

def main() -> int:
    image = cv2.imread('gambar.jpg')
    image = np.array(image)
    pyramids_method(image)

if __name__ == '__main__':
    main()