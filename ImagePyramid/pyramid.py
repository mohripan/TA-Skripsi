import imutils
import cv2
from skimage import pyramid_gaussian


def pyramids(image, scale = 1.5, minSize = (30, 30)):
    yield image
    
    while True:
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width = w)
        
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
        
        yield image
        
def pyramids_method(image, downscale = 2):
    