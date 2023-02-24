import imutils
import cv2
from skimage.transform import pyramid_gaussian


def pyramid(image, scale = 1.5, minSize = (30, 30)):
    yield image
    
    i = 0
    while True:
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width = w)
        
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
        
        yield image
        
def pyramid_gaussian(image, downscale = 2):
    for (i, resized) in enumerate(pyramid_gaussian(image, downscale = 2)):
        if resized.shape[0] < 30 or resized.shape[1] < 30:
            break
        
        cv2.imshow("Layer {}".format(i + 1), resized)
        cv2.waitKey(0)
        
def sliding_window(image, step_size, window_size):
    for y in range(0, image.shape[0], step_size):
        for x in range(0, image.shape[1], step_size):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])
            
def get_image_from_sliding_window(image_path):
    image = cv2.imread(image_path)
    images = []
    (winW, winH) = (128, 128)
    
    for (x, y, window) in sliding_window(image, step_size = 32, window_size = (winW, winH)):
        if window.shape[0] != winH or window.shape[1] != winW:
            continue
        
        clone = image.copy()
        clone = clone[y:y + winH, x:x + winW, :]
        images.append(clone)
        
    return images

# def stitching_image()