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
    if image.shape[1] > image.shape[0]:
        for x in range(0, image.shape[1], step_size):
            yield (x, image[0:, x:x + window_size[0]])
            
    else:
        for y in range(0, image.shape[0], step_size):
            yield(y, image[y:y + window_size[1], 0:])
            
def get_image_from_sliding_window(image_path):
    image = cv2.imread(image_path)
    images = []
    winH, winW = image.shape[0] // 2, image.shape[1] // 2
    
    for (x, window) in sliding_window(image, step_size = 64, window_size = (winW, 0)):
        if image.shape[1] > image.shape[0]:
            if window.shape[1] != winW:
                continue
            
            clone = image.copy()
            clone = clone[0:, x:x + winW, :]
            images.append(clone)
        
        else:
            if window.shape[0] != winH:
                continue
            
            clone = image.copy()
            clone = clone[y:y + winH, 0:]
            images.append(clone)
        
    return images

# def stitching_image()