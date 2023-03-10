import os 
import torch
from PIL import Image
import numpy as np
from Helpers.helpers import get_image_from_sliding_window, forward_prop, forward_prop_without_slide, stitching_image, clahe
import cv2
from PIL import Image

def is_sliding(slide = True):
    if slide:
        sliding_images = get_image_from_sliding_window('gambar.jpg')
        gans_images = forward_prop(sliding_images)
        stitched = stitching_image(gans_images)
        stitched = cv2.cvtColor(stitched, cv2.COLOR_RGB2BGR)
        cv2.imwrite('output.png', stitched)
        cv2.imshow("Stitched", stitched)
        cv2.waitKey(0)
    else:
        image = cv2.imread('gambar.jpg')
        image = forward_prop_without_slide(image)
        image = np.array(image)
        image = clahe(image)
        cv2.imwrite('output.png', image)
        cv2.imshow("Result", image)
        cv2.waitKey(0)

def main() -> int:
    is_sliding(slide = False)
    
    # image = Image.open('gettyimages-744-68-640x640.jpg').convert('RGB')
    # sr_image = model.predict(image)
    # sr_image.save('ha.jpg')
    
    """
    for i, image in enumerate(os.listdir("inputs")):
        image = Image.open(f"inputs/{image}").convert('RGB')
        sr_image = model.predict(image)
        sr_image.save(f'results/{i}.png')
    """
    


if __name__ == '__main__':
    main()