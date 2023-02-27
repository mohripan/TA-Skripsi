import os 
import torch
from PIL import Image
import numpy as np
from Helpers.helpers import get_image_from_sliding_window, forward_prop, stitching_image
import cv2
from PIL import Image

def main() -> int:
    sliding_images = get_image_from_sliding_window('pa.jpg')
    gans_images = forward_prop(sliding_images)
    stitched = stitching_image(gans_images)
    cv2.imwrite('output.png', stitched)
    cv2.imshow("Stitched", stitched)
    cv2.waitKey(0)
    
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