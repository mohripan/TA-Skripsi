import os 
import torch
from PIL import Image
import numpy as np
from ESRGAN import ESRGAN
from Helpers.helpers import get_image_from_sliding_window
import cv2
import slidingwindow as sw

def forward_prop(image, scale = 2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ESRGAN(device, scale = scale)
    model.load_weights('weights/RealESRGAN_x2.pth', download=True)
    images = []
    
    for i, img in enumerate(image):
        sr_image = model.predict(img)
        images.append(sr_image)
        print(f'Gambar {str(i)}')
        
    return images

def main() -> int:
    split_image = sliding_show('gettyimages-744-68-640x640.jpg')
    sr_images = forward_prop(split_image)
    stitchy = cv2.Stitcher.create()
    (dummy, output) = stitchy.stitch(sr_images)
    if dummy != cv2.STITCHER_OK:
        print("stitching ain't successful")
    else: 
        print('Your Panorama is ready!!!')
    
    output.save('haa.jpg')
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