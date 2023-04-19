import numpy as np
from Helpers.helpers import get_image_from_sliding_window, forward_prop, forward_prop_without_slide, stitching_image, equalize_hist
import cv2

def is_sliding(slide = True):
    if slide:
        sliding_images = get_image_from_sliding_window('cctv.jpg')
        gans_images = forward_prop(sliding_images, scale = 2, path = 'weights/RealESRGAN_x2.pth')
        stitched = stitching_image(gans_images)
        stitched = cv2.cvtColor(stitched, cv2.COLOR_RGB2BGR)
        final_img = equalize_hist(stitched)
        cv2.imwrite('output.png', final_img)
        cv2.imshow("Stitched", final_img)
        cv2.waitKey(0)
        
    else:
        image = cv2.imread('divisi-erg-unikom-raih-juara-1-dalam-kompetisi-nasional-medinnovation-2020_1.jpg')
        image = forward_prop_without_slide(image)
        # image = np.array(image)
        # image = equalize_hist(image)
        image = np.array(image)
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