from PIL import Image

def get_image_resolution(image_path):
    image = Image.open(image_path)
    width, height = image.size
    return width, height

def evaluate_image_resolution(image_path, threshold_width=512, threshold_height=512):
    width, height = get_image_resolution(image_path)
    
    if width >= threshold_width and height >= threshold_height:
        print(f"The image has a high resolution: {width}x{height}")
    else:
        print(f"The image has a low resolution: {width}x{height}")


