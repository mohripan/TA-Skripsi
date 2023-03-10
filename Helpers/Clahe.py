import cv2
import numpy as np
 
image = cv2.imread('gambar.jpg')
image = cv2.resize(image, (500, 600))
 
image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
clahe = cv2.createCLAHE(clipLimit = 5)
final_img = clahe.apply(image_bw) + 30
 
_, ordinary_image = cv2.threshold(image_bw, 155, 255, cv2.THRESH_BINARY)
 
cv2.imshow('ordinary threshold', ordinary_image)
cv2.imshow('CLAHE image', final_img)
cv2.waitKey(0)