import numpy as np
import os
import cv2

image_path = r'C:\Users\jin\Documents\Tensorflow-Study\tensorflow\animal_images\animal_images\cow\images.jpeg'

img = cv2.imread(image_path)    # img는 numpy array이다.

# numpy array가 가진 기본 attribute 확인
print(img.ndim)     # number of array dimensions
print(img.shape)    # tuple of array dimensions
print(img.dtype)    # data type  == unit8

print(img)
print(img.tolist())

# load an color image in grayscale
img = cv2.imread(image_path, 0)

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()