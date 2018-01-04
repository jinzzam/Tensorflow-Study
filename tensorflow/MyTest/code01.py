from matplotlib import  pyplot as plt
import numpy as np
import os
import cv2

image_path = r'C:\Users\jin\Documents\Tensorflow-Study\tensorflow\animal_images\animal_images\cow\images.jpeg'

img = cv2.imread(image_path)    # img는 numpy array이다.
# imread()함수의 2번째 매개변수로 다음 중 하나를 지정할 수 있다
# cv2.IMREAD_COLOR      default flag
# cv2.IMREAD_GRAYSCALE  grayscale mode
# cv2.IMREAD_UNCHANGED  include alpha channel

# numpy array가 가진 기본 attribute 확인
print(img.ndim)     # number of array dimensions
print(img.shape)    # tuple of array dimensions
print(img.dtype)    # data type  == unit8

print(img)
print(img.tolist())

# load an color image in grayscale
# img = cv2.imread(image_path, 0)

# cv2.namedWindow('image', cv2.WINDOW_NORMAL)
# cv2.imshow('image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# img = cv2.resize(img, (200, 200), interpolation=cv2.INTER_CUBIC)    # resize
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2 load images as BGR, convert it to RGB
#
# cv2.imwrite('converted.jpeg', img)      # 이미지 파일 저장


plt.imshow(img)
plt.show()

def plot_images(image):
    # create figure with 4*4 sub-plots.
    fig, axes = plt.subplots(4, 4)

    for i, an in enumerate(axes.flat):
        row = i // 4
        col = i % 4
        image_frag = image[row*50:(row+1)*50, col*50:(col+1)*50, :]
        axes.imshow(image_frag)

        xlabel = '{},{}'.format(row, col)
        axes.set_xlabel(xlabel)

        axes.set_xticks([])     # Remove ticks from the plot
        axes.set_yticks([])

    plt.show()

plot_images(img)