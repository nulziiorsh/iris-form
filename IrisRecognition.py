import PIL
import PIL
from PIL import Image
import numpy as np
from os import listdir
import glob
import matplotlib.pyplot as plt
import cv2

# Pre-processing: black and white
image_list = []
for filename in glob.glob('/Users/Nasanbayar/PycharmProjects/IrisForm/Iris Training Dataset/*.JPG'):
    im = Image.open(filename)
    image_list.append(im)

image_data_list = []
for filename in glob.glob('/Users/Nasanbayar/PycharmProjects/IrisForm/Iris Training Dataset/*.JPG'):
    im = Image.open(filename)
    im_data = np.asarray(im)
    image_data_list.append(im_data)
    print(im_data.shape)

name_value = 1
for image in image_list:
    bw_iris = image.convert('L')
    bw_iris.save('/Users/Nasanbayar/PycharmProjects/IrisForm/Iris Training Dataset BW/Image_GS_{}.JPG'.format(str(name_value)))
    name_value += 1

img = cv2.imread('/Users/Nasanbayar/PycharmProjects/IrisForm/Iris Training Dataset BW/Image_GS_1.JPG')
sift = cv2.SIFT_create(img)
show = plt.imshow(image_data_list[1])
