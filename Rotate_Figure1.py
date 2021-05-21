import cv2
import matplotlib.pyplot as plt
import numpy as np
import math



def rotate_bound(img,angle):

    h,w = img.shape[0], img.shape[1]

    center = (w//2,h//2)


    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos_Rotation = np.abs(rotation_matrix[0][0])
    sin_Rotation = np.abs(rotation_matrix[0][1])


    h1 = int((h*sin_Rotation)+(w*cos_Rotation))
    w1 = int((h*cos_Rotation)+(w*sin_Rotation))

    rotation_matrix[0][2] += ((w1/2)-(w//2))
    rotation_matrix[1][2] += ((h1/2)-(h//2))

    final_rotated = cv2.warpAffine(img, rotation_matrix, (w1, h1))

    return final_rotated

img = cv2.imread('base_image.png')

plt.imshow(rotate_bound(img,45),cmap='gray')
plt.show()
