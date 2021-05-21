
import cv2
import os
from segment_CT_abdomen1 import segment_extreme_points
import numpy as np
import matplotlib.pyplot as plt



for file in os.listdir("./ImageFolder1"):

    img = cv2.imread(os.path.join("./ImageFolder1",file))

    if img is not None:

        Output = segment_extreme_points(img)

        P1_hx, P1_hy = Output[0]
        P2_hx, P2_hy = Output[1]
        P1_vx, P1_vy = Output[2]
        P2_vx, P2_vy = Output[3]

        cv2.circle(np.uint8(img), (P1_vx, P1_vy), 4, (255, 0, 0, 1), -1)
        cv2.circle(np.uint8(img), (P2_vx, P2_vy), 4, (255, 0, 0, 1), -1)

        cv2.circle(np.uint8(img), (P1_hx, P1_hy), 4, (0, 0, 255, 1), -1)
        cv2.circle(np.uint8(img), (P2_hx, P2_hy), 4, (0, 0, 255, 1), -1)

        degree = "degrees rotated = "+str(Output[4])

        plt.imshow(img, cmap="gray")
        plt.title(degree)
        plt.show()



