import cv2
import numpy as np
from compare_pix1 import angle_rotation
from itertools import groupby



def segment_extreme_points(img):

    rotation_output = angle_rotation(img)

    theta = np.radians(rotation_output[0])

    gray = cv2.cvtColor(np.uint8(img), cv2.COLOR_RGB2GRAY)

    thresh = cv2.threshold(np.uint8(gray), np.mean(gray), 255, cv2.THRESH_BINARY)[1]

    # find the biggest contour in the image

    cnt = sorted(cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[-2], key=cv2.contourArea)[-1]

    # Access the image pixels location of the biggest contour

    pts = [[],[]]

    A = cnt[:,:,0]

    B = cnt[:,:,1]

    pts[0] = np.ndarray.flatten(A)
    pts[1] = np.ndarray.flatten(B)

    # set up the transformation matrix (rotation) and the inverse of the matrix.

    rot_mat = np.array(((np.cos(theta), -np.sin(theta)), (np.sin(theta), np.cos(theta))))

    rot_mat_inv = np.array(((np.cos(theta), np.sin(theta)), (-np.sin(theta), np.cos(theta))))

    # Transform the contour's pixels in rotated co-ordinates

    pts1 = np.dot(rot_mat, [pts[0], pts[1]])

    pts_intX = [int(i) for i in pts1[0]]
    pts_intY = [int(i) for i in pts1[1]]

    # find y co-ordinates for same value of x co-ordinates using group by

    A = zip(pts_intX,pts_intY)

    sorted_zipped_lists = sorted(A)

    sorted_list1 = [element for _, element in sorted_zipped_lists]
    sorted_list = sorted(pts_intX)

    A1 = zip(sorted_list, sorted_list1)

    key_func = lambda x:x[0]

    groupingX = groupby(A1, key_func)




    def translationX():

        temp = 0
        temp_x = 0
        temp_y = [0,0]

        for key, group in groupingX:

            list_same_x = list(group)


            y_max = np.max(list_same_x, axis=0)[1]
            y_min = np.min(list_same_x, axis=0)[1]

            max_diff = y_max - y_min


            if (max_diff) > temp:
                temp = max_diff
                temp_x = key
                temp_y = [y_min , y_max]


         # Get Inverse of the transformed co-ordinates


        point1 = np.dot(rot_mat_inv, [temp_x, temp_y[0]])
        point2 = np.dot(rot_mat_inv, [temp_x, temp_y[1]])

        horizontal_intX = [int(i) for i in point1]
        horizontal_intY = [int(i) for i in point2]

        point_list = [horizontal_intX, horizontal_intY ]
        max_length = ((int(point_list[0][0]) - int(point_list[1][0])) ** 2 + (
                    int(point_list[0][1]) - int(point_list[1][1])) ** 2) ** 0.5

        return max_length, point_list


    # find x co-ordinates for same value of y co-ordinates using group by

    A2 = zip(pts_intY, pts_intX)

    sorted_zipped_lists = sorted(A2)

    sorted_list1 = [element for _, element in sorted_zipped_lists]
    sorted_list = sorted(pts_intY)

    A3 = zip(sorted_list1, sorted_list)

    key_func1 = lambda x:x[1]

    groupingY = groupby(A3, key_func1)




    def translationY():

        # Move along (x,y) vector (y-axis) and find the greatest difference in x co-ordinates

        temp = 0
        temp_x = [0, 0]
        temp_y = 0

        for key, group in groupingY:

            list_same_y = list(group)

            x_max = np.max(list_same_y, axis=0)[0]
            x_min = np.min(list_same_y, axis=0)[0]

            max_diff = x_max - x_min

            if (max_diff) > temp:
                temp = max_diff
                temp_y = key
                temp_x = [x_min, x_max]

        # Get Inverse of the transformed co-ordinates

        point1 = np.dot(rot_mat_inv, [temp_x[0], temp_y])
        point2 = np.dot(rot_mat_inv, [temp_x[1], temp_y])

        vertical_intX = [int(i) for i in point1]
        vertical_intY = [int(i) for i in point2]

        point_list = [vertical_intX, vertical_intY]

        max_length = ((int(point_list[0][0]) - int(point_list[1][0])) ** 2 + (
                int(point_list[0][1]) - int(point_list[1][1])) ** 2) ** 0.5

        return max_length, point_list


    Points_V = translationX()
    Points_H = translationY()


    P1_vx = int(Points_V[1][0][0])

    P1_vy = int(Points_V[1][0][1])

    P2_vx = int(Points_V[1][1][0])

    P2_vy = int(Points_V[1][1][1])


    P1_hx = int(Points_H[1][0][0])

    P1_hy = int(Points_H[1][0][1])

    P2_hx = int(Points_H[1][1][0])

    P2_hy = int(Points_H[1][1][1])


    return (P1_hx, P1_hy), (P2_hx, P2_hy), (P1_vx, P1_vy), (P2_vx, P2_vy), rotation_output[0]

