import cv2
import numpy as np



def angle_rotation(img_rot):


    img_unrot = cv2.imread('base_image.png')

    h,w,c = img_unrot.shape


    img_resize = cv2.resize(img_rot,(w,h))


    gray = cv2.cvtColor(np.uint8(img_resize), cv2.COLOR_RGB2GRAY)

    gray1 = cv2.cvtColor(np.uint8(img_unrot), cv2.COLOR_RGB2GRAY)


    center = (w//2,h//2)

    score_list = []


    allowed_rotation1 = np.linspace(0,45,46)
    allowed_rotation2 = np.linspace(170,190,21)
    allowed_rotation3 = np.linspace(350,360,11)

    allowed_rotation = np.concatenate((allowed_rotation1, allowed_rotation2, allowed_rotation3), axis = None)

    for i in allowed_rotation:

        rotation_matrix = cv2.getRotationMatrix2D(center, i, 1.0)

        final_rotated = cv2.warpAffine(gray1,rotation_matrix,(w,h))



        #(score, diff) = compare_ssim(image_compare,final_rotated , full=True)

        diff_img = cv2.absdiff(gray, final_rotated)
        score = np.sum(diff_img)

        score_list.append(score)

    angle_index = np.argmin(score_list)

    return allowed_rotation[angle_index], h, w


