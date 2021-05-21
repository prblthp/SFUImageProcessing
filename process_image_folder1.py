import csv
import cv2
import os
from segment_CT_abdomen1 import segment_extreme_points
from time import process_time

import sys

#start_time = process_time()


with open('segment_abdomen_extremes.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["ImageId", "HorizontalX1", "HorizontalY1", "HorizontalX2", "HorizontalY2", "VerticalX1",
                     "VerticalY1", "VerticalX2", "VerticalY2"])

    i = 1

    #folder_name = sys.argv[1]
    folder_name = "./ImageFolder1"


    for image in os.listdir(folder_name):

        img = cv2.imread(os.path.join(folder_name,image))

        if img is not None:
            output = segment_extreme_points(img)
            writer.writerow([i, output[0][0], output[0][1], output[1][0], output[1][1], output[2][0], output[2][1],
                            output[3][0], output[3][1]])
        i+=1


#end_time = process_time()

print(".csv file is ready.")
#print(end_time-start_time)


