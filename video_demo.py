import cv2
import os
import numpy as np
patient_id = '015'
image_path = '/home/hanchao/acdc_segmenter/cv2warp_hsv/' + 'patient' + patient_id

slice_num = 9 
height = 256
width = 216
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
fps = 2
video1  = cv2.VideoWriter('demo/demo_motion' + '.mp4',fourcc,fps,(height,width))
video2 = cv2.VideoWriter('demo/demo_hsv' + '.mp4',fourcc,fps,(height,width))

for i in range(slice_num):
    hsv_name = os.path.join(image_path, 'patient' + patient_id + 'slice' + str(i).zfill(2) + 'hsv.png')
    motion_name = os.path.join(image_path, 'patient' + patient_id + 'slice' + str(i).zfill(2) + '.png')
    #print(motion_name, os.path.exists(motion_name)) 
    #exit()
    hsv = cv2.imread(hsv_name)
    motion = cv2.imread(motion_name)
    print(hsv.dtype)
    print(motion.dtype)
    video1.write(motion)
    video2.write(hsv)
cv2.destroyAllWindows()
video1.release()
video2.release()

