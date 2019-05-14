import cv2 
import numpy as np
import image_utils
import os
import utils
from shutil import copyfile
from flow_est_warp import find_ED_ES_num

data_path = '/home/hanchao/CMR/training/'
out_path = 'test'

lst = os.listdir(data_path)
for folder in lst:
    if folder == 'patient001.Info.cfg':
            continue 
    folder_path = os.path.join(data_path, folder)
    ED_frame, ES_frame = find_ED_ES_num(folder_path)
    nii_4d = os.path.join(folder_path, folder + '_4d.nii.gz')
    img_4d_dat = utils.load_nii(nii_4d)
    img_4d = img_4d_dat[0].astype(np.uint8)
