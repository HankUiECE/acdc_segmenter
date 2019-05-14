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
for folder in [lst[0]]:
    folder_path = os.path.join(data_path, folder)
    ED_frame, ES_frame = find_ED_ES_num(folder_path)
    nii_4d = os.path.join(folder_path, folder + '_4d.nii.gz')
    nii_3d_ed = os.path.join(folder_path, folder + '_frame' + ED_frame + '.nii.gz')
    nii_3d_ed_GT = os.path.join(folder_path, folder + '_frame' + ED_frame + '_gt.nii.gz')
    img_4d_dat = utils.load_nii(nii_4d)
    ed_dat = utils.load_nii(nii_3d_ed)
    out_affine = ed_dat[1]
    out_header = ed_dat[2]
    
    #img_4d = image_utils.normalise_image(img_4d_dat[0])
    #ed = image_utils.normalise_image(ed_dat[0])
    img_4d = img_4d_dat[0].astype(np.uint8)
    ed = ed_dat[0].astype(np.uint8)
    #print(np.max(ed_dat[0]), np.min(ed_dat[0]))
    #exit()

out_file1  = os.path.join(out_path, 'ed_4d.nii.gz')
out_file2 = os.path.join(out_path, 'ed_org.nii.gz')
out_file3 = os.path.join(out_path, 'ed_gt.nii.gz')
#utils.save_nii(out_file1, img_4d[:,:,:,0], out_affine, out_header)
#utils.save_nii(out_file2, ed, out_affine, out_header)
#copyfile(nii_3d_ed_GT, out_file3)
cv2.imshow('ed from 4d', img_4d[:,:,3, int(ED_frame)])
cv2.imshow('ed original', ed[:,:,3])
cv2.waitKey(0)
cv2.destroyAllWindows()
