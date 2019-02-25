import nibabel as nib
import numpy as np
import cv2
import os
def img_norm(img):
    if np.max(img) < 255:
        return np.int16(img)
    return np.int16(img/np.max(img) * 255.0)
patient_id = '080'
slice_num =4
image_path = '/home/hanchao/acdc_segmenter/acdc_logdir/unet2D_bn_modified_wxent_bn_hanchao/predictions_testset/image/patient' + patient_id
mask_path = '/home/hanchao/acdc_segmenter/acdc_logdir/unet2D_bn_modified_wxent_bn_hanchao/predictions_testset/prediction/patient' + patient_id
lst = os.listdir(image_path)
n_files = len(lst)
file_name = 'frame_0.nii.gz'
img_file = os.path.join(image_path, file_name)
shape = img_arr = nib.load(img_file).get_data().shape
height = shape[0] *2
width = shape[1]
fps = 2
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
#video = cv2.VideoWriter('demo1.avi',cv2.VideoWriter_fourcc('M','J','P','G'),fps,(height,width))2video = cv2.VideoWriter('demo1.mp4',fourcc,fps,(height,width))
video = cv2.VideoWriter('demo/demo' + patient_id + '.mp4',fourcc,fps,(height,width))
for i in range(n_files):
    file_name = 'frame_' + str(i) + '.nii.gz'
    img_file = os.path.join(image_path, file_name)
    msk_file = os.path.join(mask_path, file_name)
    img_arr = nib.load(img_file).get_data()
    msk_arr = nib.load(msk_file).get_data()
    img = img_arr[:,:,slice_num].T
    msk = msk_arr[:,:,slice_num].T
    #img_3ch = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    img = img_norm(img)
    img_3ch = np.stack((img,img,img),axis=2)
    #print(np.max(img_3ch), np.min(img_3ch))
    #colorize
    msk_b = msk.copy()
    msk_r = msk.copy()
    msk_g = msk.copy()
    msk_b[msk==1] = 255
    msk_g[msk==2] = 255
    msk_r[msk==3] = 255
    msk_color = np.int16(np.stack((msk_b, msk_g, msk_r), axis=2))
    blended_img = cv2.addWeighted(img_3ch,0.8,msk_color,0.2,0)
    #print(blended_img.shape)
    img_towrite = np.uint8(np.concatenate((img_3ch, blended_img),axis=1))
    #img_3ch = np.uint8(img_3ch)
    #print(np.max(img_3ch), np.min(img_3ch))
    #print(img_towrite.shape) 
    video.write(img_towrite)
    #video.write(img_3ch)
cv2.destroyAllWindows()
video.release()
