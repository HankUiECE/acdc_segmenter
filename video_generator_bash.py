import nibabel as nib
import numpy as np
import cv2
import os
image_root = '/home/hanchao/acdc_segmenter/acdc_logdir/unet2D_bn_modified_wxent_bn_hanchao/predictions_testset/image/'
mask_root = '/home/hanchao/acdc_segmenter/acdc_logdir/unet2D_bn_modified_wxent_bn_hanchao/predictions_testset/prediction/'
video_root = '/home/hanchao/acdc_segmenter/demo/'
image_lst = os.listdir(image_root)
for k in range(len(image_lst)):
    name = image_lst[k]
    image_path = os.path.join(image_root, name)
    mask_path = os.path.join(image_root, name)
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
    video_file = os.path.join(video_root, 'demo' + str(k) + '.mp4')
    video = cv2.VideoWriter(video_file,fourcc,fps,(height,width))
    for i in range(n_files):
        file_name = 'frame_' + str(i) + '.nii.gz'
        img_file = os.path.join(image_path, file_name)
        msk_file = os.path.join(mask_path, file_name)
        img_arr = nib.load(img_file).get_data()
        msk_arr = nib.load(msk_file).get_data()
        img = img_arr[:,:,0].T
        msk = msk_arr[:,:,0].T
        #img = np.uint8(img)
        img_3ch = np.stack((img,img,img),axis=2)
        #colorize
        msk_b = msk.copy()
        msk_r = msk.copy()
        msk_g = msk.copy()
        msk_b[msk==1] = 255
        msk_g[msk==2] = 255
        msk_r[msk==3] = 255
        msk_color = np.stack((msk_b, msk_g, msk_r), axis=2)
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
