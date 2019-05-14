import utils
from os import listdir
import os
import cv2 
from flow_est_warp import find_ED_ES_num
import numpy as np
def normalize(img, ratio=0.8):
    thres = np.percentile(img, ratio*100)
    thres = max(thres, 255)
    img[img>thres] = thres
    return img/float(thres) * 255.0

def to_png(img_4d, ED_frame, ES_frame, save_dir):
    num_slice = img_4d.shape[2]
    for z in range(num_slice):
        subdir = os.path.join(save_dir, 'slice' + str(z).zfill(2))
        if not os.path.exists(subdir):
            os.mkdir(subdir)
        video = img_4d[:,:,z,:]
        for frame in range(int(ED_frame)-1, int(ES_frame)):
            img = video[:,:,frame]
            #cv2.imshow('test', img.astype(np.uint8))
            #cv2.imshow('ori', img_4d[:,:,z,frame].astype(np.uint8))
            #print(np.max(img), np.max(img_4d[:,:,z,frame]), np.min(video[:,:,frame]))
            filename = 'frame' + str(frame).zfill(2) + '.png'
            filename = os.path.join(subdir, filename)
            #print(filename)
            #exit()
            #cv2.waitKey(0)
            img_rgb = cv2.cvtColor(img.astype(np.uint8),cv2.COLOR_GRAY2RGB)
            #print(img_rgb.shape, img_rgb.dtype)
            #exit()
            cv2.imwrite(filename, img_rgb)


def main():
    data_path = '/mnt/interns/hanchao/training'
    output_path = '/mnt/interns/hanchao/dataset_png'
    lst = listdir(data_path)
    for folder in lst:
        if folder == 'patient001.Info.cfg':
            continue 
        folder_path = os.path.join(data_path, folder)
        ED_frame, ES_frame = find_ED_ES_num(folder_path)
        nii_4d = os.path.join(folder_path, folder + '_4d.nii.gz')
        img_4d_dat = utils.load_nii(nii_4d)
        img_4d = img_4d_dat[0]
        #print(img_4d.shape)
        #exit()
        print('working on:', folder)
        if np.max(img_4d) > 255:
            #print(np.max(img_4d))
            img_4d = normalize(img_4d, ratio=0.995)
        save_dir = os.path.join(output_path, folder)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        to_png(img_4d, ED_frame, ES_frame, save_dir)

if __name__ == '__main__':
    main()