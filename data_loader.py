import os
import nibabel as nib
import random
import numpy as np
import cv2
class Loader:
    def __init__(self, data_root, batch_size):
        # get list of training data, i.e. a list of 
        # nii files
        self.data_root = data_root
        self.batch_size = batch_size
        self.nii_list = self.__generate_nii_list()
    
    def normalize(self, img):
        img_o = img/255
        m = np.mean(img_o)
        s = np.std(img_o)
        return np.divide((img_o - m), s)
    
    def __generate_nii_list(self):
        nii_list = []
        folder_list = os.listdir(self.data_root)
        for folder in folder_list:
            if folder == 'patient001.Info.cfg':
                continue
            folder_path = os.path.join(self.data_root, folder)
            nii_4d = os.path.join(folder_path, folder + '_4d.nii.gz')
            if not os.path.exists(nii_4d):
                raise 'Wrong file address!'
            nii_list.append(nii_4d)

        return nii_list
            
    def get_batch(self):
        # generate batchs of image pairs, img1 and img2
        # random pick a patient and generate pairs of shape(batch, height, width)
        nii_file = random.choice(self.nii_list)
        img_dat = nib.load(nii_file)
        img = img_dat.get_data() #(x, y ,z ,t), uint8 mode
        img = self.normalize(img)
        img0 = np.zeros(shape=(self.batch_size, img.shape[0], img.shape[1]))
        img1 = np.zeros(shape=(self.batch_size, img.shape[0], img.shape[1]))
        idxs = range(img.shape[3])
        for i in range(self.batch_size):
            z = random.choice(range(img.shape[2]))
            pair = random.sample(idxs, 2)
            #print(pair, img.shape)
            img0[i, :, :] = img[:,:,z, pair[0]]
            img1[i, :, :] = img[:,:,z, pair[1]]
        return img0, img1





if __name__=='__main__':
    data_root = '/home/hanchao/CMR/training/'
    batch_size = 8
    loader = Loader(data_root, batch_size)
    image1, image2 = loader.get_batch()
    '''
    for test
    
    cv2.imshow('im1', image1[4,:,:])
    cv2.imshow('im2', image2[4,:,:])
    print(np.array_equal(image1[4,:,:], image2[4,:,:]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
