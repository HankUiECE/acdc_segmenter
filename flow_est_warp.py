import numpy as np
import nibabel as nib 
import cv2
from os import listdir
import os
import glob
import utils
import image_utils 
from shutil import copyfile
def warp_flow(img, flow, interpo=cv2.INTER_LINEAR):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, interpo)
    return res

def draw_flow(img, flow, step=8):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    #(np.max(vis), np.min(vis))
    #cv2.imshow('im1', vis)
    #vis2 = np.stack((img, img, img), axis=2)
    #vis = vis2
    #print(vis.dtype, vis2.dtype)
    #print(np.array_equal(vis, vis2+1))
    #print(np.max(vis2), np.min(vis2))
    #exit()
    #cv2.imshow('im2', vis)
    #cv2.waitKey(0)
    #()
    #print(vis.shape)
    #exit()
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    #print(vis.shape)
    return vis
    #return np.transpose(vis, (1,0,2))
    
def find_ED_ES_num(path):
    infos = {}
    #print(path)
    #exit()
    for line in open(os.path.join(path, 'Info.cfg')):
        label, value = line.split(':')
        infos[label] = value.rstrip('\n').lstrip(' ')
    ED_frame = infos['ED'].zfill(2)
    ES_frame = infos['ES'].zfill(2)
    return ED_frame, ES_frame

def normalize(img):
    return img /float(np.max(img)) * 255.0

def draw_hsv(flow):
    h, w = flow.shape[:2]
    #fx, fy = flow[:,:,0], flow[:,:,1]
    #ang = np.arctan2(fy, fx) + np.pi
    #v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    #hsv[...,2] = np.minimum(v*4, 255)
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def warp_volume(ED_file, ED_gtfile, ES_file):
    ED_dat = utils.load_nii(ED_file)
    ED_gt = utils.load_nii(ED_gtfile)
    ES_dat = utils.load_nii(ES_file)
    out_affine = ED_dat[1]
    out_header = ED_dat[2]

    ED_img = normalize(ED_dat[0])
    ED_gt_img = ED_gt[0]
    ES_img = normalize(ES_dat[0])
    ES_warped = np.zeros(ED_img.shape)
    ED_warped = np.zeros(ED_img.shape)
    motion_list = []
    hsv_list = []
    #print(np.max(ED_img), np.min(ED_img))
    for z in range(ED_img.shape[2]):
        img_ed = np.flip(ED_img[:,:,z], 0)
        img_ed = np.transpose(img_ed, (1,0))
        img_es = np.flip(ES_img[:,:,z], 0)
        img_es = np.transpose(img_es, (1,0))
        #cv2.imshow('fliped', img_ed.astype(np.uint8))
        flow = cv2.calcOpticalFlowFarneback(ED_img[:,:,z].astype(np.uint8), ES_img[:,:,z].astype(np.uint8), None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow_vis = cv2.calcOpticalFlowFarneback(img_ed.astype(np.uint8), img_es.astype(np.uint8), None, 0.5, 3, 15, 3, 5, 1.2, 0)
        #flow = cv2.calcOpticalFlowFarneback(ES_img[:,:,z].astype(np.uint8), ED_img[:,:,z].astype(np.uint8), None, 0.5, 3, 15, 3, 5, 1.2, 0)
        ES_warped[:,:,z] = warp_flow(ED_gt_img[:,:,z],flow, interpo=cv2.INTER_NEAREST)
        ED_warped[:,:,z] = warp_flow(ED_img[:,:,z], flow)
        #cv2.imshow('ES', ES_warped[:,:,z].astype(np.uint8))
        #cv2.imshow('ED', ED_warped[:,:,z].astype(np.uint8))
        #print(np.max(ED_warped))
        #exit()
        #flow = cv2.calcOpticalFlowFarneback((ED_img[:,:,z].T).astype(np.uint8), (ES_img[:,:,z].T).astype(np.uint8), None, 0.5, 3, 15, 3, 5, 1.2, 0)
        #print(flow.shape)
        vis = draw_flow(img_ed.astype(np.uint8),-flow_vis)
        #vis = np.flip(vis, 0)
        hsv = draw_hsv(flow_vis)
        #hsv = np.flip(hsv, 0)
        #cv2.imshow('hsv', np.transpose(hsv, (0,1,2)))
        #cv2.imshow('img_ed', img_ed.astype(np.uint8))
        #cv2.imshow('img_es', img_es.astype(np.uint8))
        #print(np.max(hsv), np.min(hsv))
        #print(np.max(vis))
        #exit()
        #cv2.namedWindow("flow", cv2.WINDOW_NORMAL)
        #print(ED_file, z)
        #cv2.namedWindow('flow',cv2.WINDOW_NORMAL)
        #cv2.resizeWindow('flow', 600,600)
        #cv2.imshow('flow', np.transpose(vis, (0,1,2)))
        #cv2.imshow('flow2',np.transpose(vis, (1,0,2))[100:,100:])
        #cv2.waitKey(0)
        #exit()
        motion_list.append(vis)
        hsv_list.append(hsv)
    return ES_warped, ED_warped, out_affine, out_header, motion_list, hsv_list

def warp_volume_4d(nii_4d, ED_frame, ES_frame, ED_gtfile):
    img_4d_dat = utils.load_nii(nii_4d)
    img_4d_img = img_4d_dat[0]
    #print(img_4d_img.shape)
    #exit()
    ED_gt_dat = utils.load_nii(ED_gtfile)
    ED_gt_img = ED_gt_dat[0]
    out_affine = ED_gt_dat[1]
    out_header = ED_gt_dat[2]
    ES_warped = np.zeros(ED_gt_img.shape)

    for z in range(img_4d_img.shape[2]):
        cur_frame = img_4d_img[:,:,z,int(ED_frame)-1]
        cur_gt = ED_gt_img[:,:,z]
        for t in range(int(ED_frame)-1, int(ES_frame)-1):
            next_frame = img_4d_img[:,:,z,t + 1]
            #print(cur_frame.dtype, next_frame.dtype, np.max(cur_frame), np.min(cur_frame))
            #exit()
            flow = cv2.calcOpticalFlowFarneback(cur_frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            cur_frame = warp_flow(cur_frame, flow)
            cur_gt = warp_flow(cur_gt, flow)
        ES_warped[:,:,z] = cur_gt
    
    return ES_warped, out_affine, out_header

def main():
    data_path = '/home/hanchao/CMR/training/'
    output_path = 'cv2warp_hsv'
    lst = listdir(data_path)
    for folder in lst:
        if folder == 'patient001.Info.cfg':
            continue 
        folder_path = os.path.join(data_path, folder)
        ED_frame, ES_frame = find_ED_ES_num(folder_path)
        ED_file = os.path.join(folder_path, folder + '_frame' + ED_frame + '.nii.gz')
        ES_file = os.path.join(folder_path, folder + '_frame' + ES_frame + '.nii.gz')
        ED_gtfile = os.path.join(folder_path, folder + '_frame' + ED_frame + '_gt.nii.gz')
        ES_gtfile = os.path.join(folder_path, folder + '_frame' + ES_frame + '_gt.nii.gz')
        nii_4d = os.path.join(folder_path, folder + '_4d.nii.gz')
        
        #warp ED to ES using opencv
        warpedES, warpedED, out_affine, out_header, motion_list, hsv_list = warp_volume(ED_file, ED_gtfile, ES_file)
        #warpedES, out_affine, out_header = warp_volume_4d(nii_4d, ED_frame, ES_frame, ED_gtfile)
        save_folder = os.path.join(output_path, folder)
        #print(len(motion_list))
        #exit()
        for i in range(len(motion_list)):
            filename = os.path.join(save_folder, folder + 'slice' + str(i).zfill(2) + '.png')
            #print(filename)
            #exit()
            cv2.imwrite(filename, motion_list[i])
            filename = os.path.join(save_folder, folder + 'slice' + str(i).zfill(2) + 'hsv.png')
            cv2.imwrite(filename, hsv_list[i])
            #img = motion_list[i]
            #print(np.max(img), np.min(img))
            #exit()
            #cv2.imshow('flow', motion_list[i])
            #cv2.waitKey(0)
            #exit()
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        save_file_es = os.path.join(save_folder, folder + '_frame' + ES_frame + '_gt.nii.gz')
        save_file_ed = os.path.join(save_folder, folder + '_frame' + ES_frame + '.nii.gz')
        utils.save_nii(save_file_es, warpedES, out_affine, out_header)
        utils.save_nii(save_file_ed, warpedED, out_affine, out_header) 
        dst_ed = os.path.join(save_folder, folder + '_frame' + ED_frame + '.nii.gz')
        dst_ed_gt = os.path.join(save_folder, folder + '_frame' + ED_frame + '_gt.nii.gz')
        dst_es = os.path.join(save_folder, folder + '_frame' + ES_frame + 'ori_es.nii.gz')
        dst_es_gt = os.path.join(save_folder, folder + '_frame' + ES_frame + '_gtgt.nii.gz')
        copyfile(ED_file, dst_ed)
        copyfile(ES_file, dst_es)
        copyfile(ED_gtfile, dst_ed_gt)
        copyfile(ES_gtfile, dst_es_gt)
if __name__ == '__main__':
    main()
