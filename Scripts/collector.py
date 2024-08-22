import numpy as np
import pandas as pd
import cv2
import glob
import os
import tqdm
import yaml

case_1_shape = (1080, 1920)
case_2_shape = (2160, 3840)
case_3_shape = (2160, 3840)
case_4_shape = (2160, 3840)
case_5_shape = (1070, 1912)
case_6_shape = (2160, 3840)

with open("config.yaml","r") as file_object:
    yaml_file = yaml.load(file_object,Loader=yaml.SafeLoader)

    #splits_dir = yaml_file['collect_label_path']

    poss_pathes = glob.glob('../Yolov9-birds/yolov9/runs/detect/exp*')
    poss_suff = [pspth.split('\\')[-1].split('exp')[-1] for pspth in poss_pathes]
    poss_nums = [int(psnm) if psnm!='' else 0 for psnm in poss_suff]

    splits_dir = poss_pathes[np.argmax(poss_nums)] + '/labels'


    col_save_dir = yaml_file['final_label_path']



case_shapes = [case_1_shape,case_2_shape,case_3_shape,
               case_4_shape,case_5_shape,case_6_shape]

new_dim = 500

labels_pathes = glob.glob(f'{splits_dir}/*')
labels_names = ['_'.join(os.path.basename(labels_path).replace('.txt','').split('_')[:-2]) for labels_path in labels_pathes]
labels_names = np.unique(labels_names)


# all images loop

for labels_name in labels_names:
    gw = 0
    # Divide loop
    labels = None
    
    
    # image specific parameters
    shp = case_shapes[int(float(labels_name.split('_')[0]))-1]
    height = shp[0]
    width = shp[1]


    w_num = width//new_dim + 1    # number of splits in width direction
    h_num = height//new_dim + 1    # number of splits in height direction

    gap_w = (w_num*new_dim - width)//(w_num-1)
    gap_h = (h_num*new_dim - height)//(h_num-1)

    for i in range(0,w_num):
        gh = 0
        for j in range(0,h_num):
            if not os.path.isfile(f'{splits_dir}/{labels_name}_{i}_{j}.txt'):
                gh+=gap_h
                continue
            labels_file = np.loadtxt(f'{splits_dir}/{labels_name}_{i}_{j}.txt')
            #reverse yolov9
            if labels_file.shape[0]==0:
                gh+=gap_h
                continue
                
            elif len(labels_file.shape)==1:
                labels_file = labels_file.reshape((1,-1))

            elif labels_file.shape[1]==0:
                gh+=gap_h
                continue
                
 
            nxc,nyc,nw,nh = labels_file[:,1],labels_file[:,2],labels_file[:,3],labels_file[:,4]
            
            xc = nxc*new_dim
            yc = nyc*new_dim
            w =  nw*new_dim
            h =  nh*new_dim

            x1 = xc-(w/2)
            x2 = xc+(w/2)
            y1 = yc-(h/2)
            y2 = yc+(h/2)
            
            label = np.stack((x1,y1,x2,y2)).T
            label[:,::2] = label[:,::2]   + (i*new_dim-gw)  # adjust new split label x dimension
            label[:,1::2] = label[:,1::2] + (j*new_dim-gh)  # adjust new split label y dimension

            #convert to yolov9
            x1,y1,x2,y2 = label[:,0],label[:,1],label[:,2],label[:,3]
            w = x2-x1
            h = y2-y1
            xc = (x1+x2)/2
            yc = (y1+y2)/2

            nxc = xc/width
            nyc = yc/height
            nw = w/width
            nh = h/height
            
            new_label = np.stack(([0]*nxc.shape[0],nxc,nyc,nw,nh)).T
            labels = np.concatenate((labels,new_label),axis=0) if labels is not None else new_label
            
            
            gh+=gap_h
        gw+=gap_w
    np.savetxt(f'{col_save_dir}/{labels_name}.txt',labels,fmt='%.8f')

def get_iou(bb1, bb2):

    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    return iou


iou_th = 0.5

labels_pathes = glob.glob(f'{col_save_dir}/*')
for label_path in labels_pathes:
    labels_file = np.loadtxt(label_path)
    bad_iou = np.zeros(labels_file.shape[0])

    for i_b,b1 in enumerate(labels_file):
        b1 = b1[1:]
        bb1 = {'x1':b1[0]-b1[2]/2,'y1':b1[1]-b1[3]/2,'x2':b1[0]+b1[2]/2,'y2':b1[1]+b1[3]/2}
        for j_b,b2 in enumerate(labels_file):
            b2 = b2[1:]
            if bad_iou[j_b]!=1:
                if i_b!=j_b:
                    bb2 = {'x1':b2[0]-b2[2]/2,'y1':b2[1]-b2[3]/2,'x2':b2[0]+b2[2]/2,'y2':b2[1]+b2[3]/2}
                    iou = get_iou(bb1, bb2)
                    if iou > iou_th:
                        bad_iou[i_b] = 1
                        break

    labels_file = labels_file[~bad_iou.astype('bool')]
    np.savetxt(label_path,labels_file,fmt='%.8f')
