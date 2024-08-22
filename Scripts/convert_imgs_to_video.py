#Converts last run of detection into videos

import cv2
import numpy as np
import glob
import tqdm
import os


labels_path = '../Yolov9-birds/split_data/inf_collected_labels/'
images_path = '../Yolov9-birds/dataset/video_test/images/'
#poss_pathes = glob.glob('../Yolov9-birds/yolov9/runs/detect/exp*')
#poss_suff = [pspth.split('\\')[-1].split('exp')[-1] for pspth in poss_pathes]
#poss_nums = [int(psnm) if psnm!='' else 0 for psnm in poss_suff]

#sub_dir = poss_pathes[np.argmax(poss_nums)]
test_videos = [1,2,3,4,5,6]

for v in test_videos:
    #path = sub_dir + '\\' + str(v)+'*.*'
    path = images_path +  str(v)+'*.*'

    pathes = glob.glob(path)
    pathes.sort()

    h,w,_ = cv2.imread(pathes[0]).shape
    out = cv2.VideoWriter(f'../output_videos/output_video_{str(v)}.avi',cv2.VideoWriter_fourcc(*'DIVX'), 30, (w,h))

    for filename in tqdm.tqdm(pathes):
        img = cv2.imread(filename)
        lbl_name = os.path.basename(filename).split('.')[0] + '.txt'
        label_file = np.loadtxt(labels_path + lbl_name)
        for b1 in label_file:
            b1 = b1[1:]
            bb1 = {'x1':w*(b1[0]-b1[2]/2),'y1':h*(b1[1]-b1[3]/2),'x2':w*(b1[0]+b1[2]/2),'y2':h*(b1[1]+b1[3]/2)}
            cv2.rectangle(img, (int(bb1['x1']),int(bb1['y1'])), (int(bb1['x2']),int(bb1['y2'])), (0,0,255), 2) 
        out.write(img)
    
    out.release()