#Copies last trained weights into Yolobirds folder

import cv2
import numpy as np
import glob
import tqdm
import shutil
import os
poss_pathes = glob.glob('../Yolov9-birds/yolov9/runs/train/exp*')
poss_suff = [pspth.split('\\')[-1].split('exp')[-1] for pspth in poss_pathes]
poss_nums = [int(psnm) if psnm!='' and psnm.isdigit() else 0 for psnm in poss_suff]
sub_dir = poss_pathes[np.argmax(poss_nums)]
weights_path = os.path.join(sub_dir, 'weights', 'best.pt')
print("Selected subdirectory:", sub_dir)
print("Attempting to copy from:", weights_path)
print("File exists:", os.path.exists(weights_path))
shutil.copy2(weights_path , '../Yolov9-birds/')