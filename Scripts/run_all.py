#Runs evaluation on all videos separately

import os
import yaml
import torch
import sys

print(torch.cuda.is_available())
with open("config.yaml","r") as file_object:
    config = yaml.load(file_object,Loader = yaml.SafeLoader)
if config['weights'] == 'last':
    weights_pth = './Yolov9-birds/best.pt'

else:
    weights_pth = config['weights']
    weights_pth
# train
train_result = os.system(f"python ../Yolov9-birds/yolov9/train_dual.py --workers 8 --batch {config['batch_size']} --img {config['img_size']} --epochs {config['epochs']} --data ../Yolov9-birds/data.yaml --weights ../Yolov9-birds/best.pt --device 0 --cfg ../Yolov9-birds/yolov9.yaml --hyp ../Yolov9-birds/yolov9/data/hyps/hyp.scratch-high.yaml --project C:/Users/USER/Desktop/proj/Yolov9-birds/yolov9/runs/train --name exp_test --save-period 1")
if train_result != 0:
    print("Training failed, not attempting to copy weights.")
    sys.exit(1)
os.system("python copy_last_trained_weights.py")

# # run detection
# os.system(f"python ../Yolov9-birds/yolov9/detect.py --img {config['img_size']} --conf 0.1 --device 0 --save-txt --weights {weights_pth}  --source ../Yolov9-birds/split_data/video_test/images")


# # collect lables
# os.system("python collector.py")

# # convert to videos
# os.system("python convert_imgs_to_video.py")

# #Evaluate
# for i in [1,2,3,4,5,6]:
#     os.system(f"python ../Yolov9-birds/yolov9/val-mod.py --data ../Yolov9-birds/dataset/metric_test/m{str(i)}.yaml --weights {weights_pth} --batch-size {config['batch_size']}  --imgsz {config['img_size']} --task test --device 0 --single-cls --project ../output_metrics/ --name {str(i)} --exist-ok")