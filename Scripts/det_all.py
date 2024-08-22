#Runs Detection without train 

# run detection
import os
import yaml

with open("config.yaml","r") as file_object:
    config = yaml.load(file_object,Loader = yaml.SafeLoader)
if config['weights'] == 'last':
    weights_pth = '../Yolov9-birds/best.pt'

else:
    weights_pth = config['weights']


os.system(f"python ../Yolov9-birds/yolov9/detect.py --img {config['img_size']} --conf 0.027 --device 0 --save-txt --weights {weights_pth}  --source ../Yolov9-birds/dataset/video_test/images")

# convert to videos
os.system("python convert_imgs_to_video.py")

