#Runs evakuation on all videos separately

import os



for i in [1,2,3,4,5,6]:
    os.system(f"python ../Yolov9-birds/yolov9/val-mod.py --data ../Yolov9-birds/dataset/metric_test/m{str(i)}.yaml --weights ../Yolov9-birds/best.pt --batch-size 8  --imgsz 640 --task test --device 0 --single-cls --project ../output_metrics/ --name {str(i)} --exist-ok")