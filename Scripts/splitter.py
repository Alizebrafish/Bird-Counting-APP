import numpy as np
import pandas as pd
import cv2
import glob
import os
import tqdm
import yaml

# Read paths from config files
with open("config.yaml", "r") as file_object:
    yaml_file = yaml.load(file_object, Loader=yaml.SafeLoader)
    img_dir = yaml_file['img_path']
    label_dir = yaml_file['label_path']
    save_path_img = yaml_file['split_img_path']
    save_path_lbl = yaml_file['split_label_path']

# Image specific parameters
new_dim = 500

image_pathes = glob.glob(f'{img_dir}/*')
label_pathes = glob.glob(f'{label_dir}/*')

# Commenting out the splitting process
# All images loop
for image_path in image_pathes:
    img = cv2.imread(image_path)
    
    height, width, _ = img.shape
    w_num = width // new_dim + 1    # number of splits in width direction
    h_num = height // new_dim + 1    # number of splits in height direction

    gap_w = (w_num * new_dim - width) // (w_num - 1)
    gap_h = (h_num * new_dim - height) // (h_num - 1)

    img_name = os.path.basename(image_path).replace('.png', '')
    
    if label_pathes is not None:
        for label_path in label_pathes:
            if os.path.basename(label_path).replace('.txt', '') == img_name:
                labels_file = np.loadtxt(label_path)
                break

        # Reverse YOLO format
        if len(labels_file.shape) == 1:
            labels_file = labels_file.reshape((1, -1))
        elif len(labels_file.shape) == 0:
            gh += gap_h
            continue

        nxc, nyc, nw, nh = labels_file[:, 1], labels_file[:, 2], labels_file[:, 3], labels_file[:, 4]

        xc = nxc * width
        yc = nyc * height
        w = nw * width
        h = nh * height

        x1 = xc - (w / 2)
        x2 = xc + (w / 2)
        y1 = yc - (h / 2)
        y2 = yc + (h / 2)

        label = np.stack((x1, y1, x2, y2)).T

    s_labels = []
    gw = 0
    # Divide loop
    for i in range(0, w_num):
        gh = 0
        for j in range(0, h_num):
            s_img = img[j * new_dim - gh:(j + 1) * new_dim - gh, i * new_dim - gw:(i + 1) * new_dim - gw]
            
            if label_pathes is not None:
                
                s_label = label.copy().astype(float)
                s_label[:, ::2] = s_label[:, ::2] - (i * new_dim - gw)  # adjust new split label x dimension
                s_label[:, 1::2] = s_label[:, 1::2] - (j * new_dim - gh)  # adjust new split label y dimension

                # Filter out out-of-split labels
                xcond = np.logical_and(s_label[:, 2] > 0, s_label[:, 0] < new_dim)   
                ycond = np.logical_and(s_label[:, 3] > 0, s_label[:, 1] < new_dim)

                s_label = s_label[np.logical_and(xcond, ycond)]

                # Convert to YOLO format
                x1, y1, x2, y2 = s_label[:, 0], s_label[:, 1], s_label[:, 2], s_label[:, 3]
                w = x2 - x1
                h = y2 - y1
                xc = (x1 + x2) / 2
                yc = (y1 + y2) / 2

                nxc = xc / new_dim
                nyc = yc / new_dim
                nw = w / new_dim
                nh = h / new_dim

                new_label = np.stack(([0] * nxc.shape[0], nxc, nyc, nw, nh)).T
                np.savetxt(f'{save_path_lbl}/{img_name}_{i}_{j}.txt', new_label, fmt='%.8f')
                
            cv2.imwrite(f'{save_path_img}/{img_name}_{i}_{j}.png', s_img)

            gh += gap_h

        gw += gap_w


print("Skipping the splitting process as the data is already split.")
