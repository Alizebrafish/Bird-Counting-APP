import os

def fix_labels(label_dir):
    for label_file in os.listdir(label_dir):
        if label_file.endswith('.txt'):
            with open(os.path.join(label_dir, label_file), 'r') as f:
                lines = f.readlines()
            with open(os.path.join(label_dir, label_file), 'w') as f:
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id, x, y, w, h = map(float, parts)
                        if 0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1:
                            f.write(line)
                        else:
                            print(f"Correcting label in {label_file}: {line.strip()}")
                            x = min(max(x, 0), 1)
                            y = min(max(y, 0), 1)
                            w = min(max(w, 0), 1)
                            h = min(max(h, 0), 1)
                            f.write(f"{class_id} {x} {y} {w} {h}\n")
                    else:
                        print(f"Invalid format in {label_file}: {line.strip()}")

# Run the function for train and validation labels
fix_labels('C:/Users/user/Desktop/proj/Yolov9-birds/dataset/train/labels')
fix_labels('C:/Users/user/Desktop/proj/Yolov9-birds/dataset/valid/labels')
