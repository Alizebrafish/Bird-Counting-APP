import os

def correct_labels(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r') as file:
                lines = file.readlines()
            
            corrected_lines = []
            for line in lines:
                values = line.split()
                try:
                    class_id = int(values[0])
                    bbox_values = [float(v) for v in values[1:]]
                    
                    # Ensure no negative values
                    corrected_bbox_values = [max(0, v) for v in bbox_values]
                    corrected_line = f"{class_id} " + " ".join(map(str, corrected_bbox_values)) + "\n"
                    corrected_lines.append(corrected_line)
                except ValueError as e:
                    print(f"Removing invalid line in {file_path}: {line.strip()} ({e})")
            
            with open(file_path, 'w') as file:
                file.writelines(corrected_lines)

# Path to the directory containing label files
label_directory = "C:\\Users\\USER\\Desktop\\proj\\Yolov9-birds\\split_data\\train\\labels"
correct_labels(label_directory)
