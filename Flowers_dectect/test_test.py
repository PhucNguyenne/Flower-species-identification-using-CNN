import os
import shutil
import random

def copy_flower_subset(source_dir, dest_dir, num_images=9):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    for flower_type in os.listdir(source_dir):
        source_flower_path = os.path.join(source_dir, flower_type)
        dest_flower_path = os.path.join(dest_dir, flower_type)
        
        if os.path.isdir(source_flower_path):
            os.makedirs(dest_flower_path, exist_ok=True)
            images = [f for f in os.listdir(source_flower_path) if os.path.isfile(os.path.join(source_flower_path, f))]
            
            selected_images = random.sample(images, min(num_images, len(images)))
            
            for img in selected_images:
                shutil.copy(os.path.join(source_flower_path, img), os.path.join(dest_flower_path, img))
                
            print(f"Copied {len(selected_images)} images to {dest_flower_path}")

# Thay đổi đường dẫn này thành thư mục chứa tập dữ liệu gốc
source_directory = "jpeg-224x224/train"
# Thay đổi đường dẫn này thành thư mục đích
destination_directory = "jpeg-224x224/test1"

copy_flower_subset(source_directory, destination_directory)
