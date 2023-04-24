# thanks ChatGPT

from PIL import Image
import os
from tqdm import tqdm

# set the path to your image folder
folder_path = "/home/jovyan/VGGFace2/VGG-Face2/data/test/"

# initialize min and max values
min_width, min_height = float('inf'), float('inf')
max_width, max_height = float('-inf'), float('-inf')

# count the number of images in the folder
num_images = sum(len(files) for _, _, files in os.walk(folder_path) if any(file.endswith(".jpg") or file.endswith(".png") for file in files))

# loop through all the files and folders in the specified path and update the progress bar
with tqdm(total=num_images) as pbar:
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith(".jpg") or filename.endswith(".png"):
                # open the image file
                img_path = os.path.join(root, filename)
                with Image.open(img_path) as img:
                    # get the width and height of the image
                    width, height = img.size
                    
                    # update min and max values
                    if width < min_width:
                        min_width = width
                    if height < min_height:
                        min_height = height
                    if width > max_width:
                        max_width = width
                    if height > max_height:
                        max_height = height
                
                # update the progress bar
                pbar.update(1)

# print the min and max values
print("Minimum width:", min_width)
print("Minimum height:", min_height)
print("Maximum width:", max_width)
print("Maximum height:", max_height)