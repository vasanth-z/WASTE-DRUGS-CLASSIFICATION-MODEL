import os
from PIL import Image


def remove_empty_images(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            img_path = os.path.join(root, file)
            if os.path.getsize(img_path) == 0:
                print(f"Removing empty image: {img_path}")
                os.remove(img_path)

remove_empty_images("C:/Users/Staff/Downloads/Garbage_CNN_Model/waste-dataset/Dataset/train")
remove_empty_images("C:/Users/Staff/Downloads/Garbage_CNN_Model/waste-dataset/Dataset/val")
