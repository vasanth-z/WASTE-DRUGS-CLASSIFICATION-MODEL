from PIL import Image
import os

def remove_corrupt_images(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(("jpg", "jpeg", "png")):
                try:
                    img_path = os.path.join(root, file)
                    with Image.open(img_path) as img:
                        img.verify()  # Check for corruption
                except (IOError, SyntaxError):
                    print(f"Removing corrupt image: {img_path}")
                    os.remove(img_path)



# Run for both train and validation directories
remove_corrupt_images("C:/Users/Staff/Downloads/Garbage_CNN_Model/waste-dataset/Dataset/train")
remove_corrupt_images("C:/Users/Staff/Downloads/Garbage_CNN_Model/waste-dataset/Dataset/val")
