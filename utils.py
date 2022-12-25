import os
import shutil

def copy_images_to_directory(homer_no_homer_dir, images_to_copy, directory_to_paste):
    for image in images_to_copy:
        shutil.copy2(os.path.join(homer_no_homer_dir, image), directory_to_paste)