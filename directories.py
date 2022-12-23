import numpy as np
import os
import shutil

# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# All images of the faces are here:
faces_path = 'D:\Data Science\Projects\Homer_detector\Faces'

# Homer directory:
homer_dir = os.path.join(faces_path, 'Homer')

# No_Homer directory:
no_homer_dir = os.path.join(faces_path, 'No_Homer')

# This is the base directory:
base_dir = 'D:\Data Science\Projects\Homer_detector'

# Pycharm project directory:
project_dir = 'C:\\Users\\RodrigoRoman\\PycharmProjects\\Homer_Face_Detector\\Images'

# Training and validation directories will be in the Pycharm project directory
train_dir = os.path.join(project_dir, 'train')
validation_dir = os.path.join(project_dir, 'validation')
test_dir = os.path.join(project_dir, 'test')

# Training and validation directories for images with and without target (homer's face)
train_homer_dir = os.path.join(train_dir, 'homer')
train_no_homer_dir = os.path.join(train_dir, 'no_homer')

validation_homer_dir = os.path.join(validation_dir, 'homer')
validation_no_homer_dir = os.path.join(validation_dir, 'no_homer')

test_homer_dir = os.path.join(test_dir, 'homer')
test_no_homer_dir = os.path.join(test_dir, 'no_homer')

# Our defined training proportion will be 70%, validation will be 15% and test will be 15%:
train_percent = 0.70
validation_percent = 0.15
test_percent = 1 - (train_percent - validation_percent)

dataset_size_per_class = len(os.listdir(homer_dir))

train_size = int(dataset_size_per_class * train_percent)
val_size = int(dataset_size_per_class * validation_percent)
test_size = int(dataset_size_per_class * test_percent)

# We are just checking out if training proportion worked out well, to double check it, uncomment the following 2 lines: We can very well delete this later
# print(f'Total number of homer face images: {len(os.listdir(os.path.join(faces_path, "Homer")))}')
# print(f'Training dataset size: {train_size}')

#We are randomizing the list of file names for each directory
rand_homer_list = np.random.RandomState(0).choice(os.listdir(homer_dir), len(os.listdir(homer_dir)))
rand_no_homer_list = np.random.RandomState(0).choice(os.listdir(no_homer_dir), len(os.listdir(no_homer_dir)))

#Training and validation randomized lists for both directories (with and without homer)
rand_homer_train_list = rand_homer_list[:train_size]
rand_homer_val_list = rand_homer_list[train_size:(train_size + val_size)]
rand_homer_test_list = rand_homer_list[(train_size + val_size):]

rand_no_homer_train_list = rand_no_homer_list[:train_size]
rand_no_homer_val_list = rand_no_homer_list[train_size:(train_size + val_size)]
rand_no_homer_test_list = rand_no_homer_list[(train_size + val_size):]


# Now we want to create the directory structure within the Pycharm project directory and copy the images there.
directories = [train_homer_dir,
               validation_homer_dir,
               test_homer_dir,
               train_no_homer_dir,
               validation_no_homer_dir,
               test_no_homer_dir]

# If the directories do not exist, create them:
for directory in directories:
    if not os.path.isdir(directory):
        # print(directory)
        os.makedirs(directory)

# For each image in each randomized group, copy it into it's corresponding directory:
for rand_homer_train_image, rand_homer_val_image, rand_homer_test_image, rand_no_homer_train_image, rand_no_homer_val_image, rand_no_homer_test_image in zip(rand_homer_train_list,
                                                                                                                                                             rand_homer_val_list,
                                                                                                                                                             rand_homer_test_list,
                                                                                                                                                             rand_no_homer_train_list,
                                                                                                                                                             rand_no_homer_val_list,
                                                                                                                                                             rand_no_homer_test_list):

    shutil.copy2(os.path.join(homer_dir, rand_homer_train_image), train_homer_dir)
    shutil.copy2(os.path.join(homer_dir, rand_homer_val_image), validation_homer_dir)
    shutil.copy2(os.path.join(homer_dir, rand_homer_test_image), test_homer_dir)
    shutil.copy2(os.path.join(no_homer_dir, rand_no_homer_train_image), train_no_homer_dir)
    shutil.copy2(os.path.join(no_homer_dir, rand_no_homer_val_image), validation_no_homer_dir)
    shutil.copy2(os.path.join(no_homer_dir, rand_no_homer_test_image), test_no_homer_dir)




def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

