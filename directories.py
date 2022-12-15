import numpy as np
import os
import shutil

# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

faces_path = 'D:\Data Science\Projects\Homer_detector\Faces'

base_dir = 'D:\Data Science\Projects\Homer_detector'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

train_homer_dir = os.path.join(train_dir, 'homer')
train_no_homer_dir = os.path.join(train_dir, 'no_homer')

validation_homer_dir = os.path.join(validation_dir, 'homer')
validation_no_homer_dir = os.path.join(validation_dir, 'no_homer')

train_percent = 0.80
train_size = int(len(os.listdir(os.path.join(faces_path, 'Homer'))) * train_percent)

len(os.listdir(os.path.join(faces_path, 'Homer')))
print(train_size)

rand_homer_list = np.random.RandomState(0).choice(os.listdir(os.path.join(faces_path, 'Homer')), len(os.listdir(os.path.join(faces_path, 'Homer'))))
rand_no_homer_list = np.random.RandomState(0).choice(os.listdir(os.path.join(faces_path, 'No_Homer')), len(os.listdir(os.path.join(faces_path, 'No_Homer'))))

rand_homer_train_list = rand_homer_list[:train_size]
rand_homer_val_list = rand_homer_list[train_size:]

rand_no_homer_train_list = rand_no_homer_list[:train_size]
rand_no_homer_val_list = rand_no_homer_list[train_size:]

# print(f'Homer train list sample {rand_homer_train_list[:5]}')
# print(f'Homer validation list sample {rand_homer_val_list[:5]}')
# print(f'No Homer train list sample {rand_no_homer_train_list[:5]}')
# print(f'No Homer validation list sample {rand_no_homer_val_list[:5]}')
#
# print(f'Homer train list size {len(rand_homer_train_list)}')
# print(f'Homer validation list size {len(rand_homer_val_list)}')
# print(f'No Homer train list size {len(rand_no_homer_train_list)}')
# print(f'No Homer validation list size {len(rand_no_homer_val_list)}')


src = os.path.join(faces_path, rand_homer_train_list[0])
print(src)

dst = train_homer_dir
print(dst)

dst = 'D:\Data Science/Projects/Homer_detector/train/homer'
dst = r'C:\Users\RodrigoRoman\PycharmProjects\Homer_Face_Detector\Delete'

os.mkdir(dst)

shutil.copy2(src, dst)


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

