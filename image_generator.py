import numpy as np
import os
import tensorflow as tf

# Pycharm project directory:
# project_dir = 'C:\\Users\\RodrigoRoman\\PycharmProjects\\Homer_Face_Detector\\Images'
project_dir = 'C:\\Users\\rodri\\PycharmProjects\\Homer_Face_Detector\\Images'

input_shape = (640, 480, 3)

model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 300x300 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fifth convolution
    # tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    # tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('horses') and 1 for the other ('humans')
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])



# Apply data augmentation
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')


# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        os.path.join(project_dir, "train"),  # This is the source directory for training images
        target_size=input_shape[:2],  # All images will be resized to 300x300
        batch_size=int(56/2),
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')


validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

# Flow training images in batches of 128 using train_datagen generator
validation_generator = validation_datagen.flow_from_directory(
        os.path.join(project_dir, 'validation'),  # This is the source directory for training images
        target_size=input_shape[:2],  # All images will be resized to 300x300
        batch_size=12,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

history = model.fit(
      train_generator,
      steps_per_epoch=int(56/(56/2)),
      epochs=20,
      verbose=1,
      validation_data = validation_generator,
      validation_steps=12/12)


my_image = tf.keras.preprocessing.image.load_img(os.path.join(os.path.join(project_dir, 'test\homer'), my_image[0]), target_size=input_shape[:2])

x = tf.keras.preprocessing.image.img_to_array(my_image)
x = np.expand_dims(x, axis=0)

model.predict(x)