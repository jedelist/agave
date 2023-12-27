import os
import sys
import shutil
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from model_definition import create_colorization_model

def custom_image_generator(color_dir, grayscale_dir, batch_size, image_size=(256, 256)):
    color_images = sorted(os.listdir(color_dir))
    grayscale_images = sorted(os.listdir(grayscale_dir))

    while True:
        for i in range(0, len(color_images), batch_size):
            batch_color_imgs = color_images[i:i + batch_size]
            batch_gray_imgs = grayscale_images[i:i + batch_size]

            color_data = [img_to_array(load_img(os.path.join(color_dir, img), target_size=image_size)) for img in batch_color_imgs]
            grayscale_data = [img_to_array(load_img(os.path.join(grayscale_dir, img), target_size=image_size, color_mode='grayscale')) for img in batch_gray_imgs]

            yield (np.array(grayscale_data), np.array(color_data))

# Get model number from user and make directories
model_number = input("Enter the model number: ")
model_dir = os.path.join('./output/models', f'model{model_number}')
if os.path.exists(model_dir):
    shutil.rmtree(model_dir)
os.makedirs(model_dir, exist_ok=True)

train_log_path = os.path.join('./logs', 'train.log')

# Directory settings for data
DATA_DIR = './data'
COLOR_DIR = os.path.join(DATA_DIR, 'color')
GRAYSCALE_DIR = os.path.join(DATA_DIR, 'grayscale')
BATCH_SIZE = 2              # CHANGE if needed
EPOCHS = 10                 # CHANGE if needed

# Custom Data Generator
train_generator = custom_image_generator(COLOR_DIR, GRAYSCALE_DIR, BATCH_SIZE)

# Load and compile model
input_shape = (256, 256, 1)
model = create_colorization_model(input_shape)

# Callbacks measures
best_model_path = os.path.join(model_dir, "best_model.h5")
checkpoint_best = ModelCheckpoint(best_model_path, save_best_only=True, monitor='loss')

last_model_path = os.path.join(model_dir, "last_model.h5")
checkpoint_last = ModelCheckpoint(last_model_path, save_best_only=False, save_freq='epoch')

early_stopping_cb = EarlyStopping(patience=10, restore_best_weights=True)

# Redirect stdout to train.log
sys.stdout = open(train_log_path, 'w')

# Training the model
model.fit(train_generator, epochs=EPOCHS, steps_per_epoch=len(os.listdir(GRAYSCALE_DIR)) // BATCH_SIZE,
          callbacks=[checkpoint_best, checkpoint_last, early_stopping_cb])