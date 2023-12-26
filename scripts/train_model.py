import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model_definition import create_colorization_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Directory settings
DATA_DIR = './data'
COLOR_DIR = os.path.join(DATA_DIR, 'color')
GRAYSCALE_DIR = os.path.join(DATA_DIR, 'grayscale')
BATCH_SIZE = 32
EPOCHS = 100

# Prepare Data Generators
train_datagen = ImageDataGenerator(rescale=1. / 255)
color_datagen = train_datagen.flow_from_directory(COLOR_DIR, target_size=(256, 256), batch_size=BATCH_SIZE, class_mode=None)
grayscale_datagen = train_datagen.flow_from_directory(GRAYSCALE_DIR, target_size=(256, 256), batch_size=BATCH_SIZE, class_mode=None)

# Zip color and grayscale generators
def generate_data_generator(gen1, gen2):
    while True:
        X1i = gen1.next()
        X2i = gen2.next()
        yield X2i, X1i  # Return grayscale images as input and color images as targets

# Load and compile the model
input_shape = (256, 256, 1)
model = create_colorization_model(input_shape)

# Callbacks
checkpoint_cb = ModelCheckpoint("best_model.h5", save_best_only=True)
early_stopping_cb = EarlyStopping(patience=10, restore_best_weights=True)

# Training the model
train_generator = generate_data_generator(color_datagen, grayscale_datagen)
model.fit(train_generator, epochs=EPOCHS, steps_per_epoch=len(grayscale_datagen), callbacks=[checkpoint_cb, early_stopping_cb])
