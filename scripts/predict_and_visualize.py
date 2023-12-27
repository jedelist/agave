import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from shutil import copyfile

def load_and_preprocess_image(image_path, target_size):
    image = load_img(image_path, target_size=target_size, color_mode='grayscale')
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image / 255.0

def postprocess_prediction(prediction):
    prediction = (prediction + 1) * 127.5
    prediction = np.clip(prediction, 0, 255).astype('uint8')
    return prediction[0]

def visualize_results(grayscale, predicted, ground_truth):
    plt.figure(figsize=(12, 6))
    titles = ['Grayscale Image', 'Predicted Colorization', 'Ground Truth']

    for i, image in enumerate([grayscale, predicted, ground_truth]):
        plt.subplot(1, 3, i + 1)
        plt.imshow(image)
        plt.title(titles[i])
        plt.axis('off')
    plt.show()

def prepare_test_data(source_dir, target_dir, limit=50):            #No need for limit? Doesn't pull with API
    if len(os.listdir(target_dir)) < limit:
        for i, file in enumerate(os.listdir(source_dir)):
            if i >= limit:
                break
            source_file = os.path.join(source_dir, file)
            target_file = os.path.join(target_dir, file)
            copyfile(source_file, target_file)

def main():
    model_path = './output/models/model2/best_model.h5'         #CHANGE if needed and AUTOMATE
    test_images_dir = './data/test/test_images'                 #Probably will keep same
    test_ground_truth_dir = './data/test/test_ground_truth'     #Probably will keep same
    predictions_dir = './output/models/model2/predictions'      #CHANGE Automade and implement prompt for user asking for model number

    os.makedirs(predictions_dir, exist_ok=True)

    prepare_test_data('./data/color', test_ground_truth_dir)
    prepare_test_data('./data/grayscale', test_images_dir)

    model = load_model(model_path)

    for file in os.listdir(test_images_dir):
        grayscale_path = os.path.join(test_images_dir, file)
        ground_truth_path = os.path.join(test_ground_truth_dir, file)
        prediction_path = os.path.join(predictions_dir, file)

        grayscale_img = load_and_preprocess_image(grayscale_path, (256, 256))
        prediction = model.predict(grayscale_img)
        prediction = postprocess_prediction(prediction)

        ground_truth_img = load_img(ground_truth_path, target_size=(256, 256))

        plt.imsave(prediction_path, prediction)

        visualize_results(load_img(grayscale_path, target_size=(256, 256)), prediction, ground_truth_img)

if __name__ == '__main__':
    main()