import os
import sys
import time
import requests
from skimage import color, io, transform

# Unsplash API settings
API_KEY = 'YbznoIz1FAPwz5RMWJiRrQZKXw3n5tAR7vi6-6sJWfU'  # Access key
API_URL = 'https://api.unsplash.com/photos/random'
QUERY_PARAMS = {'query': 'nature', 'count': 10}  # Query parameters

# Directory settings
DATA_DIR = './data'
COLOR_DIR = os.path.join(DATA_DIR, 'color')
GRAYSCALE_DIR = os.path.join(DATA_DIR, 'grayscale')
IMAGE_SIZE = 256  # Resize to 256x256 pixels
MAX_REQUESTS_PER_HOUR = 50  # Maximum number of requests per hour
SLEEP_TIME = 3600 / MAX_REQUESTS_PER_HOUR  # Time to sleep between requests

def download_image(image_url, target_folder):
    response = requests.get(image_url, stream=True)
    if response.status_code == 200:
        image = io.imread(response.content)
        return image
    return None

def save_image(image, filename, target_folder):
    filepath = os.path.join(target_folder, filename)
    io.imsave(filepath, image)

def preprocess_image(image):
    image_resized = transform.resize(image, (IMAGE_SIZE, IMAGE_SIZE), anti_aliasing=True)
    image_lab = color.rgb2lab(image_resized)
    grayscale_image = image_lab[:, :, 0]
    grayscale_image = (grayscale_image - grayscale_image.min()) / (grayscale_image.max() - grayscale_image.min())
    return image_resized, grayscale_image

def main():
    log_file = open('./logs/preparation.log', 'w')
    sys.stdout = log_file

    os.makedirs(COLOR_DIR, exist_ok=True)
    os.makedirs(GRAYSCALE_DIR, exist_ok=True)

    total_downloaded = 0  # Track the number of downloaded images

    while total_downloaded < MAX_REQUESTS_PER_HOUR:
        response = requests.get(API_URL, headers={'Authorization': f'Client-ID {API_KEY}'}, params=QUERY_PARAMS)
        if response.status_code == 200:
            images = response.json()
            for idx, img in enumerate(images):
                image = download_image(img['urls']['regular'], COLOR_DIR)
                if image is not None:
                    color_image, grayscale_image = preprocess_image(image)
                    save_image(color_image, f'color_{total_downloaded + idx}.jpg', COLOR_DIR)
                    save_image(grayscale_image, f'grayscale_{total_downloaded + idx}.jpg', GRAYSCALE_DIR)
                    print(f'Downloaded and processed image {total_downloaded + idx + 1}')

            total_downloaded += len(images)
            time.sleep(SLEEP_TIME)  # Sleep to avoid surpassing the rate limit
        else:
            print(f'Error fetching images: {response.status_code}')
            break  # Exit if there's an error in fetching images

    log_file.close()

if __name__ == '__main__':
    main()