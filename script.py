import os
import requests
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
from serpapi import GoogleSearch
import albumentations as A

# Set up directories
dataset_dir = "food_images"
os.makedirs(dataset_dir, exist_ok=True)

# SerpApi credentials
SERPAPI_KEY = "EPqiCw6L6NzEZbJKMNX7dGpo"

# Function to fetch image URLs
def get_image_urls(query, num_images=20):
    params = {
        "q": query,
        "tbm": "isch",
        "ijn": "0",
        "api_key": SERPAPI_KEY,
    }
    search = GoogleSearch(params)
    results = search.get()
    images = results.get("images_results", [])
    return [img["original"] for img in images[:num_images]]

# Function to download images
def download_images(food_items, num_images=20):
    image_data = []
    for food in food_items:
        folder = os.path.join(dataset_dir, food.replace(" ", "_"))
        os.makedirs(folder, exist_ok=True)
        urls = get_image_urls(food, num_images)
        
        for idx, url in enumerate(urls):
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    image_path = os.path.join(folder, f"{food.replace(' ', '_')}_{idx}.jpg")
                    with open(image_path, "wb") as f:
                        f.write(response.content)
                    image_data.append((image_path, food))
            except Exception as e:
                print(f"Failed to download {url}: {e}")
    return image_data

# Data augmentation functions
def add_noise(image):
    noise = np.random.randint(5, 50, image.shape, dtype='uint8')
    return cv2.add(image, noise)

def adjust_lighting(image):
    image = Image.fromarray(image)
    enhancer = ImageEnhance.Brightness(image)
    return np.array(enhancer.enhance(np.random.uniform(0.5, 1.5)))

def change_pixel_density(image):
    h, w = image.shape[:2]
    scale = np.random.uniform(0.5, 1.5)
    image = cv2.resize(image, (int(w * scale), int(h * scale)))
    return cv2.resize(image, (w, h))

def change_rgb_scale(image):
    transform = A.RGBShift(r_shift=30, g_shift=30, b_shift=30, p=1.0)
    return transform(image=image)['image']

def blur_image(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

def rotate_image(image):
    angle = np.random.uniform(-30, 30)
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
    return cv2.warpAffine(image, M, (w, h))

# Apply augmentations
def augment_images(image_data):
    augmented_data = []
    for image_path, label in image_data:
        image = cv2.imread(image_path)
        if image is None:
            continue
        
        augmentations = [add_noise, adjust_lighting, change_pixel_density, change_rgb_scale, blur_image, rotate_image]
        for aug in augmentations:
            augmented_image = aug(image.copy())
            aug_image_path = image_path.replace(".jpg", f"_{aug.__name__}.jpg")
            cv2.imwrite(aug_image_path, augmented_image)
            augmented_data.append((aug_image_path, label))
    return augmented_data

# Main execution
food_items = ["Tofu-Potato Hash", "Vegan Avocado Breakfast Toast", "Scrambled Eggs", "Crispy Tater Tots", "Bacon (2 slice)", "Honey Granola ( 1/4 cup)", "Balsamic Vinaigrette (2 tablespoon)", "Creamy Caesar Dressing (2 tablespoon)", "Lite Italian Dressing (2 tablespoon)", "Oil & Vinegar (2 tablespoon)", "Baby Spinach (1 cup)", "Fresh Fruit Salad ( 1/2 cup)", "Grape Tomatoes ( 1/4 cup)","Romaine Lettuce (2 cup)", "Shredded Carrots ( 1/4 cup)", "Sliced Mixed Bell Peppers ( 1/4 cup)", "Sliced Mushrooms ( 1/4 cup)", "Sliced Red Onions (1 slice)", "Bacon Pieces (2 tablespoon)", "Cottage Cheese ( 1/4 cup)", "Nonfat Strawberry Yogurt ( 1/4 cup)", "Fruity Fruit Loop® Bar", "Vegan Snickerdoodle", "Low Sodium Vegetable Broth", "Carolina BBQ Pulled Pork Sandwich", "Old Bay® Potato Chips",  ]  # Add your food items
image_data = download_images(food_items)
augmented_data = augment_images(image_data)

# Save dataset labels
all_data = image_data + augmented_data
df = pd.DataFrame(all_data, columns=["image_path", "label"])
df.to_csv(os.path.join(dataset_dir, "food_labels.csv"), index=False)

print("Dataset collection and augmentation complete!")