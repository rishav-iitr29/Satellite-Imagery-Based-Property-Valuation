import pandas as pd
import requests
import os
from tqdm import tqdm

# CONFIG
MAPBOX_TOKEN = "pk**********SQ" # Mapbox access token
SAVE_DIR = "../data/images/train/images/" # "../data/images/test/images/" for test data images
ZOOM_LEVEL = 18 # Using zoom = 18
IMAGE_SIZE = "600x600" # will resize later for model input

def download_satellite_image(row):
    img_id = row['id']
    lat = row['lat']
    lon = row['long']
    
    file_path = os.path.join(SAVE_DIR, f"{img_id}.png")

    if os.path.exists(file_path):
        return
    
    # Mapbox Static Image URL
    url = f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/{lon},{lat},{ZOOM_LEVEL},0/{IMAGE_SIZE}?access_token={MAPBOX_TOKEN}"
    
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                f.write(response.content)
    except Exception as e:
        print(f"Error downloading {img_id}: {e}")


if __name__ == "__main__":
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        
    # Load your training/test data
    df = pd.read_csv("../data/raw/train(1).csv") # "../data/raw/test(1).csv" for test dataset
    
    print(f"Starting download for {len(df)} properties...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        download_satellite_image(row)

