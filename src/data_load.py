import requests
import os
import pandas as pd

root_string = os.path.abspath(os.path.join(os.getcwd(), ".."))
dataset_path_string = os.path.join(root_string, "MSc - Research Project", "data", "fitzpatrick17k.csv")

df = pd.read_csv(dataset_path_string)

for u in df['fitzpatrick_scale'].unique():

    save_path_string = os.path.join(root_string, "MSc - Research Project", "data", str(u))
    os.makedirs(save_path_string, exist_ok=True)

# Loop through each row and download the image
for idx, row in df.iterrows():
    image_url = row["url"]
    image_id = row["md5hash"] if "md5hash" in row else idx
    image_folder = row["fitzpatrick_scale"]
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }
        response = requests.get(image_url, headers=headers, timeout=10)
        response.raise_for_status()
        file_ext = 'jpg'

        # Define full path to save image
        save_image_string = os.path.join(root_string, "MSc - Research Project", "data", str(image_folder))
        image_path = os.path.join(save_image_string, f"{image_id}.{file_ext}")
        with open(image_path, "wb") as f:
            f.write(response.content)

        print(f"Downloaded: {image_path}")

    except Exception as e:
        print(f"Failed to download {image_url}: {e}")