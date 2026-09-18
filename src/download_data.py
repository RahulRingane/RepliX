import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

DATASET = "thoughtvector/customer-support-on-twitter"
DOWNLOAD_DIR = "data/raw"

os.makedirs(DOWNLOAD_DIR, exist_ok=True)

api = KaggleApi()
api.authenticate()

print("Downloading dataset...")
api.dataset_download_files(
    DATASET,
    path=DOWNLOAD_DIR,
    unzip=True
)

print("Download complete.")

print("Files:")
for file in os.listdir(DOWNLOAD_DIR):
    print(" -", file)