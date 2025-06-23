# Skin Tone Classification Prototype (Streamlit)

This prototype demonstrates a skeleton of the skin tone classification app, built with Streamlit. The app simulates model outputs and prepares for full integration with an image classification model and API.

## Directory Overview

- `scripts/app/app.py`: Streamlit frontend prototype
- `scripts/data/data_load.py`: Script to download images from Fitzpatrick17k
- `data/`: Organized images by skin tone (for training/testing)
- `data/fitzpatrick17k.csv`: Original dataset CSV with image URLs and labels
- `requirements.txt`: Python package requirements

## How to Run

1. Set up a virtual environment (optional):
```bash
python -m venv skinvenv
source skinvenv/bin/activate
