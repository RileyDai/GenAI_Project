# MSAI 495 Assignment: Image Generation

This project implements a Variational Autoencoder (VAE) to learn and generate images.

## 📂 Project Structure

```text
├── data_apples/ # Apple Dataset root folder 
├── data_impressionism/ # Apple Dataset root folder 
├── logs/ # Training logs and checkpoints 
├── data.ipynb # Notebook for data processing and visualization
├── run_vae.py # VAE model architecture 
├── README.md 
└── environment.yml # Conda environment specification 
```

## Datasets

This project utilizes two datasets:

### 1. WikiArt Impressionism Subset

- Focuses on **Impressionism** style paintings.
- Categorized into 4 visual genres:
  - `landscape`
  - `genre_painting`
  - `portrait`
  - `cityscape`

### 2. Fruit360 Apple Subset

- A simpler dataset composed of Apple class images from [Fruit360](https://www.kaggle.com/datasets/moltean/fruits).
- Used as a **control experiment** to validate basic image generation capability of the model.
- This subset includes 3 apple categories:
  - `Apple Golden`
  - `Apple Golden-Red`
  - `Apple Red`

> **Why two datasets?**  
> During implementation, I found that **paintings require much finer detail and structure** to generate realistic results. In contrast, **Fruit360 provides more structured and lower-complexity data**, which helps verify whether the VAE is fundamentally capable of generating coherent outputs before applying it to complex styles like Impressionism.

## Installation

```bash
conda env create -f environment.yml
conda activate genai
```

## Train

```bash
python run_vae.py
```

## Test

```bash
python run_vae.py --inference \
                  --model_path ./logs/vae_*/vae_epoch_100.pt \
                  --n_samples 16
```
