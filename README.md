# Satellite-Imagery-Based-Property-Valuation
### Overview

This project explores whether satellite imagery provides additional predictive signal beyond traditional tabular features for residential property valuation. Using a combination of structured housing attributes and overhead satellite images, we build and evaluate multimodal regression models to predict house prices.
The primary goal is not only accuracy, but also interpretability — understanding what visual cues (e.g., greenery, water proximity, neighborhood layout) influence model predictions.

### Project structure
.
├── data_fetcher.py              # Downloads satellite images using lat/long
├── preprocessing.ipynb          # EDA, feature engineering, and data preparation
├── model_training.ipynb         # Multimodal model training & evaluation
├── gradcam_visualization.ipynb  # Grad-CAM explainability analysis
├── data/
│   ├── raw/                     # Original tabular datasets
│   ├── processed/               # Preprocessed NumPy arrays
│   └── images/                  # Satellite images (PNG)
├── final_submission.csv         # Test set predictions
├── README.md
