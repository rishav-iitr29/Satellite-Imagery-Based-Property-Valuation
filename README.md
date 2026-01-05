# Satellite-Imagery-Based-Property-Valuation
### Overview

This project explores whether satellite imagery provides additional predictive signal beyond traditional tabular features for residential property valuation. Using a combination of structured housing attributes and overhead satellite images, we build and evaluate multimodal regression models to predict house prices.  
The primary goal is not only accuracy, but also interpretability — understanding what visual cues (e.g., greenery, water proximity, neighborhood layout) influence model predictions.

### Project structure
.\
├── data/\
│   ├── raw/                                # Original tabular datasets (given in PS)\
│   ├── processed/                          # Preprocessed NumPy arrays\
│   └── images/                             # Satellite images (PNG)\
├── scripts/\
│   ├── data_fetcher.py                     # Downloads satellite images using lat/long\
│   ├── test_predictions.py                 # Script to generate final predictions using the Late Fusion model\
│   └── preprocessing.ipynb                 # Contains EDA, transformation of raw data and baseline (XGBoost) model\
├── multimodal_fusion_model/\
│   ├── final_multimodal_model.ipynb        # Late Fusion model\
│   └── model_output/\
│       └── best_multimodal_model.pth       # Best training checkpoint to load the model\
├── residual_model/\
│   ├── residual_model.ipynb                # Residual CNN model\
│   └── model_output/\
│       └── best_residual_multimodal.pth    # Best training checkpoint to load the model\
├── ViT_model/\
│   ├── ViT_model.ipynb                     # Vision Transformer model\
│   └── model_output/\
│       └── best_ViT_model.pth              # Best training checkpoint to load the model\
├── final_predictions.csv                   # Final Test set predictions\
├── final_report.pdf                        # Final Project report\
└── README.md\
