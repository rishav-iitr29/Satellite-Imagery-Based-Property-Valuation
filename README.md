# Satellite-Imagery-Based-Property-Valuation
### Overview

This project explores whether satellite imagery provides additional predictive signal beyond traditional tabular features for residential property valuation. Using a combination of structured housing attributes and overhead satellite images, we build and evaluate multimodal regression models to predict house prices.  
The primary goal is not only accuracy, but also interpretability and understanding what visual cues (e.g., greenery, water proximity, neighborhood layout) influence model predictions.

### Important

I saved all the .ipynb files with their outputs, so you can check out epoch-wise metrics. Also do care to check the file paths (although I adjusted them mostly) if running any script.\
Download the best model checkpoints from here - [Huggingface Repository](https://huggingface.co/d3nji/Satellite-Imagery-Based-Property-Valuation/tree/main)


### Project structure
![File Structure](/misc/project_str.png)

### Data Description

#### Tabular Data
	•	Source: Kaggle House Sales Dataset (King County)
	•	Key features:
	   •	price (target)
	   •	sqft_living, sqft_lot, bedrooms, bathrooms
	   •	sqft_living15, sqft_lot15 (neighborhood density)
     •	grade, condition, view, waterfront
	   •	latitude, longitude

The target variable is log-transformed (log(price + 1)) to address right-skew and stabilize training.

#### Visual Data

Satellite images are fetched using latitude and longitude coordinates for each property, providing environmental context not explicitly present in tabular data.


### Image Acquisition Pipeline

Satellite imagery is downloaded using the Mapbox Static Images API.

Key parameters:

	•	Zoom level: 18
	•	Resolution: 400 × 400
	•	Map style: Satellite
	•	Image format: PNG

Images are later resized to 224 × 224 and normalized using ImageNet statistics before being fed into CNN models.


### Modeling Approaches

Three modeling strategies were explored:

1. Tabular Baseline (XGBoost)

A strong gradient-boosting model trained purely on structured features, serving as a performance baseline.

2. Late Fusion Multimodal Model (Final Model)\
	•	Tabular branch: Multi-layer perceptron (MLP)\
	•	Visual branch: Pre-trained ResNet-50\
	•	Fusion: Concatenation followed by a regression head\
	•	Loss: Mean Squared Error (log-price)

This model achieved the best overall performance and is used for final predictions.

3. Diagnostic Models\
	•	Residual CNN: Predicts correction over XGBoost predictions\
	•	Vision Transformer (ViT): Patch-based image representation

These models were used for analysis and explainability, not as final predictors.

### Running the project

1. Start with the preprocessing.ipynb
2. Fetch the images using data_fetcher.py
3. train the model using .ipynb in a model's folder and visualise the result using the GradCAM in that notebook as well
4. use test_predictions.py to predict the prices for test set (converted in dollars)

### Results Summary

![result table](/misc/result_table.png)

The multimodal Late Fusion model consistently outperformed the tabular baseline, demonstrating that satellite imagery adds measurable predictive value.\
(DO read the Report for detailed conclusions)

### Notes & Limitations
	•	Satellite imagery provides incremental, not dominant, improvements
	•	Gains are modest due to inherent noise in real estate pricing
	•	Grad-CAM visualizations are qualitative and should not be interpreted causally

