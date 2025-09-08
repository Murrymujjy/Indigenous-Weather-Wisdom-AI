# Indigenous-Weather-Wisdom-AI
This project is a machine learning solution to the "Ghana’s Indigenous Intel Challenge". The core task was to build a classification model to predict the type of rainfall—heavy, moderate, or small—expected in the next 12 to 24 hours.
The data for this challenge was uniquely collected using the Smart Indigenous Weather App, where trained Ghanaian farmers from the Pra River Basin submitted their weather forecasts based on Indigenous Ecological Indicators (IEIs) such as cloud formations, sun position, wind, and animal behavior.

My solution leverages a CatBoost Classifier, a model well-suited for the categorical nature of this data. The high performance was achieved through:

Advanced Feature Engineering: I created new, meaningful features like prediction_distance and Target-Guided Aggregation features (avg_distance_by_intensity and avg_distance_by_community).

Model Training: A single, well-tuned CatBoost model was trained on this enriched dataset, ultimately achieving a final score of 0.951640432.

This work not only provides an accurate predictive model but also contributes to the mission of validating traditional knowledge systems and merging them with modern AI to empower rural communities.
