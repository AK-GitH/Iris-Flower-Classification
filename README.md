# Iris Flower Classification using Python

This project demonstrates how to **classify Iris flowers** into species (`Setosa`, `Versicolor`, `Virginica`) using features like **sepal length**, **petal width**, and more. It employs a **Random Forest Classifier**, a robust ensemble learning method, to learn the pattern from historical data and accurately predict the species of a new flower.

---

## Project Description

> **Iris classification** is a classic supervised learning problem, where the goal is to categorize Iris flowers based on measurable features like petal and sepal dimensions.

This project processes the Iris dataset (`Iris.csv`), cleans it, and builds a machine learning model that can classify a flower into one of three species.

---

## Files Included

- `Iris.csv` — Raw dataset  
- `cleaning_data.py` — Script to clean the raw data:  
  - Drops unnecessary columns like `Id`  
  - Removes duplicates and outliers  
  - Handles missing values  
  - Converts `Species` to a categorical format  
  - Saves the cleaned dataset as `cleaned_iris.csv`  
- `iris_flower_predictor.py` — Machine learning pipeline to:  
  - Encode species labels  
  - Train a Random Forest Classifier  
  - Evaluate using Accuracy, Precision, Recall, and F1 Score  
  - Visualize the Confusion Matrix and Feature Importances

---

### 1. Install Required Packages

Make sure Python 3.x is installed, then install the required packages:

```bash
pip install pandas matplotlib seaborn scikit-learn
```

### 2. Run the `cleaning_data.py` which will clean the data and then run `iris_flower_predictor.py`