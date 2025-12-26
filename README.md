# Income Prediction Project

This project implements a machine learning model to predict income levels based on various demographic and employment-related features using the Adult Income dataset.

## Project Structure

- `adult.csv`: The dataset containing demographic and employment information
- `python.ipynb`: Jupyter notebook with exploratory data analysis
- `train_model.py`: Script for training and saving the income prediction model
- `requirements.txt`: List of Python package dependencies
- `models/`: Directory containing saved model and encoders

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. To train the model:
```bash
python train_model.py
```
This will:
- Load and preprocess the data from `adult.csv`
- Train a Random Forest Classifier
- Save the trained model and label encoders to the `models` directory
- Display model performance metrics

2. To explore the data analysis:
- Open `python.ipynb` in Jupyter Notebook:
```bash
jupyter notebook
```

## Dataset Features

The Adult Income dataset includes the following features:
- age: Age of individual
- workclass: Type of employment
- fnlwgt: Final weight (sampling feature)
- education: Education level
- educational-num: Numerical education level
- marital-status: Marital status
- occupation: Occupation category
- relationship: Family relationship
- race: Race category
- gender: Gender
- capital-gain: Capital gains
- capital-loss: Capital losses
- hours-per-week: Hours worked per week
- native-country: Country of origin
- income: Target variable (<=50K or >50K)