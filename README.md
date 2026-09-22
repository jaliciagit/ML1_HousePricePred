# House Price Prediction using Linear Regression

## Project Overview

This project is a Machine Learning application that predicts house prices using **Multiple Linear Regression**.

The model is trained using the Boston Housing dataset. Different characteristics of a house and its surrounding area are used as input features to predict the median value of the house.

The trained model is integrated with a **Streamlit web application**, allowing users to enter property details and receive a predicted house price through a simple interface.

---

## Objectives

- Understand the workflow of a regression-based Machine Learning project.
- Perform data exploration and preprocessing.
- Train a Multiple Linear Regression model.
- Evaluate the model using regression metrics.
- Build a simple interactive web application using Streamlit.
- Deploy the Machine Learning workflow in a user-friendly interface.

---

## Technologies Used

- **Python**
- **Pandas** – Data manipulation and analysis
- **NumPy** – Numerical computations
- **Scikit-learn** – Machine Learning model and evaluation
- **Matplotlib** – Data visualization
- **Seaborn** – Data visualization
- **SciPy** – Statistical analysis
- **Streamlit** – Web application

---

## Machine Learning Model

The project uses **Multiple Linear Regression**.

The model uses all columns except `MEDV` as input features.

### Target Variable

`MEDV` – Median value of owner-occupied homes.

### Input Features

The dataset contains the following features:

| Feature | Description |
|---|---|
| CRIM | Per capita crime rate |
| ZN | Proportion of residential land |
| INDUS | Proportion of non-retail business areas |
| CHAS | Charles River dummy variable |
| NOX | Nitric oxide concentration |
| RM | Average number of rooms |
| AGE | Proportion of older buildings |
| DIS | Distance to employment centres |
| RAD | Accessibility to radial highways |
| TAX | Property tax rate |
| PTRATIO | Pupil-teacher ratio |
| B | Proportion related to the population measure |
| LSTAT | Percentage of lower-status population |

---

## Project Workflow

The project follows these main steps:

1. Load the dataset.
2. Explore the dataset.
3. Check for missing values and duplicates.
4. Perform descriptive statistical analysis.
5. Analyse correlations between variables.
6. Visualize important relationships.
7. Separate features and target variable.
8. Split the dataset into training and testing sets.
9. Standardize the input features.
10. Train the Multiple Linear Regression model.
11. Generate predictions.
12. Evaluate model performance.
13. Perform residual and statistical analysis.
14. Integrate the model into a Streamlit application.

---

## Model Evaluation

The regression model is evaluated using:

- **MAE (Mean Absolute Error)**
- **MSE (Mean Squared Error)**
- **RMSE (Root Mean Squared Error)**
- **R² Score**
- **Adjusted R² Score**

These metrics are used to understand how accurately the model predicts house prices.

---

## Streamlit Application

The Streamlit application provides an interactive interface where users can enter the values of the house features.

After entering the required values, the application uses the Linear Regression model to generate a predicted house price.

The application trains the model directly from `boston_housing.csv` when it starts.

Therefore, a separate `.pkl` model file is not required.

---

## Project Structure

```text
ML1_HousePricePred/
│
├── app.py
├── boston_housing.csv
├── requirements.txt
├── README.md
├── .gitignore
└── .venv/
