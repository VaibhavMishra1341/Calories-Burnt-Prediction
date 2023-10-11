# Calories Burnt Prediction Project 🏋️‍♂️

![maxresdefault](https://github.com/VaibhavMishra1341/Calories-Burnt-Prediction/assets/39896268/4352cdaf-444b-4a7e-8327-4281ce75194b)

This project aims to predict calories burned during exercise based on user data, leveraging various machine learning models and thorough data analysis to achieve accurate estimations.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Conclusion](#conclusion)

## Overview

The primary objective of this project is to estimate the number of calories burned during exercise sessions. We utilize advanced machine learning techniques to construct models that precisely predict calorie expenditure based on diverse user features.

## Machine Learning Models Used

### Decision Tree Regressor

![0 12CT72krFYLtGFGx](https://github.com/VaibhavMishra1341/Calories-Burnt-Prediction/assets/39896268/abded0b5-022f-42a8-9188-8f7100ba4b75)

1. **Tree-based Model**: Decision Tree Regressor is a supervised learning model that predicts the target variable by creating a decision tree based on the features. It splits the dataset into subsets based on feature values to make predictions.
2. **Simple Interpretation**: Decision trees are easy to understand and interpret, making them useful for showcasing the decision-making process in a straightforward manner, which is beneficial for explanatory purposes.
3. **Potential for Overfitting**: Decision trees can be prone to overfitting the training data, meaning they may capture noise in the data and not generalize well to unseen data. Techniques like pruning can help mitigate this issue.
4. **Useful for Nonlinear Relationships**: Decision trees are effective for capturing nonlinear relationships between features and the target variable, making them a valuable choice for diverse datasets.

### Random Forest Regressor

![0 4hfu8vepPsbjTBuH](https://github.com/VaibhavMishra1341/Calories-Burnt-Prediction/assets/39896268/261c606f-2797-4157-9ec5-cdc4d49eb296)

1. **Ensemble Learning**: The Random Forest Regressor is an ensemble learning method that constructs multiple decision trees during training and outputs the average prediction of the individual trees.
2. **Improved Accuracy and Robustness**: By aggregating predictions from multiple trees, the Random Forest Regressor typically offers higher accuracy and is more robust to overfitting compared to a single decision tree.
3. **Feature Importance**: Random Forest can provide a feature importance score, helping identify the most influential features in the prediction process, aiding in feature selection and understanding the dataset.
4. **Efficient for Large Datasets**: It is efficient for large datasets and can handle a large number of features and instances while maintaining good performance.

### Linear Regressor

![simple_regression](https://github.com/VaibhavMishra1341/Calories-Burnt-Prediction/assets/39896268/e7b13a90-6a43-4828-9d11-bb153d82be51)

1. **Linear Model**: Linear Regression is a fundamental regression technique that models the relationship between the target variable and predictors as a linear equation. It assumes a linear relationship between the features and the target.
2. **Interpretability and Simplicity**: Linear regression is easy to interpret, making it suitable for scenarios where understanding the impact of each feature on the target is important.
3. **Assumption of Linearity**: One limitation is that it assumes a linear relationship, which may not hold true for all datasets. It may not capture complex nonlinear relationships well.
4. **Efficiency and Scalability**: Linear regression is computationally efficient and can handle large datasets, making it practical for real-world applications with a significant amount of data.

### Support Vector Regressor (SVR)

![Support-Vector-Regression](https://github.com/VaibhavMishra1341/Calories-Burnt-Prediction/assets/39896268/65d3eeb2-bfc8-4872-8365-7dc8e03994b0)

1. **Kernel-based Regression**: SVR is a regression algorithm that utilizes the concepts of Support Vector Machines (SVM) for classification and extends them to regression. It uses kernels to map data into a higher-dimensional space.
2. **Effective for Nonlinear Data**: SVR is proficient in capturing nonlinear relationships between features and the target variable by transforming the data into a higher-dimensional space using various kernel functions.
3. **Robustness to Outliers**: SVR is less sensitive to outliers in the data due to the margin-based loss function, making it suitable for datasets with noise or irregularities.
4. **Hyperparameter Tuning**: SVR requires careful selection of hyperparameters, such as the kernel type and regularization parameter, to achieve optimal performance. Grid search and cross-validation are commonly used for hyperparameter tuning.

### XGBoost Regressor

![Bagging](https://github.com/VaibhavMishra1341/Calories-Burnt-Prediction/assets/39896268/25262657-75d1-44c8-9b8d-4dddf7ef46c0)

1. **Gradient Boosting Algorithm**: XGBoost (eXtreme Gradient Boosting) is an efficient and scalable gradient boosting algorithm that uses an ensemble of weak learners (usually decision trees) to make accurate predictions.
2. **Handling Nonlinearity and Interactions**: XGBoost excels in capturing nonlinear relationships and interactions between features, making it a powerful tool for complex datasets with intricate patterns.
3. **Regularization and Overfitting Control**: XGBoost incorporates regularization techniques like L1 and L2 regularization to prevent overfitting and enhance model generalization.
4. **High Performance and Speed**: XGBoost is known for its speed and performance due to parallelization and optimization strategies, making it a popular choice for various machine learning tasks.


## Dataset

The dataset encompasses exercise and user data, encompassing critical attributes such as age, height, weight, gender, and the corresponding calories burned during exercise.

## Data Preprocessing

- Combined exercise and user data based on common identifiers.
- Conducted data cleaning, addressing missing values, and transforming categorical variables into numerical representations.

![Merging Datasets](https://github.com/gaytrisran03/Calorie-Prediction/assets/78645392/9c1dca40-5d01-4493-a952-134422dc4b3b)


## Exploratory Data Analysis

We delved into a visual exploration of the dataset, utilizing various plots to discern data distribution, correlations, and trends. Key plots encompassed:

- Gender Distribution Count Plot 📊

    ![Gender Distribution Count Plot](https://github.com/gaytrisran03/Calorie-Prediction/assets/78645392/bac88bb9-304c-41ff-9d8c-3780f1d51f1e)

- Distribution Plots for:
  - Age 📈

      ![Age Distribution](https://github.com/gaytrisran03/Calorie-Prediction/assets/78645392/32e54149-6f67-4692-9369-422e10025db6)

  - Height 📈

      ![Height Distribution](https://github.com/gaytrisran03/Calorie-Prediction/assets/78645392/94c69c4e-a495-4828-960d-74859e916eed)

  - Weight 📈

      ![Weight Distribution](https://github.com/gaytrisran03/Calorie-Prediction/assets/78645392/a2852849-e027-46c3-afed-71637d27c5a6)

  - Correlation HeatMap 📈

    ![Correlation HeatMap](https://github.com/VaibhavMishra1341/Calories-Burnt-Prediction/assets/39896268/4e126c04-9c7a-4dfc-b2eb-cbee2140157e)
    

## Model Training

We employed various regression models, including the Decision Tree Regressor, Random Forest Regressor, Linear Regressor, Support Vector Regressor and XGBoost Regressor. The dataset was divided into training and testing sets for effective model training.

## Model Evaluation

Model performance was assessed using the Mean Absolute Error (MAE) as a crucial evaluation metric. Visual representations of the MAE for each model were included to provide insights into their respective performances. Accuracy is also calculated in order to get an all-around perspective of the model performance.

![Screenshot_80](https://github.com/VaibhavMishra1341/Calories-Burnt-Prediction/assets/39896268/a03ee447-a141-415f-8900-a0ae22babb56)

![Screenshot_79](https://github.com/VaibhavMishra1341/Calories-Burnt-Prediction/assets/39896268/230160cf-5af2-4142-bdf3-88aace04efc6)

## Conclusion

This project highlights the effective utilization of machine learning models in predicting calories burned during exercise. Continuous improvements and feature engineering can significantly enhance the accuracy of the models.

--- 

