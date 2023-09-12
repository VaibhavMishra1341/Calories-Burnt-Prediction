import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics

# loading the data from csv file to a Pandas DataFrame
calories = pd.read_csv('calories.csv')

# print the first 5 rows of the dataframe
calories.head()

exercise_data = pd.read_csv('exercise.csv')

exercise_data.head()

calories_data = pd.concat([exercise_data, calories['Calories']], axis=1)

calories_data.head()

# checking for missing values
calories_data.isnull().sum()

# get some statistical measures about the data
calories_data.describe()

sns.set()

# plotting the gender column in count plot
sns.countplot(calories_data['Gender'])

# finding the distribution of "Age" column
sns.distplot(calories_data['Age'])

# finding the distribution of "Height" column
sns.distplot(calories_data['Height'])

# finding the distribution of "Weight" column
sns.distplot(calories_data['Weight'])

correlation = calories_data.corr()

# constructing a heatmap to understand the correlation

plt.figure(figsize=(10, 10))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f',
            annot=True, annot_kws={'size': 8}, cmap='Blues')

calories_data.replace({"Gender": {'male': 0, 'female': 1}}, inplace=True)

calories_data.head()

X = calories_data.drop(columns=['User_ID', 'Calories'], axis=1)
Y = calories_data['Calories']

print(X)

print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

# Decision Tree Regressor

model1 = DecisionTreeRegressor()
model1.fit(X_train, Y_train)

# Random Forest Regressor

model2 = RandomForestRegressor()
model2.fit(X_train, Y_train)

# XGB Regressor
model3 = XGBRegressor()
model3.fit(X_train, Y_train)

# Evaluation

# Prediction on Test Data

# Decision Tree Regressor
test_data_prediction1 = model1.predict(X_test)
mae1 = metrics.mean_absolute_error(Y_test, test_data_prediction1)
print("Mean Absolute Error for Decision Tree Regressor = ", mae1)


# Random Forest Regressor
test_data_prediction2 = model2.predict(X_test)
mae2 = metrics.mean_absolute_error(Y_test, test_data_prediction2)
print("Mean Absolute Error for Random Forest = ", mae2)


# XGBoost Regressor
test_data_prediction3 = model3.predict(X_test)
mae3 = metrics.mean_absolute_error(Y_test, test_data_prediction3)
print("Mean Absolute Error for XGBoost Regressor = ", mae3)

