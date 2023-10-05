import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# loading the data from csv file to a Pandas DataFrame
calories = pd.read_csv('calories.csv')

# print the first 5 rows of the dataframe
calories.head()

exercise_data = pd.read_csv('exercise.csv')

calories_data = pd.concat([exercise_data, calories['Calories']], axis=1)

# checking for missing values
calories_data.isnull().sum()

# get some statistical measures about the data
calories_data.describe()

# Displaying the Data
print("### Displaying the Data ###")
print("First 5 rows of the Calories DataFrame:\n", calories.head())
print("\nFirst 5 rows of the Exercise Data DataFrame:\n", exercise_data.head())
print("\nMerged DataFrame (Exercise Data + Calories):\n", calories_data.head())

# Checking for Missing Values and Statistical Measures
print("\n### Data Analysis ###")
print("Missing values:\n", calories_data.isnull().sum())
print("\nStatistical measures:\n", calories_data.describe())

print("\nDistribution plot for Age, Height, and Weight:\n")
plt.figure(figsize=(10, 6))
sns.displot(calories_data['Age'], label='Age', color='purple')
plt.show()
sns.displot(calories_data['Height'], label='Height', color='blue')
plt.show()
sns.displot(calories_data['Weight'], label='Weight', color='green')
plt.legend()
plt.show()

sns.set()

# Replace 'Gender' string values with numerical values
calories_data.replace({"Gender": {'male': 0, 'female': 1}}, inplace=True)

# Plotting the gender column in count plot
sns.countplot(x='Gender', data=calories_data)

# Add proper labels for 0 and 1 on the x-axis
plt.xticks(ticks=[0, 1], labels=['Male', 'Female'])

# Show the plot
plt.show()

correlation = calories_data.corr()

# constructing a heatmap to understand the correlation

plt.figure(figsize=(10, 10))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f',
            annot=True, annot_kws={'size': 8}, cmap='Blues')
plt.show()

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

# Support Vector Regressor
model4 = SVR()
model4.fit(X_train, Y_train)

# Linear Regression
model5 = LinearRegression()
model5.fit(X_train, Y_train)

# Evaluating the Models with Percent Accuracy
mean_actual_value = Y_test.mean()

# Prediction on Test Data

# Decision Tree Regressor
test_data_prediction1 = model1.predict(X_test)
mae1 = metrics.mean_absolute_error(Y_test, test_data_prediction1)
accuracy1 = 100 - (mae1 / mean_actual_value) * 100

# Random Forest Regressor
test_data_prediction2 = model2.predict(X_test)
mae2 = metrics.mean_absolute_error(Y_test, test_data_prediction2)
accuracy2 = 100 - (mae2 / mean_actual_value) * 100

# XGBoost Regressor
test_data_prediction3 = model3.predict(X_test)
mae3 = metrics.mean_absolute_error(Y_test, test_data_prediction3)
accuracy3 = 100 - (mae3 / mean_actual_value) * 100

# Support Vector Regressor
test_data_prediction4 = model4.predict(X_test)
mae4 = metrics.mean_absolute_error(Y_test, test_data_prediction4)
accuracy4 = 100 - (mae4 / mean_actual_value) * 100

# Linear Regression
test_data_prediction5 = model5.predict(X_test)
mae5 = metrics.mean_absolute_error(Y_test, test_data_prediction5)
accuracy5 = 100 - (mae5 / mean_actual_value) * 100

# Evaluating the Models with Percent Accuracy
print("\n### Model Evaluation with Percent Accuracy ###\n")
print("\nMean Absolute Error for Support Vector Regressor: {:.2f} calories".format(mae4))
print("Accuracy for Support Vector Regressor: {:.2f}%".format(accuracy4))

print("\nMean Absolute Error for Linear Regression: {:.2f} calories".format(mae5))
print("Accuracy for Linear Regression: {:.2f}%".format(accuracy5))

print("Mean Absolute Error for Decision Tree Regressor: {:.2f} calories".format(mae1))
print("Accuracy for Decision Tree Regressor: {:.2f}%".format(accuracy1))

print("\nMean Absolute Error for Random Forest: {:.2f} calories".format(mae2))
print("Accuracy for Random Forest: {:.2f}%".format(accuracy2))

print("\nMean Absolute Error for XGBoost Regressor: {:.2f} calories".format(mae3))
print("Accuracy for XGBoost Regressor: {:.2f}%".format(accuracy3))


# Visualizing MAE and Percent Accuracy for the Models
models = ['SVR', 'Linear Regression', 'Decision Tree', 'Random Forest', 'XGBoost']
mae_values = [mae4, mae5, mae1, mae2, mae3]
accuracy_values = [accuracy4, accuracy5, accuracy1, accuracy2, accuracy3]

fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.bar(models, mae_values, color='skyblue', label='Mean Absolute Error (MAE)')
ax1.set_xlabel('Models')
ax1.set_ylabel('Mean Absolute Error (MAE)')
ax1.set_ylim(0, max(mae_values) + 5)
ax1.legend(loc='upper left')

ax2 = ax1.twinx()
ax2.plot(models, accuracy_values, color='orange', marker='o', label='Percent Accuracy')
ax2.set_ylabel('Percent Accuracy (%)')
ax2.set_ylim(0, 150)
ax2.legend(loc='upper right')

plt.title('Mean Absolute Error and Percent Accuracy for Each Model')
plt.show()
