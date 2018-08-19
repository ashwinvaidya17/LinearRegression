from matplotlib import pyplot as plt
import numpy as np
from sklearn import linear_model, datasets
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error # to find the error between predicted and the actual values
# Load Dataset
diabetes = datasets.load_diabetes()

#split data into training and testing
# From the sklearn dicumentation
#----
# Use only one feature
diabetes_X = diabetes.data[:, np.newaxis, 2]

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]
#----


model = linear_model.LinearRegression()

#now fit the model
model.fit(diabetes_X_train, diabetes_y_train)

#make predictions
predictions = model.predict(diabetes_X_test)

print("Error: %0.2f" % mean_squared_error(diabetes_y_test, predictions))


#To display the data
plt.scatter(diabetes_X_test, diabetes_y_test, color='black')
plt.plot(diabetes_X_test, predictions, color = 'blue', linewidth=3)
plt.xticks()
plt.yticks()

plt.show()