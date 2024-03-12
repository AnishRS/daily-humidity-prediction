'''common libraries'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''gathering the dataset'''
df = pd.read_csv("D:\\TOP MENTOR\\resume projects\\daily weather using decision tree\\data_weather.csv")
print(df)
print("shape of the dataset is", df.shape)

'''checking null values'''
print("null values are, \n", df.isnull().sum())
'''removing null values using dropna'''
df = df.dropna()

'''splitting  feature and target variables'''
x = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values

'''standard scalar on x and y'''
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x = sc.fit_transform(x)

'''train test split'''
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=22)

'''decision tree regression'''
from sklearn.tree import DecisionTreeRegressor

'''using grid search cv for hyper parameter tuning'''
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [None, 5, 10, 15],  # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split a node
    'min_samples_leaf': [1, 2, 4],    # Minimum number of samples required at each leaf node
    'criterion':['squared_error','absolute_error']

}
model = DecisionTreeRegressor()
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='r2')
grid_search.fit(x_train, y_train)
best_param = grid_search.best_params_
best_model = grid_search.best_estimator_
print("best parameters are: ", best_param)

'''predicting the model'''
grid_y_pred = best_model.predict(x_test)

'''evaluating the model'''
from sklearn.metrics import r2_score
grid_r2 = r2_score(y_test, grid_y_pred)
print("r2 score with grid search is: ", grid_r2)

'''visluaizing the model build'''
from sklearn import  tree
fig=plt.figure(figsize=(20,20))
tree.plot_tree(best_model)
plt.show()

