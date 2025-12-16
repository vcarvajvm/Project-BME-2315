# %%
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
housing = fetch_california_housing(as_frame=True)
print(housing.data.shape, housing.target.shape)
print(housing.feature_names[0:6])

# %%
print(housing.DESCR)
# %% single feature
feature = "MedInc"

X = housing["data"][feature].values.reshape(-1, 1)
y = housing.target

# lr_model = LinearRegression()
# reg = lr_model.fit(X, y)
reg = LinearRegression().fit(X, y)
print("R^2", reg.score(X, y))
print(reg.coef_, reg.intercept_)


# %%
x_test = np.linspace(0, 15, 100).reshape(-1, 1)
y_test = reg.predict(x_test)
plt.scatter(X, y)
plt.plot(x_test, y_test, color="red")
plt.xlabel(feature)
plt.ylabel("House Value ($100k)")
plt.annotate(
    "R^2 = {:.2f}".format(reg.score(X, y)),
    xy=(0.5, 0.9),
    xycoords="axes fraction",
    fontsize=14,
    ha="center",
)
plt.show()

# %%
#############################
# What's the best R^2 for a single feature?
##############################

# Your code here


# %%
# ##############################
# what R^2 do you get with all features?
##############################

# Your code here
