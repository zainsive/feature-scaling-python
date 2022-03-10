# Feature Scaling
## Using Standard Scalar



Feature Scaling is a technique to standardize the independent features present in the data in a fixed range. It is performed during the data pre-processing to handle highly varying magnitudes or values or units
### Libraries:
- StandardScalar
- SVR (Standard Vector Regression)
- pandas
- matplotlib

## Imports

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.svm import SVR
```

## Read File

```python
dataset = pd.read_csv('filepath...')
```
load dataset (reading values from rows and columns)
```python
X = dataset.iloc[:,[1]].values
y = dataset.iloc[:,[-1]].values
```

initializing
```python
sc_X = StandardScaler()
sc_y = StandardScaler()
```
```python
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)
```
### using rbf kernal

The RBF kernel is a stationary kernel. It is also known as the “squared exponential” kernel. It is parameterized by a length scale parameter , which can either be a scalar (isotropic variant of the kernel) or a vector with the same number of dimensions as the inputs X (anisotropic variant of the kernel)

```python
regressor = SVR(kernel='rbf')
regressor.fit(X,y.ravel())
y_pred = regressor.predict(X)
```
plotting graphics based on results
```python
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = "red")
plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(y_pred.reshape(-1,1)), color= "blue")
plt.title("Truth or bluff (SVR)")
plt.xlabel("Position level")
plt.ylabel('salary')
plt.show()
```

![SVR](/results/img.png))
