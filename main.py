import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.svm import SVR

dataset = pd.read_csv('C:/Users/abc/Desktop/Position_Salaries.csv')
X = dataset.iloc[:,[1]].values
y = dataset.iloc[:,[-1]].values
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)
regressor = SVR(kernel='rbf')
regressor.fit(X,y.ravel())
y_pred = regressor.predict(X)

plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = "red")
plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(y_pred.reshape(-1,1)), color= "blue")
plt.title("Truth or bluff (SVR)")
plt.xlabel("Position level")
plt.ylabel('salary')
plt.show()