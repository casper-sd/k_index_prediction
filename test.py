import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from feature import prepare_feature_matrix
from resource import Observatory
import seaborn as sns

from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.pipeline import Pipeline

ob = Observatory('OTT')
ob.parse()

X, y = prepare_feature_matrix(ob, 'X')
print(X.shape, y.shape)
svr = SVR(kernel='rbf')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

svr.fit(X_train, y_train)
y_pred = svr.predict(X_test)
print(mean_absolute_percentage_error(y_test, y_pred))

plt.plot(y_test[:300])
plt.plot(y_pred[:300])
plt.show()
