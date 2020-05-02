import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits 
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

digits=load_digits() 
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

model=SVC(kernel='poly',gamma='scale')
X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.5, shuffle=False)
model.fit(X_train,y_train)
predicted = model.predict(X_test)
print("the accuracy of the model :",int(model.score(X_test,y_test)*100),"%")

