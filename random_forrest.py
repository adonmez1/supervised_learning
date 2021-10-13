import untitled0 as unt
import numpy as np
import scipy.special as sps
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

(data_train)=unt.train_load_data()
(data_test)=unt.test_load_data()

y_parole_train=np.logical_or(data_train.values[:,3]==46,data_train.values[:,3]==56)
y_parole_test=np.logical_or(data_test.values[:,3]==46,data_test.values[:,3]==56)

#10th COLUMN IS THE TARGET VARIABLE. TAILOR THIS PER YOUR NEED
y_train=data_train.values[:,10]
# TAILOR BELOW TO KEEP FEATURES PER YOUR NEED
x_train=data_train.drop([0,3,4,9,10], axis=1)
#10th COLUMN IS THE TARGET VARIABLE. TAILOR THIS PER YOUR NEED
y_test=data_test.values[:,10]
# TAILOR BELOW TO KEEP FEATURES PER YOUR NEED
x_test=data_test.drop([0,3,4,9,10], axis=1)


sc = StandardScaler()  
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

class_weight1 = {0: 1.,
                1: 2.}

class_weight2 = {0: 1.,
                1: 10.}

regressor = RandomForestClassifier(n_estimators=100, random_state=0)
regressor2 = RandomForestClassifier(n_estimators=100, random_state=0)

regressor.fit(x_train, y_train)  
regressor2.fit(x_train, y_parole_train)
y_pred = regressor.predict(x_test)
y_pred2 = regressor2.predict(x_test)

print(confusion_matrix(y_test,y_pred.round()))  
print(classification_report(y_test,y_pred.round()))  
print(accuracy_score(y_test, y_pred.round())) 

print(confusion_matrix(y_parole_test,y_pred2.round()))  
print(classification_report(y_parole_test,y_pred2.round()))  
print(accuracy_score(y_parole_test, y_pred2.round()))
