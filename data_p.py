import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler


df = pd.read_excel(r'AirQualityUCI.xlsx')

print(df.head())
print(list(df.columns))
print(df.dtypes)




df = df.drop(columns=list(df.columns)[-2:], axis=1)

Data_f = df[["Date","CO(GT)", "NOx(GT)", "NO2(GT)", "T","C6H6(GT)","PT08.S5(O3)","NMHC(GT)"]]

Data_f.iloc[:,0] = pd.to_numeric(Data_f.iloc[:,0])
Data_f =Data_f[Data_f > 0]
Data_f= Data_f.dropna()
Data_f.iloc[:,0] = pd.to_datetime(Data_f.iloc[:,0])

print(Data_f.head())

print(list(Data_f.shape)[1])

DATA = Data_f.values

print(Data_f.describe())


'''Data_f = Data_f.groupby(Data_f["Date"]).mean()
D_te = np.array(Data_f.index)'''





'''plt.scatter(y,X[:, 0])
plt.xlabel("Date")
plt.ylabel("CO")
plt.show()
plt.plot()

plt.scatter( y,X[:, 1],)
plt.xlabel("Date")
plt.ylabel("Nox")
plt.show()
plt.plot()

best = 0

plt.scatter(y,X[:, 2])
plt.xlabel("X")
plt.ylabel("NO2")
plt.show()
plt.plot()

plt.scatter( y,X[:, 3],)
plt.xlabel("X")
plt.ylabel("T")
plt.show()
plt.plot()

plt.scatter( y,X[:, 4],)
plt.xlabel("c6h6")
plt.ylabel("T")
plt.show()
plt.plot()
'''
'''plt.scatter( D_te,X[:, 5])
plt.xlabel("Ozone Karakurt")
plt.ylabel("T")
plt.show()
plt.plot()
'''
y = np.array(Data_f["PT08.S5(O3)"])
X = np.array(Data_f[["CO(GT)", "NOx(GT)","NMHC(GT)"]])

print(Data_f.head())

best = 0
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

for _ in range(1):
    # train data
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)

    print("svr_pr")

    print(acc)
    if acc > best:
        best = acc
        model = linear
        best_coef = linear.coef_
        best_intercept = linear.intercept_
        # save the model via pickle


    # get coefficent of the model and the intercept
    print('coefficient:\n', linear.coef_)
    print('intercept: \n', linear.intercept_)


with open("ozone.pickle", "wb") as f:
    pickle.dump(model, f)

'''pickle_in = open("ozone.pickle","rb")
linear = pickle.load(pickle_in)
best = linear.score(x_test, y_test)
model = linear
best_coef = linear.coef_
best_intercept = linear.intercept_'''

pred = model.predict(x_test)

plt.scatter(y_test,pred, color = "green")
plt.xlabel("Ozone Karakurt")
plt.ylabel("T")
plt.show()
plt.plot()

print(best)
print(y_test.size)
print(pred.size)
print(y_train.size)



from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

print('Mean squared error: %.2f'
      % mean_squared_error(y_test, pred))
print('Coefficient of determination: %.2f'
      % r2_score(y_test, pred))


#print("O3 = {} + {}CO + {}Nox + {}NO2 + {}T+  {}C6H6 ".format(best_intercept, best_coef[0],  best_coef[1], best_coef[2], best_coef[3], best_coef[4]))
#C6H6 sildik, yerine nmhC koycaz
#Nox, kaldı No2 sildik.