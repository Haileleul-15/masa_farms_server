import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import csv
from firebase import firebase

firebase = firebase.FirebaseApplication('https://masa-farms-default-rtdb.firebaseio.com/', None)
data =  { 'Name': 'Jane Doe',
          'RollNo': 5,
          'Percentage': 77.02
          }

df = pd.read_csv('ex1data2.txt', header = None)
df.head()

df = pd.concat([pd.Series(1, index = df.index, name = '00'), df], axis = 1)
df.head()

X = df.drop(columns = 2)
print(X)
maxValues = []
for i in range(1, len(X.columns)):
    maxValues.append(np.max(X[i-1]))

y = df.iloc[:, 3]

for i in range(1, len(X.columns)):
    X[i-1] = X[i-1]/np.max(X[i-1])
X.head()

theta = np.array([0]*len(X.columns))

m = len(df)

def hypothesis(theta, X):
    return theta*X

def computeCost(X, y, theta):
    y1 = hypothesis(theta, X)
    y1 = np.sum(y1, axis = 1)

    return sum(np.sqrt((y1 - y)**2))/(2*m)

def gradientDescent(X, y, theta, alpha, i):
    J = []  #cost function in each iterations
    k = 0
    while k < i:        
        y1 = hypothesis(theta, X)
        y1 = np.sum(y1, axis=1)
        for c in range(0, len(X.columns)):
            theta[c] = theta[c] - alpha*(sum((y1-y)*X.iloc[:,c])/m)
        j = computeCost(X, y, theta)
        J.append(j)
        k += 1
    return J, j, theta

def predict(X2, X3, theta, maxValues):

    return theta[0] + theta[1]*(X2/maxValues[0]) + theta[2]*(X3/maxValues[1])


J, j, theta = gradientDescent(X, y, theta, 0.3, 100)

y_hat = hypothesis(theta, X)
y_hat = np.sum(y_hat, axis=1)

X1 = np.linspace(1, 1, 41)
X2 = np.linspace(1000, 5000, 41)
X3 = np.linspace(1, 5, 41)
X2, X3 = np.meshgrid(X2,X3)

Y = predict(X2, X3, theta, maxValues)

# with open('data.csv', mode = 'w') as data_file:
#     data_writer = csv.writer(data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

#     data_writer.writerow(Y)

# rs = pd.read_csv('data.csv', header = None)
# rs.head()
# rs = rs.transpose()
# print(rs)

print(predict(1650, 3, theta, maxValues))

fix, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X2, X3, Y, cmap=cm.coolwarm, linewidth = 0, antialiased = False)
plt.show()

# test = theta[0] + (5000/maxValues[0])*theta[1] + (5/maxValues[1])*theta[2]
# print(test)

# plt.figure()
# plt.scatter(x=X3,y= Y, color='blue')         
# plt.scatter(x=list(range(0, 47)), y=y_hat, color='red')
# plt.show()

# plt.figure()
# plt.scatter(x=list(range(0, 1000)), y=J)
# plt.show()

# check how to add negative values
thetaData = {
    "theta0" : theta[0].item() + 100000,
    "theta1" : theta[1].item() + 100000,
    "theta2" : theta[2].item() + 100000
}
# add new data
result = firebase.post('/theta/',thetaData)
# update the data
firebase.put('/theta/-MVvSZqmB7Pz-L6TRkHx','theta0', theta[0].item() + 100000)