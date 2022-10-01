import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv("three.csv")

f = plt.figure(figsize=(10,10))             #two ways to give figure
#f.set_figwidth(10)
#f.set_figheight(10)
plt.scatter(df["Interest_rate"], df["Stock_index_price"], color='red')
plt.title('Stock Index Price Vs Interest Rate', fontsize=14)
plt.xlabel("Interest Rate", fontsize=14)
plt.ylabel("Stock Index Price", fontsize=14)
plt.grid()
plt.show()

f = plt.figure()
f.set_figwidth(10)
f.set_figheight(10)
plt.scatter(df['Unemployment_rate'], df['Stock_index_price'], color='blue')
plt.title('Index Price Vs Unemployment rate', fontsize=14)
plt.xlabel('Unemployment rate', fontsize=14)
plt.ylabel('Index Price', fontsize=14)
plt.grid()
plt.show()

x = df[['Interest_rate','Unemployment_rate']]
y = df['Stock_index_price']
regr = linear_model.LinearRegression()
regr.fit(x, y)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

x = np.array([[2.75,5.3]])
y = regr.predict(x)

s =  regr.intercept_ + (regr.coef_[0])*x[0][0] + (regr.coef_[1])*x[0][1]
