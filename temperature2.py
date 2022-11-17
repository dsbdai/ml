import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv('temperatures.csv')
df.head()

x=df['YEAR']
y=df['ANNUAL']
plt.title('Temperature')
plt.xlabel('year')
plt.ylabel('annual avg temperature')
plt.scatter(x,y)

x=x.values
x=x.reshape(117,1)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x,y)
 regressor.predict([[2024]])
 
#mean absolute error
predicted = regressor.predict(x)
np.mean(abs(y-predicted))
#mean square error
np.mean((y-predicted)**2)
#r square error
regressor.score(x,y) 

#visualize regression model
plt.title('Temperature')
plt.xlabel('year')
plt.ylabel('annual avg temperature')
plt.scatter(x,y, label = 'actual', color='r')
plt.plot(x,predicted, label = 'predicted', color = 'g')
plt.legend()

#visualization using seaborn
sns.regplot(x='year', y='annual', data=df)









