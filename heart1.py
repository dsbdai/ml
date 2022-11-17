import pandas as pd
df=pd.read_csv('Heart.csv')
df.head() 
df.isnull().sum()
df.count()
df.dtypes
df[df==0]
df['Age'].mean()
df[['Age','Sex','ChestPain','RestBP','Chol']]
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, random_state=0, test_size=0.25)
train.shape
test.shape
