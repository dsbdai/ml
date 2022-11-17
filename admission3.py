import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns

df = pd.read_csv('Admission_Predict.csv')

print(df.head())


# binarising output variable

from sklearn.preprocessing import Binarizer

# if value if greater than 0.75 then set it to 1 else set it to 0
bi = Binarizer(threshold=0.75)
df['Chance of Admit '] = bi.fit_transform(df[['Chance of Admit ']])

print(df)
print(df.head())

x = df.drop('Chance of Admit ', axis=1)

#output variable
y = df['Chance of Admit ']

y = y.astype('int')
print(y)

print(y.value_counts())

# train test

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.25)


from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(random_state=0)

#model formed
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

result = pd.DataFrame({
    'actual' : y_test,
    'predicted': y_pred
})

print(result)


from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report

ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.show()

print(accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))


