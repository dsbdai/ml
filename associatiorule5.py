import pandas as pd
import csv
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

dataset = []
with open('Market_Basket_Optimisation.csv') as file:
    reader = csv.reader(file, delimiter=',')
    for row in reader:
        dataset += [row]

print(dataset[0])


print(len(dataset))

#making our data structured
te = TransactionEncoder()

x = te.fit_transform(dataset)

# printing all column names
print(te.columns_)

print(len(te.columns_))

df = pd.DataFrame(x, columns=te.columns_)

print(df.head())

#find frequent itemset
freq_itemset = apriori(df, min_support=0.01, use_colnames=True)

print(freq_itemset)

#Find the rules

rules = association_rules(freq_itemset, metric='confidence', min_threshold=0.25)

rules = rules[['antecedents', 'consequents', 'support', 'confidence']]

print(rules)

#meaning - if i purchase avocado then 34% chance i'll purchase mineral water

#find consequent of cake
res = rules[rules['antecedents'] == {'cake'}]['consequents']
print(res)
