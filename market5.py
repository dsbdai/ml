import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Mall_Customers.csv')

x = df[['Annual Income (k$)','Spending Score (1-100)']]

#plt.title('Unclustered data')
#plt.xlabel('x --->  annual income')
#plt.ylabel('y --->  spending score')
#plt.scatter(x['Annual Income (k$)'], x['Spending Score (1-100)'])
#plt.show()

# evaluating clusters i.e find ideal value of k
#elbow method and silhouette method

#elbow method

from sklearn.cluster import KMeans, AgglomerativeClustering

#find the value of k using elbow method

sse = []
for k in range(1,16):
    km = KMeans(n_clusters=k)
    km.fit_predict(x)
    sse.append(km.inertia_)

print(sse)

#now ideal value of k is elbow point in inertia vs k plot

plt.title('Elbow Method')
plt.xlabel('Value of K')
plt.ylabel('SSE')
plt.grid()
plt.xticks(range(1,16))
plt.plot(range(1,16), sse, marker='.', color='red')
plt.show()

#creating clusters
km = KMeans(n_clusters=5)

labels = km.fit_predict(x)
print(labels)


#centroid

cent = km.cluster_centers_



plt.figure(figsize=(16,9))
plt.subplot(1,2,1)
plt.title('Unclustered Data')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.scatter(x['Annual Income (k$)'], x['Spending Score (1-100)'])

plt.subplot(1,2,2)
plt.title('Clustered Data')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.scatter(x['Annual Income (k$)'], x['Spending Score (1-100)'], c=labels)
plt.scatter(cent[:, 0], cent[:,1], s=100, color='k')
plt.show()

print(km.predict([[46, 78]]))