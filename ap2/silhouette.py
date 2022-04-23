
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt

option = 'raw'
# option = 'trans'
data = pd.read_csv('data/' + option + '.csv', header = None).values

X = data[:, 0:-1]
Y = data[:, -1]

score = silhouette_score(X = X, labels = Y)
print(score)

silh_samples = silhouette_samples(X = X, labels = Y)

fig, ax = plt.subplots(1, 1, figsize = (5, 5))
y_ticks = []
y_lower = y_upper = 0
color = ['blue', 'green']
for i, k in enumerate(np.unique(Y)):
    cluster = silh_samples[Y == k]
    cluster.sort()
    y_upper += len(cluster)
    
    ax.barh(range(y_lower, y_upper), cluster,height = 1)
    
    # ax.text(-0.03, (y_lower + y_upper) / 2, str(i + 1))
    y_lower += len(cluster)
    
    avg_score = np.mean(cluster)
    print(avg_score)
    ax.axvline(avg_score, linestyle = '--', 
                linewidth = 1, color = color[i])
    # ax.set_xlim([-1, 1])
ax.axvline(score, linestyle = '--', 
           linewidth = 2, color = 'red')
plt.savefig('fig/silh-' + option + '.png')
