import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as Listed
import numpy as np
from matplotlib.colors import ListedColormap
from machineNeure.Perceptron import Perceptron

file = "iris.csv";

df = pd.read_csv(file,header=None);
df.head(10);

y = df.loc[0:100,4].values;

y = np.where(y=='Iris-setosa',-1,1);

x = df.loc[0:100,[0,2]].values;

plt.scatter(x[:50,0],x[:50,1],color='red',marker='o',label='setosa');
plt.scatter(x[50:100,0],x[50:100,1],color='blue',marker='x',label='versicolor');
plt.xlabel('xxxx');
plt.ylabel('yyyy');
plt.legend(loc = 'upper left');
#plt.show();

ppn = Perceptron(eta=0.1,n_iter=10);
ppn.fit(x, y);
plt.plot(range(1,len(ppn.errors)+1),ppn.errors,marker='o');
# plt.xlabel('Epochs');
# plt.ylabel('错误分类次数');
# plt.show();

def plot_decision_regins(x,y,classifie,resolution=0.02):
    marker = ('s','x','o','v');
    colors = ('red','blue','lightgreen','gray','cyan');
    cmap = ListedColormap(colors[:len(np.unique(y))]);
     
    x1_min,x1_max = x[:,0].min() - 1 ,x[:,0].max();
    x2_min,x2_max = x[:,1].min() - 1 ,x[:,1].max();
#     print(x1_min,x1_max);
#     print(x2_min,x2_max);
    xx1,xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution),np.arange(x2_min,x2_max,resolution));
     
    z = classifie.predict(np.array([xx1.ravel(),xx2.ravel()]).T);
#     print(z)
    z = z.reshape(xx1.shape);
    plt.contourf(xx1,xx2,z,alpha=0.4,resolution=0.02);
    plt.xlim(xx1.min(),xx1.max());
    plt.ylim(xx2.min(),xx2.max());
     
    for idx,c1 in enumerate(np.unique(y)):
        plt.scatter(x=x[y==c1,0], y=x[y==c1,1], alpha=0.8, c=cmap(idx), marker=marker[idx], label=c1);
 
plot_decision_regins(x, y, ppn, resolution=0.02); 
# plt.xlabel('xxxx');
# plt.ylabel('yyyy');
# plt.legend(loc = 'upper left');
plt.show(); 
