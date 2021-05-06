import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#data1 = np.genfromtxt("../../data/office-ql.csv")
#data2 = np.genfromtxt("../../data/office-causal-ql.csv")
data1 = np.genfromtxt("../../data/craft-ql.csv")
data2 = np.genfromtxt("../../data/craftcausal-ql.csv")
data3= np.genfromtxt("../../data/craft-shaped.csv")
data4= np.genfromtxt("../../data/craft-hrl.csv")
data5= np.genfromtxt("../../data/craft-seq.csv")
data6= np.genfromtxt("../../data/craft-pop.csv")
print(data1.shape)
print(data2.shape)
print(data3.shape)
length = 130
plt.plot(range(length), data1[:length, 1] )

plt.plot(range(length), data2[:length, 1])
#plt.plot(range(length), data3[:length, 1])
#plt.plot(range(length), data4[:length, 1])
plt.plot(range(length), data5[:length, 1])
plt.plot(range(length), data6[:length, 1])

plt.show()
