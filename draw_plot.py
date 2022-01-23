from cProfile import label
from turtle import color
import matplotlib.pyplot as plt

x = [0,1,2,3,4,5,6,7,8,9]
loss = [0.2399, 0.2131, 0.1645, 0.1417, 0.1305, 0.1239, 0.1192, 0.1155, 0.1126, 0.1101]
ACC = [0.6670, 0.7859, 0.8408, 0.8518, 0.8573, 0.8598, 0.8613, 0.8620, 0.8637, 0.8638]
AUC = [0.6081, 0.8414, 0.8681, 0.8776, 0.8834, 0.8869, 0.8892, 0.8908, 0.8930, 0.8940]

plt.figure(figsize=(12,6), dpi=80)
plt.figure(1)
plt.rcParams['font.sans-serif']='SimHei'

ax1 = plt.subplot(121)
plt.plot(x, loss, color='b', marker='o', label='LOSS')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title("LOSS")
plt.legend(loc='best')

ax2 = plt.subplot(122)
plt.plot(x, ACC, color='r', marker='s', label='ACC')
plt.plot(x, AUC, color='g', marker='d', label='AUC')
plt.xlabel('Epoch')
plt.ylabel('ACC | AUC')
plt.title("ACC & AUC")
plt.legend(loc='best')

plt.show()