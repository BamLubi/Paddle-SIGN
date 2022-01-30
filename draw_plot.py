import matplotlib.pyplot as plt

loss = [196.0975, 145.2021, 135.2727, 130.6693, 127.6749, 125.6001, 124.0152, 122.79936, 121.7900]
ACC = [0.9210, 0.9277, 0.9293, 0.9289, 0.9290, 0.9288, 0.9286, 0.9286, 0.9286]
AUC = [0.8855, 0.8869, 0.8871, 0.8871, 0.8869, 0.8866, 0.8860, 0.8862, 0.8861]

plt.figure(figsize=(12,6), dpi=80)
plt.figure(1)
plt.rcParams['font.sans-serif']='SimHei'

ax1 = plt.subplot(121)
plt.plot(range(len(loss)), loss, color='b', label='LOSS')
# plt.plot(range(len(loss)), loss, color='b', marker='o', label='LOSS')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title("LOSS")
plt.legend(loc='best')

ax2 = plt.subplot(122)
plt.plot(range(len(ACC)), ACC, color='r', label='ACC')
plt.plot(range(len(AUC)), AUC, color='g', label='AUC')
# plt.plot(range(len(ACC)), ACC, color='r', marker='s', label='ACC')
# plt.plot(range(len(AUC)), AUC, color='g', marker='d', label='AUC')
plt.xlabel('Epoch')
plt.ylabel('ACC | AUC')
plt.title("ACC & AUC")
plt.legend(loc='best')

plt.show()