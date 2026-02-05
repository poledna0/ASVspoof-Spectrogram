import numpy as np
import matplotlib.pyplot as plt

data = np.load('/home/heniruqe/b/dataset/stft/PA/ASVspoof2019_PA_dev/flac/PA_D_0026463.npy')

plt.figure(figsize=(10,4))
plt.imshow(data, aspect='auto', origin='lower')
plt.show()
