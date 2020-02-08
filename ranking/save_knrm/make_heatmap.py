import torch
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np

data = torch.load("result.pt0")

gdata = data['dmulti'].detach().cpu()
gdata = gdata.squeeze(0)

print(gdata.shape)

gdata[gdata<0] = 0

np_data = gdata.numpy()
plt.figure(figsize=(20, 5))
ax = sns.heatmap(np_data)

plt.savefig("test.png")
