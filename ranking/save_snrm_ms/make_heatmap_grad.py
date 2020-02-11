import torch
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np

def make_sentence(indx):
    s = []
    for n in indx:
        s.append(vocab[n])
    return s 

## read vocab
vocab = []
fp = open('../vocab_ms.index','r')
for line in fp:
    psd = (line.rstrip()).split(' ')
    term = psd[1]
    vocab.append(term)
fp.close()
print("vocab read")
print('term of 0 index = ', vocab[0])
##

data = torch.load("result.pt0")
qdata = data['qmulti'].detach().squeeze(0).cpu()
ddata = data['dmulti'].detach().squeeze(0).cpu()
#gdata = gdata.squeeze(0)

print(qdata.shape)
print(ddata.shape)

qdata[qdata<0] = 0
q_str = make_sentence(data['text_left'].detach().cpu().squeeze(0).numpy())
print(q_str)
d_str = make_sentence(data['text_right'].detach().cpu().squeeze(0).numpy())
print(d_str)

ddata[ddata<0] = 0
np_data = qdata.numpy()
plt.figure(figsize=(20, 5))
ax = sns.heatmap(np_data)
plt.savefig("qmulti.png")

np_data = ddata.numpy()
plt.figure(figsize=(20, 5))
ax = sns.heatmap(np_data)
plt.savefig("dmulti.png")
