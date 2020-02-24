import torch
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import sys

model = "match_pyramid_ms"   ## 10%
vocab_file = "../vocab_ms.index" ## 10%

def readVocab():
    vocab = []
    fp = open(vocab_file,'r')
    for line in fp:
        psd = (line.rstrip()).split(' ')
        term = psd[1]
        vocab.append(term)
    fp.close()
    print("vocab read")
    print('term of 0 index = ', vocab[0])
    return vocab
    ##

if __name__ == '__main__':
    vocab = readVocab()
    num = int(sys.argv[1])
    #num = 59  # +51  -59
#    num -= 1
    fname = "../save_" + model + "/result_cam.pt" + str(num)
    res = torch.load(fname)

    #gdata = data['hmulti'].detach().cpu()
    #gdata = data['rmulti'].detach().cpu()
    gdata = res['embcross'].detach().cpu()
    gdata = gdata.squeeze(0)
    gdata = gdata.squeeze(0)
    
    print(gdata.shape)

#    gdata[gdata<0] = 0
    
    np_data = gdata.numpy()
    plt.figure(figsize=(20, 5))
    plt.axis('off')
    #ax = sns.heatmap(np_data, cmap="YlGnBu", xticklabels=False, yticklabels=False, cbar=False)
    ## for mixing
    ax = sns.heatmap(np_data, cmap="Greys", xticklabels=False, yticklabels=False, cbar=False)
    imgname = model + "_heat_" + str(num) + ".jpg"
    plt.savefig(imgname, pad_inches=0, bbox_inches='tight')

    ## naive
    ax = sns.heatmap(np_data, cmap="Greys")
    imgname = model + "_heat_nomix_" + str(num) + ".jpg"
    plt.savefig(imgname, pad_inches=0, bbox_inches='tight')



