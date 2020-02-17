import torch
import numpy as np
import sys

model = "match_pyramid_ms"    ## 10% sample
vocab_file = "../vocab_ms.index"
#model = "match_pyramid_ms5"    ## 10% sample
#vocab_file = "../vocab_msfull.index"

def readVocab():
    vocab = []
    fp = open(vocab_file,'r')    ## 10%sample
    #fp = open('../vocab_msfull.index','r') ## full
    for line in fp:
        psd = (line.rstrip()).split(' ')
        term = psd[1]
        vocab.append(term)
    fp.close()
    print("vocab read")
    print('term of 0 index = ', vocab[0])
    return vocab
    ##


def print_margin_d(cam, vocab, query, doc, shape, qfname, index):
    tcam = cam.detach().cpu().numpy()
    tcam = np.maximum(tcam, 0)
    tcam = tcam - np.min(tcam)
    tcam = tcam / np.max(tcam)
    tcam_s = np.sum(tcam, axis=0)

    qs = make_sentence(query, vocab)
    ds = make_sentence(doc, vocab)

    ti = np.argsort(tcam_s)

    nots=None
    for ind in reversed(ti):
        fq = ds[ind]
        if(fq not in qs):
            if(nots is None): nots = fq
            else:
                nots += ' ' + fq 
        else:
            break
    if(nots is not None):
        print(index, nots, qs)

    if(i == margin_num):
        fp = open(qfname, 'w')
        for ind in reversed(ti):
            fq = ds[ind]
            fv = tcam_s[ind]
            print(fq, ind, fv, file=fp)
        

def make_sentence(indx, vocab):
    s = []
    for n in indx:
        s.append(vocab[n])
    return s 

if __name__ == '__main__':
    num = sys.argv[1]
    span=20
    vocab = readVocab()

    fname = "../save_" + model + "/result_cam.pt" + num
    res = torch.load(fname)

    cam = res['cam']
    query = res['text_left'].squeeze()
    doc = res['text_right'].squeeze()

    cam = cam.detach().cpu().numpy()
    cam = np.maximum(cam, 0)
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)

    tcam = np.sum(cam, axis=0)

    qs = make_sentence(query, vocab)
    ds = make_sentence(doc, vocab)

    maxi = np.maximum(len(tcam)-span, 1) 
    maxpos = 0
    maxsum = 0
    for i in range(maxi):
        tsum = np.sum(tcam[i:i+span])
        if(tsum > maxsum): 
            maxpos = i
            maxsum = tsum
    print('maxpos=', maxpos)                  
    print('maxsum=', maxsum)
    print('query=', qs)
    print('snippet=', ds[maxpos:maxpos+span])
    print('cam val=', tcam[maxpos:maxpos+span])

    out={}
    out['snippet'] = ds[maxpos:maxpos+span]
    out['cam'] = torch.from_numpy(tcam[maxpos:maxpos+span])
    out['query'] = qs
    
    sni_fname = model + "_snippet.pt" + num 
    torch.save(out, sni_fname)
 


