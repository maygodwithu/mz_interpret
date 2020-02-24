import torch
import numpy as np
import scipy.stats as stats
import sys
import operator
import seaborn as sns
from matplotlib import pyplot as plt

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

def graph_margin_d(cam, cross, vocab, query, doc, shape, qfname, qfname2, index):
    tcam = cam.detach().cpu().numpy()
    tcam = np.maximum(tcam, 0)
    tcam = tcam - np.min(tcam)
    tcam = tcam / np.max(tcam)
    tcam_s = np.sum(tcam, axis=0)

    cross = cross.squeeze(0)
    cross = cross.squeeze(0)
    cross = cross.detach().cpu().numpy()
    cross = np.maximum(cross, 0)
    cross -= np.min(cross)
    cross /= np.max(cross)
    cross_s = np.sum(cross, axis=0)

    qs = make_sentence(query, vocab)
    ds = make_sentence(doc, vocab)

    plt.axis('off')
    plt.bar(range(len(tcam_s)), tcam_s, alpha=0.6)
    plt.savefig(qfname, pad_inches=0, bbox_inches='tight')

    plt.bar(range(len(cross_s)), cross_s, color='black', alpha=0.6)
    plt.savefig(qfname2, pad_inches=0, bbox_inches='tight')

def print_margin_d(cam, cross, vocab, query, doc, shape, qfname, index):
    tcam = cam.detach().cpu().numpy()
    tcam = np.maximum(tcam, 0)
    tcam = tcam - np.min(tcam)
    tcam = tcam / np.max(tcam)
    tcam_s = np.sum(tcam, axis=0)

    cross = cross.squeeze(0)
    cross = cross.squeeze(0)
    cross = cross.detach().cpu().numpy()
    cross = np.maximum(cross, 0)
    cross -= np.min(cross)
    cross /= np.max(cross)
    cross_s = np.sum(cross, axis=0)

    qs = make_sentence(query, vocab)
    ds = make_sentence(doc, vocab)

    ti = np.argsort(tcam_s)[::-1]
    ti_c = np.argsort(cross_s)[::-1]

    nots=None
    for ind in ti:
        fq = ds[ind]
        if(fq not in qs):
            if(nots is None): nots = fq
            else:
                nots += ' ' + fq 
        else:
            break
    if(nots is not None):
        print(index, nots, qs)

    dsum={}
    disum={}
    fp = open(qfname, 'w')
#    for ind in reversed(ti):
    for k in range(len(ti)):
        ind = ti[k]
        ind_i = ti_c[k]
        fq = ds[ind]
        fv = tcam_s[ind]
        fq_i = ds[ind]
        fv_i = cross_s[ind]

        if fq in dsum:
            dsum[fq] += fv
        else:
            dsum[fq] = fv

        if fq in disum:
            disum[fq] += fv_i
        else:
            disum[fq] = fv_i

        print(fq, ind, fv, fq_i, ind_i, fv_i, file=fp)

    print("===== margin", file=fp)
    for fq, fv in sorted(dsum.items(), key=operator.itemgetter(1), reverse=True):
        print(fq, dsum[fq], disum[fq], file=fp) 

def print_wilcox(cam, cross, vocab, query, doc, shape, qfname, index):
    tcam = cam.detach().cpu().numpy()
    tcam = np.maximum(tcam, 0)
    tcam = tcam - np.min(tcam)
    tcam = tcam / np.max(tcam)
    tcam_s = np.sum(tcam, axis=0)

    cross = cross.squeeze(0)
    cross = cross.squeeze(0)
    cross = cross.detach().cpu().numpy()
    cross = np.maximum(cross, 0)
    cross_s = np.sum(cross, axis=0)

    qs = make_sentence(query, vocab)
    ds = make_sentence(doc, vocab)

    ti = np.argsort(tcam_s)[::-1]
    ti_c = np.argsort(cross_s)[::-1]

    rank = np.zeros(len(tcam_s))
    rank_c = np.zeros(len(cross_s))
    rank_d = np.zeros(len(tcam_s))

    i=0
    for k in ti:
        i+=1  
        rank[k] = i

    i=0
    for k in ti_c:
        i+=1  
        rank_c[k] = i

    for i in range(len(rank)):
        #rank_d[i] = (rank_c[i] - rank[i]) * (cross_s[i] - tcam_s[i])
        rank_d[i] = (rank_c[i] - rank[i]) 

    dtsum={}
    dtcnt={}
    ti_d = np.argsort(rank_d)
    for k in ti_d:
        dterm = ds[k]
        if(dterm in dtsum):
            dtsum[dterm] += rank_d[k]
            dtcnt[dterm] += 1.0
        else:
            dtsum[dterm] = rank_d[k]
            dtcnt[dterm] =1.0

    for term in dtsum:
        dtsum[term] /= dtcnt[term]
       
    fp = open(qfname, 'a')
    print("====== diff", file=fp)
    i=0
    for term, diff in sorted(dtsum.items(), key=operator.itemgetter(1)):
        print('-\t', term, diff, file=fp)
        i+=1
        if(i>=20): break

    i=0
    for term, diff in sorted(dtsum.items(), key=operator.itemgetter(1), reverse=True):
        print('+\t', term, diff, file=fp)
        i+=1
        if(i>=20): break

def make_sentence(indx, vocab):
    s = []
    if(indx is None or len(indx) < 1): return None
    for n in indx:
        s.append(vocab[n])
    return s 

if __name__ == '__main__':
    vocab = readVocab()
    num = int(sys.argv[1])

    fname = "../save_" + model + "/result_cam.pt" + str(num)
    res = torch.load(fname)

    ## cam
    cam = res['cam']
    query = res['text_left'].squeeze()
    doc = res['text_right'].squeeze()
    cross = res['embcross']
    mean = torch.mean(cam)
    qfname = model + "_cam_margind_" + str(num) + ".txt"
    print_margin_d(cam, cross, vocab, query, doc, cam.shape, qfname, num)
    ## wilcoxon
    print_wilcox(cam, cross, vocab, query, doc, cam.shape, qfname, num)
    ## graph
    qfname = model + "_margin_camgraph_" + str(num) + ".png"
    qfname2 = model + "_margin_intergraph_" + str(num) + ".png"
    graph_margin_d(cam, cross, vocab, query, doc, cam.shape, qfname, qfname2, num)


