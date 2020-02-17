import torch
import numpy as np
import scipy.stats as stats

model = "match_pyramid_ms"    ## 10% sample
vocab_file = "../vocab_ms.index"
#model = "match_pyramid_ms5"    ## 10% sample
#vocab_file = "../vocab_msfull.index"
margin_num = 9537
#margin_num = 50

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
    vocab = readVocab()
    topk = 1000
    fp = open("../save_" + model + "/result.pt_stat", 'r')

    car = []
    cnar = []
    cam_sum=0
    cam_nsum = 0
    dcnt = 0
    dncnt = 0
    bkey = None  
    doci=0
    i=0
    num = 0
    for line in fp:
        key, score = (line.rstrip()).split('\t')
        if(bkey is None): bkey = key
        if(bkey != key):
            num += 1
            doci=0
            if(num >= topk): break

        score = float(score)

        fname = "../save_" + model + "/result_cam.pt" + str(i)
        res = torch.load(fname)

        ## cam
        cam = res['cam']
        query = res['text_left'].squeeze()
        doc = res['text_right'].squeeze()
        mean = torch.mean(cam)
        if(doci == 0):
            qfname = model + "_cam_margind_" + str(i) + ".txt"
            print_margin_d(cam, vocab, query, doc, cam.shape, qfname, i)

        cam = cam.detach().cpu().numpy()
#       cam = np.maximum(cam, 0)

#       cam = cam - np.min(cam)
#       cam = cam / np.max(cam)

        ##normalize
        kurt = stats.kurtosis(cam.flatten())

        cam_sort = np.sort(cam.flatten())[::-1]
        pos_sum = np.sum(cam[cam>0])
        neg_sum = -1 * np.sum(cam[cam<0])
        posneg_ratio = pos_sum / (pos_sum + neg_sum)
     
        mkurt = kurt
        #mkurt = posneg_ratio
        if(doci == 0):
            cam_sum += mkurt
            dcnt += 1
        else:
            cam_nsum += mkurt
            dncnt += 1

        cam = cam - np.min(cam)
        cam = cam / np.max(cam)

#        print(doci, score, kurt, mkurt, pos_sum, neg_sum, posneg_ratio, tot_sum, top_sum/tot_sum)
        doci += 1
        i += 1

        bkey = key


    print("truth = ", cam_sum / float(dcnt))
    print("not truth = ", cam_nsum / float(dncnt))

