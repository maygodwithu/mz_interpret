import torch
import numpy as np
import scipy.stats as stats
import sklearn.metrics as metrics

#model = "match_pyramid_ms5" ## model full
model = "match_pyramid_ms"    ## 10% sample
vocab_file = "../vocab_ms.index"
#ndcg_n = 3
ndcg_n = 5

def readVocab():
    vocab = []
    fp = open(vocab_file, 'r')
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

#    fp = open(qfname, 'w')
    for ind in reversed(ti):
        fq = ds[ind]
        if(fq not in qs):
            print(index, fq, qs)
        break
#        print(ds[ind], tcam_s[ind], file=fp)
#    fp.close()

def make_sentence(indx, vocab):
    s = []
    for n in indx:
        s.append(vocab[n])
    return s 

def ndcg(score, n):
    # true
    true = []
    for i in range(len(score)):
        if(i==0):
            true.append(1)
        else:
            true.append(0)
    #print(np.array(true))
    #print(np.array(score))

    return metrics.ndcg_score(np.array([true]), np.array([score]), k=n)
    

def compare_ndcg(model):
    i=0
    doci=0
    pnr_a = []
    kurt_a = []
    score_a = []
    org_n = []
    kurt_n = []
    pnr_n = []
    kurtpnr_n = []
    bkey = None
    fp = open("../save_" + model + "/result.pt_stat", 'r')
    for line in fp:
        key, score = (line.rstrip()).split('\t')
        if(bkey is None): bkey = key
        if(bkey != key):
            npscore = np.array(score_a)
            npkurt = np.array(kurt_a)
            nppnr = np.array(pnr_a)
            npkurt -= np.min(npkurt)
            npkurt /= np.max(npkurt)
            #print(npscore)
            #print(npkurt)
            #print(npscore*npkurt)
            org_ndcg = ndcg(npscore, ndcg_n)
            kurt_ndcg = ndcg(npscore+2*npkurt, ndcg_n)
            pnr_ndcg = ndcg(npscore+nppnr, ndcg_n)
            kurtpnr_ndcg = ndcg(npscore+2*npkurt+nppnr, ndcg_n)
            print(org_ndcg, kurt_ndcg, pnr_ndcg, kurtpnr_ndcg)
            org_n.append(org_ndcg)
            kurt_n.append(kurt_ndcg)
            pnr_n.append(pnr_ndcg)
            kurtpnr_n.append(kurtpnr_ndcg)
            
            pnr_a = []
            kurt_a = []
            score_a = []
            doci=0

        score = float(score)
        fname = "../save_" + model + "/result_cam.pt" + str(i)
        res = torch.load(fname)

        cam = res['cam']
        cam = cam.detach().cpu().numpy()
        kurt = stats.kurtosis(cam.flatten())
        #print(cam)
        #print(kurt)
        pos_sum = np.sum(cam[cam>0])
        neg_sum = -1 * np.sum(cam[cam<0])
        posneg_ratio = pos_sum / (pos_sum + neg_sum)

        ##
        score_a.append(score)
        kurt_a.append(kurt)
        pnr_a.append(posneg_ratio)

        doci += 1
        i += 1
        bkey = key
        if(i>=49000): break

    print("org ndcg = ", np.array(org_n).mean())
    print("org + kurt ndcg = ", np.array(kurt_n).mean())
    print("org + pnr ndcg = ", np.array(pnr_n).mean())
    print("org + kurt + pnr ndcg = ", np.array(kurtpnr_n).mean())
  
    print(stats.ttest_ind(np.array(org_n), np.array(kurt_n)))

if __name__ == '__main__':
#    vocab = readVocab()

    compare_ndcg(model)

