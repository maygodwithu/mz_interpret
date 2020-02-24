import sys
import numpy as np

def lencheck(doc):
    ndoc = None
    qcnt=0
    psd = (doc.rstrip()).split(' ')
    for q in psd:
        if(qcnt >= 500): break
        if(ndoc is None):
            ndoc = q
        else:
            ndoc += ' ' + q
        qcnt += 1
    return ndoc, len(psd)
   
def make_sample(k):
    fname = "/home/jkchoi/.matchzoo/datasets/msmarco/msmarco-dev.tsv"
    fp = open(fname, 'r')
    sel = np.random.choice(5000, k, replace=False)
 
    fps = open("truth.sel"+str(k)+".txt", 'w')
    #for v in sel:
        #print(v, file=fps)
    #fps.close()

    bkey = None
    doci=0
    i = 0
    fpo = open("truth.sample"+str(k)+".txt", 'w')
    for line in fp:
        i+=1
        psd = (line.rstrip()).split('\t')
        if(len(psd)<7): 
            bkey = key
            continue
        key = psd[0]
        query = psd[1]
        doc = psd[5]
        label = psd[6]

        if(bkey is None): bkey = key
        if(bkey != key):
            if(doci in sel):
                assert(label == '1')
                ndoc, doc_len = lencheck(doc)
                if(doc_len <= 350 and 'define' not in query):
                    print("%s\t%s\t%s" % (bkey, query, ndoc), file=fpo)
                    print(doci, (i-2), query, file=fps)
            doci += 1
        bkey = key
        

if __name__ == "__main__":
    #make_sample(50)
    make_sample(300)
