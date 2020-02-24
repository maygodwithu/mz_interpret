import torch
import numpy as np
import sys

model = "match_pyramid_ms"    ## 10% sample
vocab_file = "../vocab_ms.index"
#model = "match_pyramid_ms5"    ## 10% sample
#vocab_file = "../vocab_msfull.index"

PREFERRED_SNIPPET_LENGTH = 20
MAX_SNIPPET_LENGTH = 30
SNIPPET_MATCH_WINDOW_SIZE = 5

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

def get_highest_density_window(document_words, terms, size=PREFERRED_SNIPPET_LENGTH):
    """ Given a list of positions in a corpus, returns the shortest span
    of words that contain all query terms

    >>> get_highest_density_window('this case is a good test', 'test case', size=5)
    (1, 6)

    """
#    terms = get_normalized_terms(query)

#    document_words = doc.split()
    document_size = len(document_words)

    count = high_count = 0
    high_count = 0
    start_index = read_start = 0
    end_index = read_end = start_index + size - 1

    # If the document is shorter than the desired window, just return the doc
    if document_size < size:
        return (start_index, document_size)

    # calculate the # of term occurances in the initial window
    re = {}
    for i in range(read_start, read_end):
        dq = document_words[i]
        if(dq in terms):
            if(dq in re): 
                count += 0.01
                high_count += 0.01
                continue
            count += 1 
            high_count += 1
            re[dq] = 1
 
    read_start += 1
    read_end += 1

    # Use a "sliding window" technique to count occurences
    # Move the window one work at a time. After each iteration if we've
    # picked up a matched term term increment. Decrement if we just dropped
    # a matched term
    while read_end < document_size:
        re = {}
        count = 0
        for i in range(read_start, read_end):
            dq = document_words[i]
            if(dq in terms):
                if(dq in re): 
                    count += 0.01
                    continue
                count += 1 
                re[dq] = 1
 
        if count > high_count:
            high_count = count
            start_index = read_start
            end_index = read_end

        read_start += 1
        read_end += 1

    # Return end_index + 1 so that callers can more intuitively use
    # non-inclusive range operaters
    return (start_index, end_index + 1, high_count)

def get_highest_density_window_with_cam(document_words, terms, tcam, size=PREFERRED_SNIPPET_LENGTH):
    """ Given a list of positions in a corpus, returns the shortest span
    of words that contain all query terms

    >>> get_highest_density_window('this case is a good test', 'test case', size=5)
    (1, 6)

    """
#    terms = get_normalized_terms(query)

#    document_words = doc.split()
    document_size = len(document_words)
## tcam modify
    tcam /= size

    count = 0
    high_count = 0
    camv = 0
    high_camv = 0
    start_index = read_start = 0
    end_index = read_end = start_index + size - 1

    # If the document is shorter than the desired window, just return the doc
    if document_size < size:
        return (start_index, document_size)

    # calculate the # of term occurances in the initial window
    re = {}
    for i in range(read_start, read_end):
        dq = document_words[i]
        if(dq in terms):
            if(dq in re): 
                count += 0.01
                high_count += 0.01
                camv += 0.01
                high_camv += 0.01
                continue
            count += 1 
            high_count += 1
            camv += 1
            high_camv += 1
            re[dq] = 1

        ## add cam
        camv += tcam[i] 
        high_camv += tcam[i]

    read_start += 1
    read_end += 1

    # Use a "sliding window" technique to count occurences
    # Move the window one work at a time. After each iteration if we've
    # picked up a matched term term increment. Decrement if we just dropped
    # a matched term
    while read_end < document_size:
        re = {}
        count = 0
        camv = 0
        for i in range(read_start, read_end):
            dq = document_words[i]
            if(dq in terms):
                if(dq in re): 
                    count += 0.01
                    camv += 0.01
                    continue
                count += 1 
                camv += 1
                re[dq] = 1
 
        camv -= tcam[read_start - 1]
        camv += tcam[read_end]
            
        #if count > high_count:
        if(camv > high_camv):
            high_count = count
            high_camv = camv
            start_index = read_start
            end_index = read_end

        read_start += 1
        read_end += 1

    # Return end_index + 1 so that callers can more intuitively use
    # non-inclusive range operaters
    return (start_index, end_index + 1, high_count, high_camv) 
 

def make_sentence(indx, vocab):
    s = []
    for n in indx:
        s.append(vocab[n])
    return s 

def make_string(ar):
    tstr = None
    for q in ar:
        if(tstr is None):
            tstr = q
        else:
            tstr += ' ' + q
    return tstr



if __name__ == '__main__':
    num = sys.argv[1]
    span=PREFERRED_SNIPPET_LENGTH
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

    tcam = np.sum(cam, axis=0) #/ len(query)

    qs = make_sentence(query, vocab)
    ds = make_sentence(doc, vocab)

    ## info
    print('query=', qs)
    ## naive snippet
    rs, re, rcnt = get_highest_density_window(ds, qs, size=span)
    print("naive res pos =", rs, re, rcnt)
    print("naive snippet =", ds[rs:re])

    ## with cam value
    crs, cre, ccnt, cmax = get_highest_density_window_with_cam(ds, qs, tcam, size=span)
    print("cam res pos =", crs, cre, ccnt, cmax)
    print("cam snippet =", ds[crs:cre])

    ## only cam
    maxi = np.maximum(len(tcam)-span, 1) 
    maxpos = 0
    maxsum = 0
    for i in range(maxi):
        tsum = np.sum(tcam[i:i+span])
        if(tsum > maxsum): 
            maxpos = i
            maxsum = tsum
    print('only cam pos =', maxpos, maxsum)                  
    print('only cam sni =', ds[maxpos:maxpos+span])
#    print('cam val=', tcam[maxpos:maxpos+span])

    out={}
    out['naive_snippet'] = ds[rs:re]
    out['cam_snippet'] = ds[crs:cre]
    out['only_cam_snippet'] = ds[maxpos:maxpos+span]
    out['cam'] = torch.from_numpy(tcam[maxpos:maxpos+span])
    out['query'] = qs
    
    sni_fname = model + "_snippet.pt" + num 
    torch.save(out, sni_fname)
 


