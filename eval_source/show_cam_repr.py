import torch
import cv2
import numpy as np
import sys


def show_cam_on_image(img, mask, imgname):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap #+ np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite(imgname, np.uint8(255*cam))

def readVocab():
    vocab = []
    fp = open('../vocab_msfull.index','r')
    for line in fp:
        psd = (line.rstrip()).split(' ')
        term = psd[1]
        vocab.append(term)
    fp.close()
    print("vocab read")
    print('term of 0 index = ', vocab[0])
    return vocab
    ##

def make_sentence(indx):
    s = []
    for n in indx:
        s.append(vocab[n])
    return s 

def print_gtopk_qd(tv, ti, vocab, topk, shape):
    for i in range(topk):
        mp = ti[-1*i-1]
        mv = tv[-1*i-1]
        max_q = mp // shape[1]
        max_d = mp % shape[1]
        print(i, mv, max_q.item(), max_d.item(), vocab[text_left[0,max_q]], vocab[text_right[0,max_d]])

def print_topk_qd(tv, ti, vocab, topk, shape, mean, qfname):
    qdv = {}
    for i in range(topk):
        mp = ti[-1*i-1].item()
        mv = tv[-1*i-1].item()
        max_q = mp // shape[1]
        max_d = mp % shape[1]
        if(mv < mean): break

        #print(i, mv, max_q, max_d, vocab[text_left[0,max_q]], vocab[text_right[0,max_d]])

        q = vocab[text_left[0,max_q]]
        d = vocab[text_right[0,max_d]]

        if(q not in qdv):
            qdv[q] = []

        qdv[q].append((d, max_d, mv))

    fp = open(qfname, 'w')
    for q in qdv:
        for (d, max_d, mv) in qdv[q]:
            print(q, d, max_d, mv, file=fp)
    fp.close()
          

if __name__ == '__main__':
    vocab = readVocab()
#    num = 601  # +51  -59
    num = int(sys.argv[1])
    num -= 1
    #model = "match_pyramid_ms"
    model = "arci_ms5"
    fname = "../save_" + model + "/result_cam.pt" + str(num)
    res = torch.load(fname)

    #cam = res['qcam']
    #mb = res['qemb']
    cam = res['dcam']
    emb = res['demb']
    print(cam)
    cam = cv2.resize(cam.numpy(), (emb.shape[-1], 1))
    print(cam)
    text_left = res['text_left']
    text_right = res['text_right']

    logname = model + "_logstat_" + str(num) + ".txt"
    logf = open(logname, 'w') 

    print(fname, file=logf)
    print(res['text_left'], file=logf)
    print(res['score'], file=logf)
    print(cam.shape, file=logf)
    print(emb.shape, file=logf)

    ## cam mean
    tv, ti = torch.sort(cam.flatten())
    mean = torch.mean(cam)
    var = torch.var(cam)
    print('cam mean=', mean, file=logf)
    print('cam var=', var, file=logf)

    print('query = ', make_sentence(res['text_left'].squeeze()), file=logf)
    print('doc = ', make_sentence(res['text_right'].squeeze()), file=logf)
    qfname = model + "_camqd_" + str(num) + ".txt"
    #print_topk_qd(tv, ti, vocab, 1000, cam.shape, mean, qfname)

    ## cross
    tv, ti = torch.sort(emb.flatten())
    mean = torch.mean(emb)
    var = torch.var(emb)
    qfname = model + "_interqd_" + str(num) + ".txt"
    #print_topk_qd(tv, ti, vocab, 1000, cam.shape, mean, qfname)
    print('cross mean=', mean, file=logf)
    print('cross var=', var, file=logf)
    print('cross max=', tv[-1], ti[-1], file=logf)
    print('cross max2=', tv[-2], ti[-2], file=logf)
    print('cross min=', tv[0], ti[0], file=logf)
    print('cross min2=', tv[1], ti[2], file=logf)

    cam = cam.detach().cpu().numpy()
    #cross = cross.detach().cpu().numpy()
    cam = np.maximum(cam,0)
    cam = cv2.resize(cam, (500, 200))

    cam = cam - np.min(cam)
    cam = cam / np.max(cam)

    imgname = model + "_cam_" + str(num) + ".jpg"
    show_cam_on_image(None, cam, imgname)

