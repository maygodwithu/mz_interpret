import torch
import numpy as np
import scipy.stats as stats

def makeCor(sar, car):
    sca = np.array(sar) 
    cma = np.array(car) 
    cor = np.corrcoef(sca, cma)
    return cor[0,1]

def KLD(pk, qk):
    pk /= pk.sum()
    qk /= qk.sum()

    kld = np.sum(stats.entropy(pk, qk))
    return kld

if __name__ == '__main__':
    topk = 100
    model = "match_pyramid_ms"

    fp = open("../save_" + model + "/result.pt_stat", 'r')
    num = 0
    sar = []
    car = []
    t_car = []
    cor_ar = []
     
    bkey = None  
    i = 0
    for line in fp:
        key, score = (line.rstrip()).split('\t')
        if(bkey is None): bkey = key
        if(bkey != key):
            cor = makeCor(sar, car)
            cor_ar.append(cor)
            car = []
            sar = []
            num += 1
            if(num >= topk): break
        
        score = float(score)
        sar.append(score)

        fname = "../save_" + model + "/result_cam.pt" + str(i)
        res = torch.load(fname)

        ## cam
        cam = res['cam']
        cam = cam.detach().cpu().numpy()
        cam = np.maximum(cam, 0)
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)

        ## emb-similarity
        gdata = res['embcross'].detach().cpu()
        gdata = gdata.squeeze(0)
        gdata = gdata.squeeze(0)

        ## emb-sim VS cam 
        klv = KLD(cam, gdata.numpy())
        print("kld = ", klv)
       
        car.append(np.mean(cam)) 
        t_car.append(np.mean(cam))
        
        bkey = key
        i += 1
        
    #cor = makeCor(sar, car)
    cor_ar.append(cor)
    cor_np = np.array(cor_ar)
    print(cor_np)
    print("corr mean = ", np.mean(cor_np))
    print("corr var = ", np.std(cor_np))
    print("cam mean = ", np.mean(t_car)) 
    print("cam var = ", np.std(t_car)) 



