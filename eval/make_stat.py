import torch
import numpy as np
import scipy.stats as stats

topk = 4999
model = "match_pyramid_ms"    ## 10% sample
vocab_file = "../vocab_ms.index"
#model = "match_pyramid_ms5"    ## 10% sample
#vocab_file = "../vocab_msfull.index"

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

def pearson_all(pk, qk):
    pk = pk.flatten()    
    qk = qk.flatten()

#    pk_i = (-pk).argsort()
#    qk_i = (-qk).argsort()

    pcor = stats.pearsonr(pk, qk)
    return pcor[0]

def spearman_all(pk, qk):
    pk = pk.flatten()    
    qk = qk.flatten()

#    pk_i = (-pk).argsort()
#    qk_i = (-qk).argsort()

    pcor = stats.spearmanr(pk, qk)
    return pcor[0]

def pearson(pk, qk):
    pcor_sum = 0
    for i in range(pk.shape[0]):
#        pk_i = (-pk[i]).argsort()        
#        qk_i = (-qk[i]).argsort()
        
#        pcor = stats.pearsonr(pk_i, qk_i)
        pcor = stats.pearsonr(pk[i], qk[i])
        if(pcor[0] is np.nan): continue 
        pcor_sum += pcor[0]
        print(i, pcor)
    return pcor_sum / pk.shape[0]

def spearman(pk, qk):
    pcor_sum = 0
    for i in range(pk.shape[0]):
#        pk_i = (-pk[i]).argsort()        
#        qk_i = (-qk[i]).argsort()
        
#        pcor = stats.pearsonr(pk_i, qk_i)
        pcor = stats.spearmanr(pk[i], qk[i])
        if(pcor[0] is np.nan): continue 
        pcor_sum += pcor[0]
        print(i, pcor)
    return pcor_sum / pk.shape[0]


if __name__ == '__main__':

    fp = open("../save_" + model + "/result.pt_stat", 'r')
    num = 0
    sar = []
    car = []
    t_sar = []
    t_car = []
    t_mar = []
    cor_ar = []
    pcor_ar = []
     
    bkey = None  
    i = 0
    doci=0
    for line in fp:
        key, score = (line.rstrip()).split('\t')
        if(bkey is None): bkey = key
        if(bkey != key):
            cor = makeCor(sar, car)
            cor_ar.append(cor)
 
            car = []
            sar = []
            num += 1
            doci=0
            if(num >= topk): break
        
        score = float(score)
        sar.append(score)
        t_sar.append(score)

        fname = "../save_" + model + "/result_cam.pt" + str(i)
        res = torch.load(fname)

        ## cam
        cam = res['cam']
        cam = cam.detach().cpu().numpy()
        car.append(np.mean(cam))   ## before normalize
#        print(doci, score, cam_sum)

        ## emb-similarity
        gdata = res['embcross'].detach().cpu()
        gdata = gdata.squeeze(0)
        gdata = gdata.squeeze(0)
        t_mar.append(np.mean(gdata.numpy()))

        ## emb-sim VS cam 
        #pcor = pearson_all(cam, gdata.numpy())
        #pcor = pearson(cam, gdata.numpy())
        pcor = spearman_all(cam, gdata.numpy())
        #pcor = spearman(cam, gdata.numpy())
      
        #print("pearson = ", pcor)
        if(pcor is not np.nan):
            pcor_ar.append(pcor)

        ## normalized positive cam
        cam = np.maximum(cam, 0)
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        t_car.append(np.mean(cam))
        
        bkey = key
        i += 1
        doci += 1

        ##rank score
        
        
    cor_np = np.array(cor_ar)
    pcor_np = np.array(pcor_ar)
    #print(cor_np)
    print("score_mean=", np.mean(t_sar))
    print("score_std=", np.std(t_sar))
    print("interaction_mean=", np.mean(t_mar))
    print("interaction_std=", np.mean(t_mar))
    print("cam_mean= ", np.mean(t_car)) 
    print("cam_std= ", np.std(t_car)) 
    print("corr_mean= ", np.mean(cor_np))
    print("corr_std= ", np.std(cor_np))
    print("spearman_corr_mean= ", np.mean(pcor_np))
    print("spearman_corr_std= ", np.std(pcor_np))



