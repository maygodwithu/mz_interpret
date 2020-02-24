import torch
import numpy as np
import scipy.stats as stats
import sklearn.metrics as metrics
import cv2
import seaborn as sns
from matplotlib import pyplot as plt

#model = "match_pyramid_ms5" ## model full
model = "match_pyramid_ms"    ## 10% sample
vocab_file = "../vocab_ms.index"

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

def make_graph(pcam, ncam):
    pdata = np.sort(pcam.flatten())[::-1]
    ndata = np.sort(ncam.flatten())[::-1]
    x = np.arange(0, 2000, 1)

    plt.plot(x, pdata, label='truth')
    plt.plot(x, ndata, label='not truth')

    plt.xlabel('sorted order')
    plt.ylabel('cam value')

    plt.title("Sorted Cam value")
    plt.legend()

    imgname = model + "_posneg.jpg"
    plt.savefig(imgname, bbox_inches='tight', dpi=300)
    print(stats.ttest_ind(pdata, ndata))

def compare_kurt_posneg(model):
    i=0
    doci=0
    bkey = None
    pi=0
    ni=0
    psimple_sum = []
    nsimple_sum = []
    pcam_sum = None
    ncam_sum = None
    cam_sum = None
    kurt_a = []
    kurt_na = []
    ksum = 0
    ssum_a = []
    fp = open("../save_" + model + "/result.pt_stat", 'r')
    for line in fp:
        key, score = (line.rstrip()).split('\t')
        if(bkey is None): bkey = key
        if(bkey != key):
            kurt_na.append(ksum/(doci-1))
            #nsimple_sum.append(ssum/(doci-1))
            nsimple_sum.append(np.mean(np.array(ssum_a)))
            doci=0
            ksum=0
            ssum_a=[]

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

        ## sum
        cam = cv2.resize(cam, (200, 10))
        cam = np.sort(cam.flatten())[::-1]
        if(doci == 0):
            pi += 1
            if(pcam_sum is None): 
                pcam_sum = cam
            else:
                pcam_sum += cam
            kurt_a.append(kurt)
            psimple_sum.append(pos_sum)
        else:
            ni += 1
            if(ncam_sum is None): 
                ncam_sum = cam
            else:
                ncam_sum += cam
            ksum += kurt
            ssum_a.append(pos_sum)
  
            #kurt_na.append(kurt)

        doci += 1
        i += 1
        bkey = key
        #if(i>=100): break
        if(i>=49000): break

    np_kurt_a = np.array(kurt_a)
    np_kurt_na = np.array(kurt_na)

    np_psimple_sum = np.array(psimple_sum)
    np_nsimple_sum = np.array(nsimple_sum)

    print("truth cam=", pcam_sum/pi)
    print("not truth cam=", ncam_sum/ni)
    print("truth kurt mean = ", np.mean(np_kurt_a))
    print("truth kurt std = ", np.std(np_kurt_a))
    print("truth kurt normal test = ", stats.normaltest(np_kurt_a))
    print("truth kurt Q1 = ", np.percentile(np_kurt_a, 25))
    print("truth kurt median = ", np.median(np_kurt_a))
    print("truth kurt Q3 = ", np.percentile(np_kurt_a, 75))
    print("not truth kurt mean = ", np.mean(np_kurt_na))
    print("not truth kurt std = ", np.std(np_kurt_na))
    print("not truth kurt Q1 = ", np.percentile(np_kurt_na, 25))
    print("not truth kurt median = ", np.median(np_kurt_na))
    print("not truth kurt Q3 = ", np.percentile(np_kurt_na, 75))
    print("not truth kurt normal test = ", stats.normaltest(np_kurt_na))
    print("ttest =", stats.wilcoxon(np_kurt_a[:4000], np_kurt_na[:4000]))

#    plt.figure(figsize=(7,6))
#    plt.boxplot([np_kurt_a, np_kurt_na])
#    plt.savefig("boxplot.jpg")
#    plt.figure(figsize=(7,6))
#    plt.boxplot([np_psimple_sum, np_nsimple_sum])
#    plt.savefig("boxplot_camsum.jpg", bbox_inches='tight', dpi=300)
    make_bargraph(np_psimple_sum, np_nsimple_sum)
    make_graph(np.array(pcam_sum/pi), np.array(ncam_sum/ni))
  
def make_bargraph(psum, nsum):
    x_pos = np.arange(2)
    CTEs = [np.mean(psum), np.mean(nsum)]
    error = [np.std(psum), np.std(nsum)]
    fig, ax = plt.subplots()
    ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel('CAM value sum')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Truth', 'Not truth'])
    #ax.set_title('Coefficent of Thermal Expansion (CTE) of Three Metals')
    ax.yaxis.grid(True)
    
    # Save the figure and show
    plt.tight_layout()
    plt.savefig('barplot_camsum.png', dpi=300)

if __name__ == '__main__':
#    vocab = readVocab()

    #compare_ndcg(model)
    compare_kurt_posneg(model)

