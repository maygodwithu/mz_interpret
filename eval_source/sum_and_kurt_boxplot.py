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
    ksum_a = []
    ksum = 0
    ssum_a = []
    fp = open("../save_" + model + "/result.pt_stat", 'r')
    for line in fp:
        key, score = (line.rstrip()).split('\t')
        if(bkey is None): bkey = key
        if(bkey != key):
            #kurt_na.append(ksum/(doci-1))
            #kurt_na.append([np.min(np.array(ksum_a))])
            kurt_na.append(np.random.choice(np.array(ksum_a),1))
            #nsimple_sum.append(ssum_a/(doci-1))
            #nsimple_sum.append(np.min(np.array(ssum_a)))
            nsimple_sum.append(np.random.choice(np.array(ssum_a), 1))
            doci=0
            ksum=0
            ssum_a=[]
            ksum_a=[]

        score = float(score)
        fname = "../save_" + model + "/result_cam.pt" + str(i)
        res = torch.load(fname)

        cam = res['cam']
        cam = cam.detach().cpu().numpy()
        kurt = stats.kurtosis(cam.flatten())
    #    pos_sum = np.sum(cam[cam>0])
    #    neg_sum = -1 * np.sum(cam[cam<0])
    #    posneg_ratio = pos_sum / (pos_sum + neg_sum)
        cam = np.maximum(cam, 0)
        #cam -= np.min(cam)
        #cam /= np.max(cam)
        pos_sum = np.sum(cam)

        ## sum
        if(doci == 0):
            pi += 1
            kurt_a.append(kurt)
            psimple_sum.append(pos_sum)
        else:
            ni += 1
            #ksum += kurt
            ksum_a.append(kurt)
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
    #print("ttest =", stats.wilcoxon(np_kurt_a[:4000], np_kurt_na.flatten()[:4000]))
    print("ttest =", stats.mannwhitneyu(np_kurt_a[:4000], np_kurt_na.flatten()[:4000]))
    print("ttest =", stats.mannwhitneyu(np_psimple_sum[:4500], np_nsimple_sum.flatten()[:4500]))
    print(np_psimple_sum.shape)
    print(np_nsimple_sum.shape)

    make_boxgraph(np_kurt_a, np_kurt_na, 'kurt')    ## kurt box plot
#    make_boxgraph(np_psimple_sum, np_nsimple_sum, 'camsum')
#    make_bargraph(np_psimple_sum, np_nsimple_sum)
    #make_graph(np.array(pcam_sum/pi), np.array(ncam_sum/ni))

def make_boxgraph(psum, nsum, name):
    plt.figure(figsize=(3,4))
    plt.boxplot([psum, nsum])
    plt.xticks([1, 2], ['TRUE', 'NOT TRUE'])
    plt.savefig("boxplot_"+name+".jpg", bbox_inches='tight', dpi=300)
  
def make_bargraph(psum, nsum):
    CTEs = [np.mean(psum), np.mean(nsum)]
    error = [np.std(psum), np.std(nsum)]
    fig, ax = plt.subplots()
 #   fig.figure(figsize=(10,4))
    ax.bar(['TRUE','NOT TRUE'], CTEs, yerr=error, align='center', alpha=0.5, ecolor='black')
    ax.set_ylabel('CAM value sum')
    #ax.set_title('Coefficent of Thermal Expansion (CTE) of Three Metals')
    #ax.yaxis.grid(True)
    
    # Save the figure and show
    plt.tight_layout()
    plt.savefig('barplot_camsum.png', dpi=300)

if __name__ == '__main__':
#    vocab = readVocab()

    #compare_ndcg(model)
    compare_kurt_posneg(model)

