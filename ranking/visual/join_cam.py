import torch
import cv2
import numpy as np


def show_cam_on_image(img, mask, imgname):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite(imgname, np.uint8(255*cam))

def preprocess_image(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = preprocessed_img.requires_grad_(False)
    return input

def readVocab():
    vocab = []
    fp = open('../vocab.index','r')
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

def print_topk_qd(tv, ti, vocab, topk, shape):
    for i in range(topk):
        mp = ti[-1*i-1]
        mv = tv[-1*i-1]
        max_q = mp // shape[1]
        max_d = mp % shape[1]
        print(i, mv, max_q, max_d, vocab[text_left[0,max_q]], vocab[text_right[0,max_d]])

if __name__ == '__main__':
    vocab = readVocab()
    num = 59  # +51  -59
    num -= 1
    model = "match_pyramid_ms"
    #model = "snrm_ms"
    fname = "../save_" + model + "/result_cam.pt" + str(num)
    res = torch.load(fname)

    cam = res['cam']
    cross = res['embcross']
    text_left = res['text_left']
    text_right = res['text_right']

    print(fname)
    print(res['text_left'])
    print(res['score'])
    print(cam.shape)
    print(cross.shape)
    #print(cam)

    ## cam mean
    tv, ti = torch.sort(cam.flatten())
    mean = torch.mean(cam)
    var = torch.var(cam)
    print('cam mean=', mean)
    print('cam var=', var)

    print('query = ', make_sentence(res['text_left'].squeeze()))
    print('doc = ', make_sentence(res['text_right'].squeeze()))
    #print_topk_qd(tv, ti, vocab, 200, cam.shape)

    ## cross
    tv, ti = torch.sort(cross.flatten())
    mean = torch.mean(cross)
    var = torch.var(cross)
    #print_topk_qd(tv, ti, vocab, 100, cam.shape)
    print('cross mean=', mean)
    print('cross var=', var)
    print('cross max=', tv[-1], ti[-1])
    print('cross max2=', tv[-2], ti[-2])
    print('cross min=', tv[0], ti[0])
    print('cross min2=', tv[1], ti[2])

    cam = cam.detach().cpu().numpy()
    cross = cross.detach().cpu().numpy()
    cam = np.maximum(cam,0)
    cam = cv2.resize(cam, (500, 200))

    cam = cam - np.min(cam)
    cam = cam / np.max(cam)

    imgname = model + "_join_" + str(num) + ".jpg"

    heatname = model + "_heat_" + str(num) + ".jpg"
    img = cv2.imread(heatname, cv2.IMREAD_COLOR)
    img = np.float32(cv2.resize(img, (500, 200))) / 255
    input = preprocess_image(img)

    show_cam_on_image(img, cam, imgname)


