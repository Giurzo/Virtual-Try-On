import cv2
import torch
import math
import matplotlib.pyplot as plt

def hough_space(img):
    h, w  = img.shape
    max_d = (h**2+w**2)**0.5
    hough = torch.zeros((180, int(max_d)))

    i_mat = torch.arange(h)[:,None,None]
    j_mat = torch.arange(w)[None,:,None]
    t_mat = torch.arange(180)[None,None,:]

    roh = i_mat * torch.cos(t_mat / 180 * math.pi) + j_mat * torch.sin(t_mat / 180 * math.pi)
    roh = roh.long()

    for t in range(180):
        unique, counts = torch.unique(roh[:,:,t][img != 0], return_counts=True)
        hough[t,unique] += counts

    return hough


def hough_space_par(img):
    b, h, w  = img.shape
    max_d = (h**2+w**2)**0.5
    hough = np.zeros((180, int(max_d)))

    i_mat = np.arange(h)[None,:,None,None]
    j_mat = np.arange(w)[None,None,:,None]
    t_mat = np.arange(180)[None,None,None,:]

    roh = i_mat * np.cos(t_mat / 180 * math.pi) + j_mat * np.sin(t_mat / 180 * math.pi)
    roh = roh.astype(int)

    for t in range(180):
        for b_ in range(b):
            unique, counts = np.unique(roh[b,:,:,t][img != 0], return_counts=True)
            hough[b_,t,unique] += counts

    return hough


def hough_space_limited(img):
    hough = hough_space(img)
    T, D = hough.shape

    t_max = 20
    d_max = 100

    for t in range(T//t_max):
        for d in range(D//d_max):
            window = hough[t*t_max:(t+1)*t_max,d*d_max:(d+1)*d_max].copy().flatten()
            mask = np.zeros_like(window)
            mask[window.argmax()] = 1
            window *= mask
            hough[t*t_max:(t+1)*t_max,d*d_max:(d+1)*d_max] = window.reshape(t_max,d_max)

    return hough


def hough_space_print_lines(img):
    src = img.copy()

    hough = hough_space(img)
    idx = hough.argsort(axis=None)[-100:]
    #treshold = 75
    #idx = idx[h.flatten()[idx]>treshold]

    theta = idx / hough.shape[1]
    dist = idx % hough.shape[1]
    print(theta,dist)

    for t, d in zip(theta, dist):
        cos_v = math.cos(t / 180 * math.pi)
        sin_v = math.sin(t / 180 * math.pi)
        x0 = sin_v * d
        y0 = cos_v * d
        pt1 = (int(x0 + 10000*cos_v), int(y0 - 10000*sin_v))
        pt2 = (int(x0 - 10000*cos_v), int(y0 + 10000*sin_v))
        cv2.line(src, pt1, pt2, (0,0,255), 1, cv2.LINE_AA)
        cv2.circle(src, (int(x0),int(y0)), 10, (0,255,255), 10)
    
    plt.imshow(src, aspect="auto")
    plt.show()

def hough_space_print(img):
    hough = hough_space(img)
    hough = hough / hough.max() * 255
    plt.imshow(hough, aspect="auto")
    plt.show()

from dataset_canny import dataset
src = cv2.imread("dataset/upper_body/images/denim_denim-shirts/12141279ui_0_r.jpg")
src = cv2.Canny(src, 50, 200)
H = hough_space(src)
H = torch.log(H+1)
cv2.imshow("1v", src/src.max())
cv2.imshow("1", H.numpy()/H.numpy().max())

src = cv2.imread("dataset/upper_body/images/denim_denim-shirts/38814303li_1_f.jpg")
src = cv2.Canny(src, 50, 200)
H = hough_space(src)
H = torch.log(H+1)
cv2.imshow("2v", src/src.max())
cv2.imshow("2", H.numpy()/H.numpy().max())

cv2.waitKey()
cv2.destroyAllWindows()