import torch
import time

start = time.time()

loader = DataLoader(dataset, batch_size=10)
src = iter(loader).next()[0].numpy()
print(src.shape)

def convolution(img, kernel):
    n, H, W, iC = input.shape
    oC, _, kH, kW = kernel.shape

    padding = kernel.shape[-1] // 2
    pass

def sobel(img):
    pass


b, h, w, c = src.shape
Gx = np.zeros((b, h-1, w-1))
Gy = np.zeros((b, h-1, w-1))

N = np.array(
    [[-1,0,1],
     [-2,0,2],
     [-1,0,1]]
)

Sx = np.array(
    [[-1,0,1],
     [-2,0,2],
     [-1,0,1]]
)
Sy = Sx.transpose(0,1)

for i in range(h-1-3):
    for j in range(w-1-3):
        Gx[:,i,j] = np.sum(src[:, i:i+3, j:j+3, :] * Sx[None,:,:, None], axis=(1,2,3))

for i in range(h-1-3):
    for j in range(w-1-3):
        Gy[:,i,j] = np.sum(src[:, i:i+3, j:j+3, :] * Sy[None,:,:, None], axis=(1,2,3))

M = (Gx**2 + Gy**2)**0.5
M = (M-M.min())/(M.max()-M.min())*255
