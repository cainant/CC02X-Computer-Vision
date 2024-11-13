import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils

figsize=(30, 25)

def canny(image_name):
    src = cv2.imread(f'assets/{image_name}.jpg')
    src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    dst = cv2.Canny(src, 50, 200, None, 3)            

    _, axs = plt.subplots(1, 2, figsize=figsize)
    axs[0].axis('off')
    axs[1].axis('off')
    axs[0].imshow(imutils.opencv2matplotlib(src))
    axs[1].imshow(imutils.opencv2matplotlib(dst))

    plt.savefig(f'out/canny_{image_name}.jpg')
    plt.close()

imgs = ['messi']
for img in imgs:
    canny(img)
