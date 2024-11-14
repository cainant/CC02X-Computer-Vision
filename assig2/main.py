import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils

figsize=(30, 25)

def plot_and_save(fig1, fig2, path):
    _, axs = plt.subplots(1, 2, figsize=figsize)
    axs[0].axis('off')
    axs[1].axis('off')
    axs[0].imshow(imutils.opencv2matplotlib(fig1))
    axs[1].imshow(imutils.opencv2matplotlib(fig2))

    plt.savefig(path)
    plt.close()
    pass

def canny(image_name):
    src = cv2.imread(f'assets/{image_name}.jpg')
    src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    dst = cv2.Canny(src, 50, 200, None, 3)

    plot_and_save(src, dst, f'out/canny_{image_name}.jpg')

def color_canny(image_name):
    src = cv2.imread(f'assets/{image_name}.jpg')
    canny_b = cv2.Canny(src[:, :, 0], 100, 200)
    canny_g = cv2.Canny(src[:, :, 1], 100, 200)
    canny_r = cv2.Canny(src[:, :, 2], 100, 200)
    canny_color = cv2.merge([canny_b, canny_g, canny_r])

    plot_and_save(src, canny_color, f'out/canny_color_{image_name}.jpg')

def laplacian(image_name):
    src = cv2.imread(f'assets/{image_name}.jpg')
    src = cv2.GaussianBlur(src, (3, 3), 0)
    src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    dst = cv2.Laplacian(src, ddepth=2, ksize=3)
    plot_and_save(src, dst, f'out/laplace_{image_name}.jpg')

def sobel(image_name):
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S
    src = cv2.imread(f'assets/{image_name}.jpg')
    src = cv2.GaussianBlur(src, (3, 3), 0)
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    
    
    grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    
    
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    plot_and_save(src, grad, f'out/sobel_{image_name}.jpg')


def color_laplace(image_name):
    src = cv2.imread(f'assets/{image_name}.jpg')
    laplaciano_b = cv2.Laplacian(src[:, :, 0], cv2.CV_64F, ksize=3)
    laplaciano_g = cv2.Laplacian(src[:, :, 1], cv2.CV_64F, ksize=3)
    laplaciano_r = cv2.Laplacian(src[:, :, 2], cv2.CV_64F, ksize=3)

    laplaciano_color = cv2.merge([laplaciano_b, laplaciano_g, laplaciano_r])
    laplaciano_color = np.uint8(np.absolute(laplaciano_color))

    plot_and_save(src, laplaciano_color, f'out/laplaciano_color_{image_name}.jpg')

def color_sobel(image_name):
    src = cv2.imread(f'assets/{image_name}.jpg')
    laplaciano_b = cv2.Sobel(src[:, :, 0], cv2.CV_64F, 1, 0, ksize=3)
    laplaciano_g = cv2.Sobel(src[:, :, 1], cv2.CV_64F, 1, 0, ksize=3)
    laplaciano_r = cv2.Sobel(src[:, :, 2], cv2.CV_64F, 1, 0, ksize=3)

    laplaciano_color = cv2.merge([laplaciano_b, laplaciano_g, laplaciano_r])
    laplaciano_color = np.uint8(np.absolute(laplaciano_color))

    plot_and_save(src, laplaciano_color, f'out/sobel_color_{image_name}.jpg')

imgs = ['messi']
for img in imgs:
    canny(img)
    laplacian(img)
    sobel(img)
    color_laplace(img)
    color_sobel(img)
    color_canny(img)
