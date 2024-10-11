import cv2
from matplotlib import pyplot as plt
import imutils

def save_grayscale_image_hist(image: cv2.typing.MatLike, image_name: str, hist_title: str):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])

    _, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].axis("off")
    axs[0].imshow(image, cmap='gray')

    axs[1].set_title('Histograma')
    axs[1].set_xlabel("intensidade do pixel")
    axs[1].set_ylabel("n° de pixels")
    axs[1].plot(hist)
    plt.xlim([0, 256])

    plt.savefig(f'out/histograms/{image_name}/{hist_title}.png')
    plt.close()

def generate_grayscale_hist(image_name: str):
    grayscale_image = cv2.imread('assets/' + image_name + '.jpg')
    grayscale_image = cv2.cvtColor(grayscale_image, cv2.COLOR_BGR2GRAY)
    save_grayscale_image_hist(grayscale_image, image_name, 'grayscale')

    equalized_grayscale_image = cv2.equalizeHist(grayscale_image)    
    save_grayscale_image_hist(equalized_grayscale_image, image_name, 'grayscale_equalized')


    negative_grayscale_image = abs(255-grayscale_image)
    save_grayscale_image_hist(negative_grayscale_image, image_name, 'grayscale_negative')


def save_colored_image_hist(image: cv2.typing.MatLike, image_name: str, hist_title: str):
    chans = cv2.split(image)
    colors = ("b", "g", "r")

    _, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].axis("off")
    axs[0].imshow(imutils.opencv2matplotlib(image))

    axs[1].set_title('Histograma')
    axs[1].set_xlabel("intensidade do pixel")
    axs[1].set_ylabel("n° de pixels")

    for (chan, color) in zip(chans, colors):
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])

        axs[1].plot(hist, color=color)
        plt.xlim([0, 256])

    plt.savefig(f'out/histograms/{image_name}/{hist_title}.png')
    plt.close()

def generate_colored_hist(image_name: str):
    colored_image = cv2.imread('assets/' + image_name + '.jpg')
    save_colored_image_hist(colored_image, image_name, 'colored')


    equalized_colored_image = cv2.cvtColor(colored_image, cv2.COLOR_BGR2YUV)
    equalized_colored_image[:, :, 0] = cv2.equalizeHist(equalized_colored_image[:, :, 0])
    equalized_colored_image = cv2.cvtColor(equalized_colored_image, cv2.COLOR_YUV2BGR)
    save_colored_image_hist(equalized_colored_image, image_name, 'colored_equalized')


    negative_colored_image = abs(255-colored_image)
    save_colored_image_hist(negative_colored_image, image_name, 'colored_negative')

if __name__ == '__main__':
    images = ['centro_noite', 'pp', 'centro_dia', 'primeira_ponte']
    for image in images:
        generate_grayscale_hist(image)
        generate_colored_hist(image)
