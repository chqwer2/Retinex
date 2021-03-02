import numpy as np
from PIL import Image
from scipy.signal import convolve2d
from skimage.measure import compare_ssim, compare_psnr
import cv2

def _as_floats(image0, image1):
    """
    Promote im1, im2 to nearest appropriate floating point precision.
    """
    float_type = np.result_type(image0.dtype, image1.dtype, np.float32)
    image0 = np.asarray(image0, dtype=float_type)
    image1 = np.asarray(image1, dtype=float_type)
    return image0, image1


def PSNR(y_true, y_pred):

    # err = mean_squared_error(image_true, image_test)
    y_true, y_pred = _as_floats(y_true, y_pred)
    out = 10 * np.log10((255**2) / np.mean(np.square(y_pred - y_true), dtype=np.float64))
    return np.mean(out)


def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def filter2(x, kernel, mode='same'):
    # x = np.squeeze(x)
    out = np.zeros(x.shape)

    for i in range(3):
        out[0, :, :, i] = convolve2d(x[0,:,:,i], np.rot90(kernel, 2), mode='same')
    return out


def compute_ssim(im1, im2, k1=0.01, k2=0.03, win_size=11, L=255):
    if not im1.shape == im2.shape:
        raise ValueError("Input Imagees must have the same dimensions")
    # if len(im1.shape) > 2:
    #     raise ValueError("Please input the images with 1 channel")

    M, N = im1.shape[1:3]
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    window = matlab_style_gauss2D(shape=(win_size, win_size), sigma=1.5)
    window = window / np.sum(np.sum(window))

    if im1.dtype == np.uint8:
        im1 = np.double(im1)
    if im2.dtype == np.uint8:
        im2 = np.double(im2)

    mu1 = filter2(im1, window, 'valid')
    mu2 = filter2(im2, window, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1 * im1, window, 'valid') - mu1_sq
    sigma2_sq = filter2(im2 * im2, window, 'valid') - mu2_sq
    sigmal2 = filter2(im1 * im2, window, 'valid') - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigmal2 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return np.mean(np.mean(ssim_map))


def sk_psnr(im1, im2):
    # blur = cv2.GaussianBlur(origin, (5, 5), 0)
    psnr_dark = compare_psnr(im1, im2, data_range=255)
    return psnr_dark

def sk_ssim(im1, im2):
    ssim_dark = compare_ssim(im1, im2, multichannel=True)
    return ssim_dark

if __name__ == "__main__":
    im1 = Image.open("data/low/1.png")
    im2 = Image.open("data/high/1.png")
    print(np.array(im1).shape)

    print("sk SSIM score:", sk_ssim(np.array(im1), np.array(im2)))
    print("SSIM score:", compute_ssim(np.expand_dims(np.array(im1),axis=0), np.expand_dims(np.array(im2),axis=0)))
    print("sk psnr score:", sk_psnr(np.array(im1), np.array(im2)))
    print("PSNR score:", PSNR(np.expand_dims(np.array(im1), axis=0), np.expand_dims(np.array(im2), axis=0)))
