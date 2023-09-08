import numpy as np

def convert_type(img):
    if type(img) != np.ndarray:
        return img.detach().cpu().numpy()
    else:
        return img


def calc_psnr(img1, img2):
    
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')  # Images are identical, PSNR is infinity.
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr