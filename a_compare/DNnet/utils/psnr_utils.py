import numpy as np
import math
from PIL import Image
from glob import glob
from os.path import join
from ntpath import basename

def getPSNR(X, Y):
    """ 计算两张图片的峰值信噪比(PSNR),输入是RGB图像 """
    gtr_data = np.array(X, dtype=np.float32)
    enh_data = np.array(Y, dtype=np.float32)
    diff = enh_data - gtr_data
    diff = diff.flatten('C')
    rmse = math.sqrt(np.mean(diff **2.))
    if rmse == 0:
        return 100
    else:
        return 20 * math.log10(255.0 / rmse)

def measure_PSNRs(gtr_dir, enh_dir, im_res=(256, 256)):
    """
    gtr_dir 包含基准图像(ground-truths)的文件夹路径
    enh_dir 包含增强后图像的文件夹路径
    """
    gtr_paths = sorted(glob(join(gtr_dir, "*.*")))
    enh_paths = sorted(glob(join(enh_dir, "*.*")))
    psnrs = []
    for gtr_path, enh_path in zip(gtr_paths, enh_paths):  
        gtr_fn = basename(gtr_path).split('.')[0]
        enh_fn = basename(enh_path).split('.')[0]
        if (gtr_fn == enh_fn):
            g_im = Image.open(gtr_path).resize(im_res)
            e_im = Image.open(enh_path).resize(im_res)
            # 将图像转换为灰度图(L通道)进行PSNR计算(这是当前的最佳实践)
            # g_im = g_im.convert("L")
            # e_im = e_im.convert("L")
            psnr = getPSNR(np.array(g_im), np.array(e_im))
            psnrs.append(psnr)
    return np.array(psnrs)