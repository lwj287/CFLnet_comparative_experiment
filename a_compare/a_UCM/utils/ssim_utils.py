import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from glob import glob
from os.path import join
from ntpath import basename

def compute_ssim(X, Y):
    """
    计算单通道的结构相似性(SSIM)
    """
    # 初始化论文中建议的参数
    K1 = 0.01  # 用于稳定除法的常数
    K2 = 0.03  # 用于稳定除法的常数
    sigma = 1.5  # 高斯滤波器的标准差
    win_size = 5  # 窗口大小

    # 计算均值
    ux = gaussian_filter(X, sigma)  # 对图像X进行高斯滤波得到均值
    uy = gaussian_filter(Y, sigma)  # 对图像Y进行高斯滤波得到均值

    # 计算方差和协方差
    uxx = gaussian_filter(X * X, sigma)  # X平方的高斯滤波结果
    uyy = gaussian_filter(Y * Y, sigma)  # Y平方的高斯滤波结果
    uxy = gaussian_filter(X * Y, sigma)  # X*Y的高斯滤波结果

    # 用无偏估计标准化标准差
    N = win_size ** X.ndim  # 窗口中的像素数量
    unbiased_norm = N / (N - 1)  # 论文中的公式4,无偏归一化系数
    vx  = (uxx - ux * ux) * unbiased_norm  # X的方差
    vy  = (uyy - uy * uy) * unbiased_norm  # Y的方差
    vxy = (uxy - ux * uy) * unbiased_norm  # X和Y的协方差

    R = 255  # 图像像素的最大可能值
    C1 = (K1 * R) ** 2  # 用于避免除以零的常数
    C2 = (K2 * R) ** 2  # 用于避免除以零的常数
    
    # 计算SSIM(论文中的公式13)
    sim = (2 * ux * uy + C1) * (2 * vxy + C2)
    D = (ux ** 2 + uy ** 2 + C1) * (vx + vy + C2)
    SSIM = sim / D
    mssim = SSIM.mean()

    return mssim

def getSSIM(X, Y):
    """
    计算两张图像之间的平均结构相似性
    """
    assert (X.shape == Y.shape), "提供的图像维度不同"  # X和Y要求为np数组,才有shape属性,image对象没有shape属性
    nch = 1 if X.ndim == 2 else X.shape[-1]
    mssim = []
    for ch in range(nch):
        Xc, Yc = X[..., ch].astype(np.float32), Y[..., ch].astype(np.float32)
        mssim.append(compute_ssim(Xc, Yc))
    return np.mean(mssim)

def measure_SSIMs(gtr_dir, enh_dir, im_res=(256, 256)):
    """
    gtr_dir 包含基准图像(ground-truths)的文件夹路径
    enh_dir 包含增强后图像的文件夹路径
    """
    gtr_paths = sorted(glob(join(gtr_dir, "*.*")))
    enh_paths = sorted(glob(join(enh_dir, "*.*")))
    ssims = []
    for gtr_path, enh_path in zip(gtr_paths, enh_paths):
        gtr_fn = basename(gtr_path).split('.')[0]
        enh_fn = basename(enh_path).split('.')[0]
        if gtr_fn == enh_fn:
            r_im = Image.open(gtr_path).resize(im_res)
            g_im = Image.open(enh_path).resize(im_res)
            ssim = getSSIM(np.array(r_im), np.array(g_im))
            ssims.append(ssim)
    return np.array(ssims)