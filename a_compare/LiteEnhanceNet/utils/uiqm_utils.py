import numpy as np
import math
from scipy import ndimage
from PIL import Image
from glob import glob
from os.path import join

def mu_a(x, alpha_L=0.1, alpha_R=0.1):
    """
      计算非对称alpha修剪均值
    """
    # 按强度对像素排序,用于裁剪
    x = sorted(x)
    # 获取像素数量
    K = len(x)
    # 计算 T alpha L和 T alpha R
    T_a_L = math.ceil(alpha_L * K)
    T_a_R = math.floor(alpha_R * K)
    # 计算mu_alpha权重
    weight = (1 / (K - T_a_L - T_a_R))
    # 遍历从 T_a_L+1 到 K-T_a_R 的扁平化图像
    s = int(T_a_L + 1)
    e = int(K - T_a_R)
    val = sum(x[s:e])
    val = weight * val
    return val

def s_a(x, mu):
    """计算非对称alpha修剪方差"""
    val = 0
    for pixel in x:
        val += math.pow((pixel - mu), 2)
    return val / len(x)

def _uicm(x):
    """计算水下图像颜色度量(UIQM中的颜色分量)"""
    # 提取RGB通道并扁平化
    R = x[:, :, 0].flatten()
    G = x[:, :, 1].flatten()
    B = x[:, :, 2].flatten()
    # 计算RG和YB色差分量
    RG = R - G
    YB = ((R + G) / 2) - B
    # 计算色差分量的alpha修剪均值
    mu_a_RG = mu_a(RG)
    mu_a_YB = mu_a(YB)
    # 计算色差分量的方差
    s_a_RG = s_a(RG, mu_a_RG)
    s_a_YB = s_a(YB, mu_a_YB)
    # 计算颜色度量值
    l = math.sqrt((math.pow(mu_a_RG, 2) + math.pow(mu_a_YB, 2)))
    r = math.sqrt(s_a_RG + s_a_YB)
    return (-0.0268 * l) + (0.1586 * r)

def sobel(x):
    """应用Sobel边缘检测"""
    dx = ndimage.sobel(x, 0)  # x方向边缘检测
    dy = ndimage.sobel(x, 1)  # y方向边缘检测
    mag = np.hypot(dx, dy)  # 计算梯度 magnitude
    mag *= 255.0 / np.max(mag)  # 归一化到0-255范围
    return mag

def eme(x, window_size):
    """
      增强度量估计(Enhancement Measure Estimation)
      x.shape[0] = 高度
      x.shape[1] = 宽度
    """
    # 计算窗口数量,如4块则为2x2等
    k1 = int(x.shape[1] / window_size)  # 这里做了int处理
    k2 = int(x.shape[0] / window_size)
    # 权重
    w = 2. / (k1 * k2)
    blocksize_x = window_size
    blocksize_y = window_size
    # 确保图像尺寸可被窗口大小整除,裁剪部分像素不影响结果
    x = x[:blocksize_y * k2, :blocksize_x * k1]
    val = 0
    # 遍历所有窗口块
    for l in range(k1):
        for k in range(k2):
            block = x[k * window_size:window_size * (k + 1), l * window_size:window_size * (l + 1)]
            max_ = np.max(block)  # 块内最大值
            min_ = np.min(block)  # 块内最小值
            # 边界检查,避免log(0)错误
            if min_ == 0.0:
                val += 0
            elif max_ == 0.0:
                val += 0
            else:
                val += math.log(max_ / min_)
    return w * val

def _uism(x):
    """
      水下图像清晰度度量(Underwater Image Sharpness Measure)
    """
    # 获取图像通道
    R = x[:, :, 0]
    G = x[:, :, 1]
    B = x[:, :, 2]
    # 对每个RGB分量应用Sobel边缘检测器
    Rs = sobel(R)
    Gs = sobel(G)
    Bs = sobel(B)
    # 将每个通道检测到的边缘与通道本身相乘
    R_edge_map = np.multiply(Rs, R)
    G_edge_map = np.multiply(Gs, G)
    B_edge_map = np.multiply(Bs, B)
    # 计算每个通道的eme值
    r_eme = eme(R_edge_map, 10)
    g_eme = eme(G_edge_map, 10)
    b_eme = eme(B_edge_map, 10)
    # 加权系数(类似RGB转灰度的系数)
    lambda_r = 0.299
    lambda_g = 0.587
    lambda_b = 0.144
    return (lambda_r * r_eme) + (lambda_g * g_eme) + (lambda_b * b_eme)

def _uiconm(x, window_size):
    """
      水下图像对比度度量(Underwater image contrast measure)
      参考: https://github.com/tkrahn108/UIQM/blob/master/src/uiconm.cpp
      论文: https://ieeexplore.ieee.org/abstract/document/5609219
    """
    # 计算窗口数量
    k1 = int(x.shape[1] / window_size)  # 这里也做了int处理
    k2 = int(x.shape[0] / window_size)
    # 权重
    w = -1. / (k1 * k2)
    blocksize_x = window_size
    blocksize_y = window_size
    # 确保图像尺寸可被窗口大小整除
    x = x[:blocksize_y * k2, :blocksize_x * k1]
    # 熵缩放因子,较高的值有助于随机性
    alpha = 1
    val = 0
    # 遍历所有窗口块
    for l in range(k1):
        for k in range(k2):
            block = x[k * window_size:window_size * (k + 1), l * window_size:window_size * (l + 1), :]
            max_ = np.max(block)  # 块内最大值
            min_ = np.min(block)  # 块内最小值
            top = max_ - min_
            bot = max_ + min_
            # 避免除零和NaN值
            if math.isnan(top) or math.isnan(bot) or bot == 0.0 or top == 0.0:
                val += 0.0
            else:
                val += alpha * math.pow((top / bot), alpha) * math.log(top / bot)
    return w * val

def getUIQM(x):
    """
      UIQM计算函数
      x: 输入图像
    """
    x = x.astype(np.float32)
    ### UCIQE: https://ieeexplore.ieee.org/abstract/document/7300447
    #c1 = 0.4680; c2 = 0.2745; c3 = 0.2576
    ### UIQM 论文参数: https://ieeexplore.ieee.org/abstract/document/7305804
    c1 = 0.0282; c2 = 0.2953; c3 = 3.5753
    uicm = _uicm(x)  # 计算颜色度量
    uism = _uism(x)  # 计算清晰度度量
    uiconm = _uiconm(x, 10)  # 计算对比度度量
    uiqm = (c1 * uicm) + (c2 * uism) + (c3 * uiconm)
    return uiqm

def measure_UIQMs(dir_name, im_res=(256, 256)):
    """计算指定目录中所有图像的UIQM值"""
    paths = sorted(glob(join(dir_name, "*.*")))
    uqims = []
    for img_path in paths:
        im = Image.open(img_path).resize(im_res)
        uiqm = getUIQM(np.array(im))  # 计算UIQM值
        uqims.append(uiqm)
    return np.array(uqims)