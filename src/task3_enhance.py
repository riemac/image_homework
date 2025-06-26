#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
任务3：图像增强（去除底纹噪声，增加对比度）
"""

import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from scipy.ndimage import maximum_filter

# 添加项目根目录到路径，确保可以导入src中的模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.image_io import read_image, save_image
from utils.color_space import adjust_brightness, adjust_contrast
from utils.filters import bilateral_filter, gaussian_blur, apply_clahe, unsharp_mask, median_blur
import config as config


def display_images(images, titles, figsize=(15, 10)):
    """显示多张图像，便于比较"""
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    
    if n == 1:
        axes = [axes]
    
    for i, (image, title) in enumerate(zip(images, titles)):
        if len(image.shape) == 2 or image.shape[2] == 1:  # 灰度图
            axes[i].imshow(image, cmap='gray')
        else:  # RGB图
            axes[i].imshow(image)
        
        axes[i].set_title(title)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def display_fft_spectrum(image, title="FFT Spectrum", figsize=(10, 5)):
    """显示图像的FFT频谱"""
    # 确保图像是灰度图
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # 执行FFT
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    
    # 显示原图和频谱
    plt.figure(figsize=figsize)
    plt.subplot(121), plt.imshow(gray, cmap='gray')
    plt.title('input image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='viridis')
    plt.title(title), plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.show()
    
    return magnitude_spectrum


def detect_periodic_noise(image, threshold_factor=0.2, exclude_center_ratio=0.1, max_peaks=10):
    """
    检测图像中的周期性噪声
    
    参数:
    image: 输入图像
    threshold_factor: 峰值检测阈值因子（相对于最大值）
    exclude_center_ratio: 排除中心区域的比例（排除DC分量）
    
    返回:
    peaks: 检测到的峰值坐标列表 [(y1,x1), (y2,x2), ...]
    magnitude_spectrum: 幅度谱
    """
    # 确保图像是灰度图
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # 执行FFT
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.log(np.abs(fshift) + 1)
    
    # 创建掩码排除中心区域
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    center_mask = np.ones((rows, cols), dtype=bool)
    r = int(min(rows, cols) * exclude_center_ratio)
    y, x = np.ogrid[:rows, :cols]
    mask_area = (y - crow) ** 2 + (x - ccol) ** 2 <= r ** 2
    center_mask[mask_area] = False
    
    # 应用掩码并找出峰值
    masked_spectrum = magnitude_spectrum.copy()
    masked_spectrum[~center_mask] = 0
    
    # 设置阈值
    threshold = threshold_factor * np.max(masked_spectrum)

    # 用最大值滤波找局部极大值
    neighborhood = maximum_filter(masked_spectrum, size=3)
    peaks = np.argwhere((masked_spectrum == neighborhood) & (masked_spectrum > threshold))
    
    # 按强度排序，只保留最显著的max_peaks个
    if len(peaks) > 0:
        peak_values = masked_spectrum[peaks[:,0], peaks[:,1]]
        sorted_idx = np.argsort(peak_values)[::-1]
        peaks = peaks[sorted_idx][:max_peaks]
        peaks = [tuple(p) for p in peaks]
    else:
        peaks = []
    return peaks, magnitude_spectrum


def create_notch_filter(image_shape, peaks, radius=5):
    """
    创建陷波滤波器
    
    参数:
    image_shape: 图像形状 (height, width)
    peaks: 峰值坐标列表 [(y1,x1), (y2,x2), ...]
    radius: 陷波半径
    
    返回:
    notch_filter: 陷波滤波器掩码
    """
    rows, cols = image_shape
    crow, ccol = rows // 2, cols // 2
    
    # 使用meshgrid创建坐标网格，效率远高于for循环
    y, x = np.ogrid[:rows, :cols]
    
    # 初始化全1的掩码
    mask = np.ones((rows, cols), dtype=np.float32)
    
    for peak in peaks:
        py, px = peak
        # 计算所有点到当前峰值的距离
        d_sq = (y - py)**2 + (x - px)**2
        mask[d_sq <= radius**2] = 0
        
        # 对称点也置零
        sym_py, sym_px = 2 * crow - py, 2 * ccol - px
        if 0 <= sym_py < rows and 0 <= sym_px < cols:
            d_sym_sq = (y - sym_py)**2 + (x - sym_px)**2
            mask[d_sym_sq <= radius**2] = 0
            
    return mask


def apply_notch_filter(image, peaks, radius=5):
    """
    应用陷波滤波器去除周期性噪声
    
    参数:
    image: 输入图像
    peaks: 峰值坐标列表 [(y1,x1), (y2,x2), ...]
    radius: 陷波半径
    
    返回:
    filtered_image: 滤波后的图像
    """
    # 确保图像是灰度图
    if len(image.shape) > 2:
        is_color = True
        # 处理彩色图像的每个通道
        channels = cv2.split(image)
        filtered_channels = []
        
        for channel in channels:
            filtered_channel = apply_notch_filter_to_channel(channel, peaks, radius)
            filtered_channels.append(filtered_channel)
        
        # 合并通道
        filtered_image = cv2.merge(filtered_channels)
    else:
        is_color = False
        filtered_image = apply_notch_filter_to_channel(image, peaks, radius)
    
    return filtered_image


def apply_notch_filter_to_channel(channel, peaks, radius=5):
    """对单个通道应用陷波滤波器"""
    rows, cols = channel.shape
    
    # 执行FFT
    f = np.fft.fft2(channel)
    fshift = np.fft.fftshift(f)
    
    # 创建陷波滤波器
    mask = create_notch_filter((rows, cols), peaks, radius)
    
    # 应用滤波器
    fshift_filtered = fshift * mask
    
    # 反变换回空域
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    
    # 归一化到0-255
    img_back = np.clip(img_back, 0, 255).astype(np.uint8)
    
    return img_back


def enhance_image(image, 
                  noise_threshold=0.2, 
                  notch_radius=5, 
                  bilateral_d=9, 
                  bilateral_sigma_color=75, 
                  bilateral_sigma_space=75,
                  clahe_clip_limit=3.0, 
                  clahe_grid_size=(8, 8),
                  unsharp_strength=1.5,
                  median_size=3,
                  contrast_factor=1.3):
    """
    综合图像增强：去除底纹噪声并增强对比度
    
    参数:

    image: 输入图像
    noise_threshold: 噪声检测阈值
    notch_radius: 陷波滤波器半径
    bilateral_d: 双边滤波的直径
    bilateral_sigma_color: 双边滤波的颜色标准差
    bilateral_sigma_space: 双边滤波的空间标准差
    clahe_clip_limit: CLAHE的对比度限制
    clahe_grid_size: CLAHE的网格大小
    unsharp_strength: 锐化强度
    median_size: 中值滤波核大小
    contrast_factor: 对比度增强因子
    
    返回:

    enhanced_image: 增强后的图像
    """
    # 步骤1: 检测周期性噪声
    peaks, _ = detect_periodic_noise(image, threshold_factor=noise_threshold)
    
    # 步骤2: 应用陷波滤波器去除周期性噪声
    if peaks:
        filtered_image = apply_notch_filter(image, peaks, radius=notch_radius)
    else:
        filtered_image = image.copy()
    
    # 步骤3: 应用中值滤波去除残余噪点
    median_filtered = median_blur(filtered_image, kernel_size=median_size)
    
    # 步骤4: 应用双边滤波平滑图像同时保持边缘
    bilateral_filtered = bilateral_filter(median_filtered, 
                                         d=bilateral_d, 
                                         sigma_color=bilateral_sigma_color, 
                                         sigma_space=bilateral_sigma_space)
    
    # 步骤5: 应用CLAHE增强对比度
    clahe_enhanced = apply_clahe(bilateral_filtered, 
                                clip_limit=clahe_clip_limit, 
                                tile_grid_size=clahe_grid_size)
    
    # 步骤6: 应用锐化增强细节
    sharpened = unsharp_mask(clahe_enhanced, strength=unsharp_strength)
    
    # 步骤7: 最终对比度调整
    enhanced_image = adjust_contrast(sharpened, factor=contrast_factor)
    
    return enhanced_image


def main():
    """主函数，用于命令行运行"""
    parser = argparse.ArgumentParser(description='图像增强：去除底纹噪声并增强对比度')
    parser.add_argument('--input', type=str, default=config.SCENERY_IMAGE_PATH,
                        help='输入图像路径')
    parser.add_argument('--output', type=str, default=config.SCENERY_OUTPUT_PATH,
                        help='输出图像路径')
    parser.add_argument('--show', action='store_true',
                        help='显示处理前后的对比图')
    parser.add_argument('--show-fft', action='store_true',
                        help='显示FFT频谱')
    parser.add_argument('--noise-threshold', type=float, default=0.2,
                        help='噪声检测阈值')
    parser.add_argument('--notch-radius', type=int, default=5,
                        help='陷波滤波器半径')
    parser.add_argument('--bilateral-d', type=int, default=9,
                        help='双边滤波的直径')
    parser.add_argument('--bilateral-sigma-color', type=float, default=75,
                        help='双边滤波的颜色标准差')
    parser.add_argument('--bilateral-sigma-space', type=float, default=75,
                        help='双边滤波的空间标准差')
    parser.add_argument('--clahe-clip-limit', type=float, default=3.0,
                        help='CLAHE的对比度限制')
    parser.add_argument('--clahe-grid-size', type=int, default=8,
                        help='CLAHE的网格大小')
    parser.add_argument('--unsharp-strength', type=float, default=1.5,
                        help='锐化强度')
    parser.add_argument('--median-size', type=int, default=3,
                        help='中值滤波核大小')
    parser.add_argument('--contrast-factor', type=float, default=1.3,
                        help='对比度增强因子')
    
    args = parser.parse_args()
    
    # 读取图像
    image = read_image(args.input)
    
    # 显示FFT频谱
    if args.show_fft:
        display_fft_spectrum(image)
    
    # 应用图像增强
    enhanced = enhance_image(
        image,
        noise_threshold=args.noise_threshold,
        notch_radius=args.notch_radius,
        bilateral_d=args.bilateral_d,
        bilateral_sigma_color=args.bilateral_sigma_color,
        bilateral_sigma_space=args.bilateral_sigma_space,
        clahe_clip_limit=args.clahe_clip_limit,
        clahe_grid_size=(args.clahe_grid_size, args.clahe_grid_size),
        unsharp_strength=args.unsharp_strength,
        median_size=args.median_size,
        contrast_factor=args.contrast_factor
    )
    
    # 保存结果
    save_image(enhanced, args.output)
    print(f"增强后的图像已保存至: {args.output}")
    
    # 显示对比图
    if args.show:
        display_images([image, enhanced], ['原始图像', '增强后图像'])


if __name__ == "__main__":
    main()