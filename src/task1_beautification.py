#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
任务1：美女图像美化（去斑，美白）
"""

import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 添加项目根目录到路径，确保可以导入src中的模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.image_io import read_image, save_image
from utils.color_space import adjust_brightness, adjust_contrast, adjust_saturation
from utils.filters import gaussian_blur, bilateral_filter
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


def skin_detection(image):
    """
    检测皮肤区域
    返回皮肤区域的掩码
    """
    # 转换到YCrCb颜色空间，该空间更适合皮肤检测
    image_ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    
    # 皮肤在YCrCb空间的范围（根据经验值）
    min_YCrCb = np.array([0, 133, 77], np.uint8)
    max_YCrCb = np.array([255, 173, 127], np.uint8)
    
    # 创建掩码
    skin_mask = cv2.inRange(image_ycrcb, min_YCrCb, max_YCrCb)
    
    # 应用形态学操作来改善掩码
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    
    # # 扩大掩码区域以确保覆盖所有皮肤
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    # skin_mask = cv2.dilate(skin_mask, kernel, iterations=1)

    # 改为适度的膨胀，并添加高斯模糊以软化边缘
    skin_mask = cv2.GaussianBlur(skin_mask, (7, 7), sigmaX=2)  # 添加高斯模糊
    
    return skin_mask


def remove_spots(image, strength=0.8, d=9, sigma_color=75, sigma_space=75):
    """
    去除脸部斑点
    
    参数:
    image: 输入图像，RGB格式
    strength: 去斑强度，0-1之间的值
    d: 双边滤波的直径
    sigma_color: 双边滤波的颜色标准差
    sigma_space: 双边滤波的空间标准差
    
    返回:
    处理后的图像
    """
    # 检测皮肤区域
    skin_mask = skin_detection(image)
    
    # 复制原图，用于结果
    result = image.copy()
    
    # 对皮肤区域应用双边滤波，保留边缘的同时平滑斑点
    # 在皮肤区域应用强度可调的双边滤波
    filtered = bilateral_filter(image, d=d, sigma_color=sigma_color, sigma_space=sigma_space)
    
    # # 使用皮肤掩码将滤波后的图像与原图融合
    # mask_3d = cv2.cvtColor(skin_mask, cv2.COLOR_GRAY2RGB) / 255.0

    # 使用浮点型掩码进行加权融合（而非二值掩码）
    mask_3d = cv2.cvtColor(skin_mask, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
    
    # 根据强度参数调整融合程度
    result = (1 - strength) * result + strength * (mask_3d * filtered + (1 - mask_3d) * result)
    result = result.astype(np.uint8)
    
    return result


def skin_whitening(image, brightness_factor=1.15, contrast_factor=1.1, saturation_factor=0.95):
    """
    美白肤色
    
    参数:
    image: 输入图像，RGB格式
    brightness_factor: 亮度因子
    contrast_factor: 对比度因子
    saturation_factor: 饱和度因子
    
    返回:
    处理后的图像
    """
    # 检测皮肤区域
    skin_mask = skin_detection(image)
    mask_3d = cv2.cvtColor(skin_mask, cv2.COLOR_GRAY2RGB) / 255.0
    
    # 复制原图，用于结果
    result = image.copy()
    
    # 对皮肤区域进行美白处理
    # 1. 调整亮度
    brightened = adjust_brightness(image, brightness_factor)
    
    # 2. 调整对比度
    brightened_contrast = adjust_contrast(brightened, contrast_factor)
    
    # 3. 调整饱和度（降低饱和度使肤色看起来更白皙）
    whitened = adjust_saturation(brightened_contrast, saturation_factor)
    
    # 只在皮肤区域应用美白效果
    result = mask_3d * whitened + (1 - mask_3d) * image
    result = result.astype(np.uint8)
    
    return result


def beautify_image(image, 
                   remove_spots_strength=0.8, d=9, sigma_color=75, sigma_space=75,
                   brightness_factor=1.15, contrast_factor=1.1, saturation_factor=0.95):
    """
    综合美化功能：去斑+美白
    
    参数:
    image: 输入图像，RGB格式
    remove_spots_strength: 去斑强度
    brightness_factor: 亮度因子
    contrast_factor: 对比度因子
    saturation_factor: 饱和度因子
    
    返回:
    处理后的图像
    """
    # 先去斑
    no_spots_image = remove_spots(image, remove_spots_strength, d=d, sigma_color=sigma_color, sigma_space=sigma_space)
    
    # 再美白
    beautified_image = skin_whitening(no_spots_image, 
                                     brightness_factor, 
                                     contrast_factor, 
                                     saturation_factor)
    
    return beautified_image


def main():
    """主函数，用于命令行运行"""
    parser = argparse.ArgumentParser(description='美女图像美化：去斑和美白')
    parser.add_argument('--input', type=str, default=config.BEAUTY_IMAGE_PATH,
                        help='输入图像路径')
    parser.add_argument('--output', type=str, default=config.BEAUTY_OUTPUT_PATH,
                        help='输出图像路径')
    parser.add_argument('--show', action='store_true',
                        help='显示处理前后的对比图')
    parser.add_argument('--spots-strength', type=float, default=0.8,
                        help='去斑强度，0-1之间')
    parser.add_argument('--brightness', type=float, default=1.15,
                        help='美白亮度因子')
    parser.add_argument('--contrast', type=float, default=1.1,
                        help='美白对比度因子')
    parser.add_argument('--saturation', type=float, default=0.95,
                        help='美白饱和度因子')
    
    args = parser.parse_args()
    
    # 读取图像
    image = read_image(args.input)
    
    # 应用美化
    beautified = beautify_image(image, 
                               args.spots_strength,
                               args.brightness, 
                               args.contrast, 
                               args.saturation)
    
    # 保存结果
    save_image(beautified, args.output)
    print(f"美化后的图像已保存至: {args.output}")
    
    # 显示对比图
    if args.show:
        display_images([image, beautified], ['原始图像', '美化后图像'])


if __name__ == "__main__":
    main()