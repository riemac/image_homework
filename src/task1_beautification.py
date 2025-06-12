"""
美女图像美化模块
实现对美女图像的美化处理
"""
import os
import cv2
import numpy as np

from utils.image_io import read_image, save_image
from utils.color_space import adjust_brightness, adjust_contrast, adjust_saturation
from utils.filters import bilateral_filter, unsharp_mask, apply_clahe
import config

def beautify_image(image_path=None, output_path=None, params=None):
    """
    美化图像处理主函数
    
    参数:
        image_path (str): 输入图像路径，默认使用配置文件中的路径
        output_path (str): 输出图像路径，默认使用配置文件中的路径
        params (dict): 美化参数，默认使用配置文件中的参数
    
    返回:
        numpy.ndarray: 美化后的图像
    """
    # 使用默认参数
    if image_path is None:
        image_path = config.BEAUTY_IMAGE_PATH
    if output_path is None:
        output_path = config.BEAUTY_OUTPUT_PATH
    if params is None:
        params = config.BEAUTY_PARAMS
    
    # 读取图像
    print(f"正在读取图像: {image_path}")
    image = read_image(image_path, color_mode='rgb')
    
    # 应用美化处理
    print("正在应用美化处理...")
    
    # 1. 应用CLAHE增强对比度
    enhanced = apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8))
    
    # 2. 调整亮度
    enhanced = adjust_brightness(enhanced, params['brightness_factor'])
    
    # 3. 调整对比度
    enhanced = adjust_contrast(enhanced, params['contrast_factor'])
    
    # 4. 调整饱和度
    enhanced = adjust_saturation(enhanced, params['saturation_factor'])
    
    # 5. 应用双边滤波保持边缘的同时平滑皮肤
    enhanced = bilateral_filter(
        enhanced, 
        d=params['bilateral_d'],
        sigma_color=params['bilateral_sigma_color'],
        sigma_space=params['bilateral_sigma_space']
    )
    
    # 6. 应用锐化增强细节
    enhanced = unsharp_mask(enhanced, kernel_size=5, strength=params['sharpness_factor'])
    
    # 保存结果
    save_image(enhanced, output_path, color_mode='rgb')
    
    return enhanced

def main():
    """
    主函数
    """
    # 确保输入图像存在
    if not os.path.exists(config.BEAUTY_IMAGE_PATH):
        print(f"错误: 输入图像不存在 {config.BEAUTY_IMAGE_PATH}")
        print(f"请将美女图像放置在 {config.BEAUTY_IMAGE_PATH} 路径下")
        return
    
    # 执行美化处理
    beautify_image()
    print(f"美化处理完成，结果已保存至: {config.BEAUTY_OUTPUT_PATH}")

if __name__ == "__main__":
    main()