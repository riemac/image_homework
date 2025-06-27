"""
图像读写工具模块
提供图像读取、保存和基本处理功能
"""
import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt # 导入matplotlib用于Jupyter内联显示
import sys # 导入sys模块用于检测运行环境

def read_image(image_path, color_mode='rgb'):
    """
    读取图像文件
    
    参数:
        image_path (str): 图像文件路径
        color_mode (str): 颜色模式，可选 'rgb', 'bgr', 'gray'
        
    返回:
        numpy.ndarray: 图像数据
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像文件不存在: {image_path}")
    
    if color_mode.lower() == 'bgr':
        # OpenCV默认读取为BGR
        return cv2.imread(image_path)
    elif color_mode.lower() == 'rgb':
        # 读取为RGB
        img = cv2.imread(image_path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif color_mode.lower() == 'gray':
        # 读取为灰度图
        return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        raise ValueError(f"不支持的颜色模式: {color_mode}")

def save_image(image, save_path, color_mode='rgb'):
    """
    保存图像文件
    
    参数:
        image (numpy.ndarray): 图像数据
        save_path (str): 保存路径
        color_mode (str): 颜色模式，可选 'rgb', 'bgr', 'gray'
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    if color_mode.lower() == 'rgb':
        # 转换为BGR后保存
        cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    elif color_mode.lower() == 'bgr':
        # 直接保存BGR
        cv2.imwrite(save_path, image)
    elif color_mode.lower() == 'gray':
        # 保存灰度图
        cv2.imwrite(save_path, image)
    else:
        raise ValueError(f"不支持的颜色模式: {color_mode}")
    
    print(f"图像已保存至: {save_path}")

def display_image(image, title="Image", color_mode='rgb'):
    """
    使用OpenCV或Matplotlib显示图像，根据运行环境自动调整
    
    参数:
        image (numpy.ndarray): 图像数据
        title (str): 窗口标题
        color_mode (str): 颜色模式，可选 'rgb', 'bgr', 'gray'
    """
    # 检查是否在Jupyter Notebook环境中运行
    if 'ipykernel' in sys.modules or 'jupyter_client' in sys.modules:
        # Jupyter环境下使用matplotlib显示
        plt.figure(figsize=(10, 8))
        if color_mode.lower() == 'rgb':
            plt.imshow(image)
        elif color_mode.lower() == 'bgr':
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        elif color_mode.lower() == 'gray':
            plt.imshow(image, cmap='gray')
        else:
            raise ValueError(f"不支持的颜色模式: {color_mode}")
        plt.title(title)
        plt.axis('off') # 不显示坐标轴
        plt.show()
    else:
        # 非Jupyter环境下使用OpenCV显示
        if color_mode.lower() == 'rgb':
            display_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            display_img = image
        
        cv2.imshow(title, display_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()