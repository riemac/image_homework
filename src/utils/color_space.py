"""
色彩空间转换工具模块
提供各种色彩空间之间的转换功能
"""
import cv2
import numpy as np

def rgb_to_hsv(image):
    """
    RGB图像转换为HSV色彩空间
    
    参数:
        image (numpy.ndarray): RGB格式的图像数据
        
    返回:
        numpy.ndarray: HSV格式的图像数据
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

def hsv_to_rgb(image):
    """
    HSV图像转换为RGB色彩空间
    
    参数:
        image (numpy.ndarray): HSV格式的图像数据
        
    返回:
        numpy.ndarray: RGB格式的图像数据
    """
    return cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

def rgb_to_gray(image):
    """
    RGB图像转换为灰度图
    
    参数:
        image (numpy.ndarray): RGB格式的图像数据
        
    返回:
        numpy.ndarray: 灰度图像数据
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def adjust_brightness(image, factor):
    """
    调整图像亮度
    
    参数:
        image (numpy.ndarray): RGB格式的图像数据
        factor (float): 亮度调整因子，>1增加亮度，<1降低亮度
        
    返回:
        numpy.ndarray: 调整后的图像
    """
    hsv_img = rgb_to_hsv(image)
    hsv_img[:, :, 2] = np.clip(hsv_img[:, :, 2] * factor, 0, 255)
    return hsv_to_rgb(hsv_img)

def adjust_saturation(image, factor):
    """
    调整图像饱和度
    
    参数:
        image (numpy.ndarray): RGB格式的图像数据
        factor (float): 饱和度调整因子，>1增加饱和度，<1降低饱和度
        
    返回:
        numpy.ndarray: 调整后的图像
    """
    hsv_img = rgb_to_hsv(image)
    hsv_img[:, :, 1] = np.clip(hsv_img[:, :, 1] * factor, 0, 255)
    return hsv_to_rgb(hsv_img)

def adjust_contrast(image, factor):
    """
    调整图像对比度
    
    参数:
        image (numpy.ndarray): RGB格式的图像数据
        factor (float): 对比度调整因子，>1增加对比度，<1降低对比度
        
    返回:
        numpy.ndarray: 调整后的图像
    """
    mean = np.mean(image, axis=(0, 1))
    adjusted = factor * (image - mean) + mean
    return np.clip(adjusted, 0, 255).astype(np.uint8)