"""
苹果橘子融合模块
实现对苹果和橘子图像的融合处理
"""
import os
import cv2
import numpy as np

from utils.image_io import read_image, save_image
from utils.filters import gaussian_blur, edge_detection
import config

def resize_to_match(img1, img2):
    """
    调整第二张图片的大小以匹配第一张图片
    
    参数:
        img1 (numpy.ndarray): 参考图像
        img2 (numpy.ndarray): 需要调整大小的图像
        
    返回:
        numpy.ndarray: 调整大小后的图像
    """
    h1, w1 = img1.shape[:2]
    return cv2.resize(img2, (w1, h1), interpolation=cv2.INTER_AREA)

def create_mask(image, method='threshold'):
    """
    创建图像掩码
    
    参数:
        image (numpy.ndarray): 输入图像
        method (str): 掩码创建方法，可选 'threshold', 'edge'
        
    返回:
        numpy.ndarray: 掩码图像
    """
    # 转为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    if method == 'threshold':
        # 使用阈值分割创建掩码
        _, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    elif method == 'edge':
        # 使用边缘检测创建掩码
        edges = edge_detection(image, method='canny')
        # 膨胀边缘
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(edges, kernel, iterations=2)
    else:
        raise ValueError(f"不支持的掩码创建方法: {method}")
    
    return mask

def blend_images(apple_path=None, orange_path=None, output_path=None, params=None):
    """
    融合苹果和橘子图像
    
    参数:
        apple_path (str): 苹果图像路径，默认使用配置文件中的路径
        橘子_path (str): 橘子图像路径，默认使用配置文件中的路径
        output_path (str): 输出图像路径，默认使用配置文件中的路径
        params (dict): 融合参数，默认使用配置文件中的参数
    
    返回:
        numpy.ndarray: 融合后的图像
    """
    # 使用默认参数
    if apple_path is None:
        apple_path = config.APPLE_IMAGE_PATH
    if orange_path is None:
        orange_path = config.orange_IMAGE_PATH
    if output_path is None:
        output_path = config.FUSION_OUTPUT_PATH
    if params is None:
        params = config.FUSION_PARAMS
    
    # 读取图像
    print(f"正在读取苹果图像: {apple_path}")
    apple_img = read_image(apple_path, color_mode='rgb')
    
    print(f"正在读取橘子图像: {orange_path}")
    orange_img = read_image(orange_path, color_mode='rgb')
    
    # 调整橘子图像大小以匹配苹果图像
    orange_img = resize_to_match(apple_img, orange_img)
    
    # 方法1: 简单的线性融合
    alpha = params['alpha']
    simple_blend = cv2.addWeighted(apple_img, alpha, orange_img, 1-alpha, 0)
    
    # 方法2: 使用拉普拉斯金字塔融合
    # 创建高斯金字塔
    apple_copy = apple_img.copy()
    orange_copy = orange_img.copy()
    
    # 创建高斯金字塔
    apple_pyramid = [apple_copy]
    orange_pyramid = [orange_copy]
    
    for i in range(6):
        apple_copy = cv2.pyrDown(apple_copy)
        orange_copy = cv2.pyrDown(orange_copy)
        apple_pyramid.append(apple_copy)
        orange_pyramid.append(orange_copy)
    
    # 创建拉普拉斯金字塔
    apple_laplacian = [apple_pyramid[5]]
    orange_laplacian = [orange_pyramid[5]]
    
    for i in range(5, 0, -1):
        apple_expanded = cv2.pyrUp(apple_pyramid[i])
        orange_expanded = cv2.pyrUp(orange_pyramid[i])
        
        # 调整大小以匹配上一级
        apple_expanded = cv2.resize(apple_expanded, (apple_pyramid[i-1].shape[1], apple_pyramid[i-1].shape[0]))
        orange_expanded = cv2.resize(orange_expanded, (orange_pyramid[i-1].shape[1], orange_pyramid[i-1].shape[0]))
        
        apple_laplacian.append(cv2.subtract(apple_pyramid[i-1], apple_expanded))
        orange_laplacian.append(cv2.subtract(orange_pyramid[i-1], orange_expanded))
    
    # 融合拉普拉斯金字塔
    laplacian_merged = []
    for apple_lap, orange_lap in zip(apple_laplacian, orange_laplacian):
        laplacian_merged.append(cv2.addWeighted(apple_lap, alpha, orange_lap, 1-alpha, 0))
    
    # 重建融合图像
    reconstruction = laplacian_merged[0]
    for i in range(1, 6):
        reconstruction = cv2.pyrUp(reconstruction)
        reconstruction = cv2.resize(reconstruction, (laplacian_merged[i].shape[1], laplacian_merged[i].shape[0]))
        reconstruction = cv2.add(reconstruction, laplacian_merged[i])
    
    # 应用高斯模糊平滑过渡
    pyramid_blend = reconstruction.astype(np.uint8)
    pyramid_blend = gaussian_blur(pyramid_blend, kernel_size=params['gaussian_blur_kernel'])
    
    # 保存结果
    save_image(simple_blend, os.path.join(config.OUTPUT_DIR, 'simple_blend.jpg'), color_mode='rgb')
    save_image(pyramid_blend, output_path, color_mode='rgb')
    
    return pyramid_blend

def main():
    """
    主函数
    """
    # 确保输入图像存在
    if not os.path.exists(config.APPLE_IMAGE_PATH):
        print(f"错误: 苹果图像不存在 {config.APPLE_IMAGE_PATH}")
        print(f"请将苹果图像放置在 {config.APPLE_IMAGE_PATH} 路径下")
        return
    
    if not os.path.exists(config.orange_IMAGE_PATH):
        print(f"错误: 橘子图像不存在 {config.orange_IMAGE_PATH}")
        print(f"请将橘子图像放置在 {config.orange_IMAGE_PATH} 路径下")
        return
    
    # 执行融合处理
    blend_images()
    print(f"融合处理完成，结果已保存至: {config.FUSION_OUTPUT_PATH}")

if __name__ == "__main__":
    main()