"""
滤波器集合模块
提供各种图像滤波和增强功能
"""
import cv2
import numpy as np
from scipy import ndimage

def gaussian_blur(image, kernel_size=5, sigma=0):
    """
    高斯模糊
    
    参数:
        image (numpy.ndarray): 输入图像
        kernel_size (int): 高斯核大小，必须是奇数
        sigma (float): 高斯核标准差，0表示自动计算
        
    返回:
        numpy.ndarray: 模糊后的图像
    """
    if kernel_size % 2 == 0:
        kernel_size += 1  # 确保核大小为奇数
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

def median_blur(image, kernel_size=5):
    """
    中值滤波，对去除椒盐噪声很有效
    
    参数:
        image (numpy.ndarray): 输入图像
        kernel_size (int): 核大小，必须是奇数
        
    返回:
        numpy.ndarray: 滤波后的图像
    """
    if kernel_size % 2 == 0:
        kernel_size += 1  # 确保核大小为奇数
    return cv2.medianBlur(image, kernel_size)

def bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
    """
    双边滤波，可以在保持边缘的同时平滑图像
    
    参数:
        image (numpy.ndarray): 输入图像
        d (int): 过滤期间使用的每个像素邻域的直径
        sigma_color (float): 颜色空间中的标准差
        sigma_space (float): 坐标空间中的标准差
        
    返回:
        numpy.ndarray: 滤波后的图像
    """
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

def unsharp_mask(image, kernel_size=5, strength=1.5):
    """
    锐化滤波（反锐化掩蔽）
    
    参数:
        image (numpy.ndarray): 输入图像
        kernel_size (int): 高斯核大小
        strength (float): 锐化强度
        
    返回:
        numpy.ndarray: 锐化后的图像
    """
    blurred = gaussian_blur(image, kernel_size)
    sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)

def edge_detection(image, method='sobel', low_threshold=None, high_threshold=None):
    """
    边缘检测
    
    参数:
        image (numpy.ndarray): 输入图像
        method (str): 边缘检测方法，可选 'sobel', 'canny'
        low_threshold (int, optional): Canny边缘检测的低阈值。仅当 method='canny' 时有效。
        high_threshold (int, optional): Canny边缘检测的高阈值。仅当 method='canny' 时有效。
        
    返回:
        numpy.ndarray: 边缘检测结果
    """
    # 如果是彩色图像，先转为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
        
    if method.lower() == 'sobel':
        # Sobel边缘检测
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        # 计算梯度幅值
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        # 归一化到0-255
        magnitude = np.uint8(255 * magnitude / np.max(magnitude))
        return magnitude
    
    elif method.lower() == 'canny':
        if low_threshold is None or high_threshold is None:
            # 提供默认值或抛出错误，这里选择抛出错误，因为Canny通常需要这两个参数
            raise ValueError("对于Canny边缘检测，'low_threshold' 和 'high_threshold' 不能为空。")
        # Canny边缘检测
        return cv2.Canny(gray, low_threshold, high_threshold)
    
    else:
        raise ValueError(f"不支持的边缘检测方法: {method}")

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    应用CLAHE（对比度受限的自适应直方图均衡化）
    
    参数:
        image (numpy.ndarray): 输入图像
        clip_limit (float): 对比度限制
        tile_grid_size (tuple): 网格大小
        
    返回:
        numpy.ndarray: 增强后的图像
    """
    # 如果是彩色图像，在LAB空间中只对L通道应用CLAHE
    if len(image.shape) == 3:
        # 转换到LAB颜色空间
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        # 分离通道
        l, a, b = cv2.split(lab)
        
        # 创建CLAHE对象
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        # 应用CLAHE到L通道
        l = clahe.apply(l)
        
        # 合并通道
        lab = cv2.merge((l, a, b))
        # 转换回RGB
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    else:
        # 灰度图像直接应用CLAHE
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(image)

def remove_shadow(image, kernel_size=(15, 15)):
    """
    使用形态学开运算去除图像中的阴影。
    
    参数:
        image (numpy.ndarray): 输入灰度图像。
        kernel_size (tuple): 用于形态学开运算的核大小。应足够大以覆盖阴影区域，同时小于目标物体尺寸。
        
    返回:
        numpy.ndarray: 去除阴影后的图像。
    """
    # 确保图像是灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
        
    # 创建形态学核
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    
    # 执行开运算来估算背景光照（阴影）
    background = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    
    # 从原图中减去背景光照以去除阴影
    # 使用 cv2.subtract 而非简单的减法，以处理负值情况并将其裁剪到0
    corrected_image = cv2.subtract(gray, background)
    
    # 结果可能对比度较低，进行一些简单的归一化或对比度增强
    # 这里可以根据需要添加 CLAHE 或其他增强
    corrected_image = cv2.normalize(corrected_image, None, 0, 255, cv2.NORM_MINMAX)
    
    return corrected_image.astype(np.uint8)