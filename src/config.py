"""
配置参数模块
存储项目中使用的各种参数和路径
"""
import os

# 项目根目录
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据目录
DATA_DIR = os.path.join(ROOT_DIR, 'data')
INPUT_DIR = os.path.join(DATA_DIR, 'input')
OUTPUT_DIR = os.path.join(DATA_DIR, 'output')

# 输入图像路径
BEAUTY_IMAGE_PATH = os.path.join(INPUT_DIR, 'beauty.jpg')
APPLE_IMAGE_PATH = os.path.join(INPUT_DIR, 'apple.jpg')
PEAR_IMAGE_PATH = os.path.join(INPUT_DIR, 'pear.jpg')

# 输出图像路径
BEAUTY_OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'beauty_enhanced.jpg')
FUSION_OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'apple_pear_fusion.jpg')

# 美化参数
BEAUTY_PARAMS = {
    'brightness_factor': 1.1,    # 亮度调整
    'contrast_factor': 1.2,      # 对比度调整
    'saturation_factor': 1.3,    # 饱和度调整
    'sharpness_factor': 1.5,     # 锐化强度
    'bilateral_d': 9,            # 双边滤波参数
    'bilateral_sigma_color': 75, # 双边滤波颜色标准差
    'bilateral_sigma_space': 75, # 双边滤波空间标准差
}

# 融合参数
FUSION_PARAMS = {
    'alpha': 0.5,                # 融合比例
    'gaussian_blur_kernel': 5,   # 高斯模糊核大小
    'edge_detection_method': 'sobel', # 边缘检测方法
}

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)