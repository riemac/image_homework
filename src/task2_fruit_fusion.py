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

def normalize_color(src_img, ref_img):
    """
    将src_img的亮度和色调归一化到ref_img
    """
    src_yuv = cv2.cvtColor(src_img, cv2.COLOR_RGB2YUV)
    ref_yuv = cv2.cvtColor(ref_img, cv2.COLOR_RGB2YUV)
    # 只调整Y通道
    src_y, src_u, src_v = cv2.split(src_yuv)
    ref_y, _, _ = cv2.split(ref_yuv)
    src_y = src_y.astype(np.float32)
    ref_y = ref_y.astype(np.float32)
    # 匹配均值和方差
    src_y = (src_y - src_y.mean()) / (src_y.std() + 1e-5) * (ref_y.std() + 1e-5) + ref_y.mean()
    src_y = np.clip(src_y, 0, 255).astype(np.uint8)
    norm_yuv = cv2.merge([src_y, src_u, src_v])
    return cv2.cvtColor(norm_yuv, cv2.COLOR_YUV2RGB)

def create_gradient_mask(h, w, band_width_ratio=0.15):
    """
    创建宽渐变带的左右掩码
    band_width_ratio: 渐变带宽度占总宽度比例
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    band_width = int(w * band_width_ratio)
    mask[:, w//2:] = 255
    gradient = np.tile(np.linspace(0, 255, band_width).astype(np.uint8), (h, 1))
    mask[:, w//2-band_width:w//2] = gradient
    mask = cv2.GaussianBlur(mask, (51, 51), 0)
    return mask

def blend_apple_orange(
    apple_img, orange_img, 
    band_width_ratio=0.15, 
    center_offset_x=0, center_offset_y=0, 
    mode=cv2.MIXED_CLONE
):
    h, w = apple_img.shape[:2]
    # 归一化橙子亮度到苹果
    orange_img_norm = normalize_color(orange_img, apple_img)
    # 生成渐变掩码
    mask = create_gradient_mask(h, w, band_width_ratio)
    # 融合中心
    center = (w//2 + center_offset_x, h//2 + center_offset_y)
    # 融合
    result = cv2.seamlessClone(orange_img_norm, apple_img, mask, center, mode)
    return result, mask

def half_half_blend(img1, img2, band_width_ratio=0.08):
    """
    左半为img1，右半为img2，中间渐变过渡
    band_width_ratio: 渐变带宽度占总宽度比例
    """
    h, w = img1.shape[:2]
    band_width = int(w * band_width_ratio)
    mask = np.zeros((h, w), dtype=np.float32)
    mask[:, :w//2-band_width//2] = 0
    mask[:, w//2+band_width//2:] = 1
    # 渐变带
    for i in range(band_width):
        alpha = i / band_width
        mask[:, w//2-band_width//2+i] = alpha
    # 扩展到3通道
    mask3 = np.stack([mask]*3, axis=2)
    blended = img1 * (1 - mask3) + img2 * mask3
    return blended.astype(np.uint8), (mask*255).astype(np.uint8)

def main():
    apple_path = config.APPLE_IMAGE_PATH
    orange_path = config.ORANGE_IMAGE_PATH
    output_path = config.FUSION_OUTPUT_PATH
    # 读取图片
    apple_img = read_image(apple_path, color_mode='rgb')
    orange_img = read_image(orange_path, color_mode='rgb')
    # 尺寸匹配
    orange_img = cv2.resize(orange_img, (apple_img.shape[1], apple_img.shape[0]))
    # 融合
    result, mask = blend_apple_orange(apple_img, orange_img)
    # 保存
    save_image(result, output_path, color_mode='rgb')
    print(f"融合完成，结果已保存至: {output_path}")

if __name__ == "__main__":
    main()
