import cv2
import numpy as np
import os

from src.utils.image_io import read_image, save_image, display_image
from src.utils.filters import gaussian_blur, edge_detection, apply_clahe, remove_shadow

def locate_vehicle_number(image_path,
                          gaussian_kernel_size=5,
                          canny_low_threshold=100,
                          canny_high_threshold=200,
                          morph_kernel_width=15,
                          morph_kernel_height=3,
                          min_contour_area=100,
                          max_contour_area=30000,
                          min_aspect_ratio=2.0,
                          max_aspect_ratio=6.0,
                          save_results=False,
                          output_dir="data/output",
                          apply_clahe_preprocess=False,
                          clahe_clip_limit=2.0,
                          clahe_tile_grid_size=(8, 8),
                          enable_angle_correction=False,
                          angle_correction_threshold=1.0,
                          min_rect_height_for_angle_correction=10,
                          debug_mode=False,
                          apply_shadow_removal=False,
                          shadow_kernel_size=(30, 30)):
    """
    定位图像中的机动车车号。

    参数:
        image_path (str): 输入图像的路径。
        gaussian_kernel_size (int): 高斯模糊的核大小，必须是奇数。
        canny_low_threshold (int): Canny边缘检测的低阈值。
        canny_high_threshold (int): Canny边缘检测的高阈值。
        morph_kernel_width (int): 形态学闭运算的核宽度。
        morph_kernel_height (int): 形态学闭运算的核高度。
        min_contour_area (int): 轮廓筛选的最小面积。
        max_contour_area (int): 轮廓筛选的最大面积。
        min_aspect_ratio (float): 轮廓筛选的最小长宽比。
        max_aspect_ratio (float): 轮廓筛选的最大长宽比。
        save_results (bool): 是否保存结果图像。
        output_dir (str): 结果图像的保存目录。
        apply_clahe_preprocess (bool): 是否应用CLAHE预处理。
        clahe_clip_limit (float): CLAHE的对比度限制。
        clahe_tile_grid_size (tuple): CLAHE的网格大小。
        enable_angle_correction (bool): 是否启用角度校正功能。
        angle_correction_threshold (float): 角度校正的阈值（度），小于此角度认为不需要校正。
        min_rect_height_for_angle_correction (int): 最小矩形高度，避免对太小的轮廓进行角度校正。
        debug_mode (bool): 是否启用调试模式，显示处理过程中的中间图像。
        apply_shadow_removal (bool): 是否在预处理阶段应用阴影去除。
        shadow_kernel_size (tuple): 用于阴影去除的形态学核大小。

    返回:
        numpy.ndarray: 绘制了车号定位框的图像。
    """
    # 1. 读取原始彩色图像和灰度图像
    original_image = read_image(image_path, color_mode='rgb')
    gray_image = read_image(image_path, color_mode='gray')

    if debug_mode:
        display_image(original_image, title=f"Debug: Original - {os.path.basename(image_path)}")
        display_image(gray_image, title=f"Debug: Grayscale - {os.path.basename(image_path)}", color_mode='gray')

    current_image = gray_image

    # 2.1 (可选) 预处理：应用阴影去除
    if apply_shadow_removal:
        current_image = remove_shadow(current_image, kernel_size=shadow_kernel_size)
        if debug_mode:
            display_image(current_image, title=f"Debug: Shadow Removed - {os.path.basename(image_path)}", color_mode='gray')

    # 2.2 (可选) 预处理：应用CLAHE增强对比度
    if apply_clahe_preprocess:
        processed_image = apply_clahe(current_image, clip_limit=clahe_clip_limit, tile_grid_size=clahe_tile_grid_size)
        if debug_mode:
            display_image(processed_image, title=f"Debug: CLAHE Processed - {os.path.basename(image_path)}", color_mode='gray')
    else:
        processed_image = current_image

    # 2.3 预处理：高斯模糊
    blurred_image = gaussian_blur(processed_image, kernel_size=gaussian_kernel_size)
    if debug_mode:
        display_image(blurred_image, title=f"Debug: Blurred - {os.path.basename(image_path)}", color_mode='gray')

    # 3. 边缘检测：Canny
    edges = edge_detection(blurred_image, method='canny', low_threshold=canny_low_threshold, high_threshold=canny_high_threshold)
    if debug_mode:
        display_image(edges, title=f"Debug: Canny Edges - {os.path.basename(image_path)}", color_mode='gray')

    # 4. 形态学操作：闭运算以连接边缘
    # 创建一个矩形结构元素
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel_width, morph_kernel_height))
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    if debug_mode:
        display_image(closed_edges, title=f"Debug: Closed Edges - {os.path.basename(image_path)}", color_mode='gray')

    # 5. 查找轮廓
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 复制原始图像用于绘制
    result_image = original_image.copy()
    detected_plates = []

    # 6. 筛选轮廓
    candidates = [] # 存储可能的车牌候选区域
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_contour_area or area > max_contour_area:
            continue

        rect = cv2.minAreaRect(contour)
        width = rect[1][0]
        height = rect[1][1]
        angle = rect[2] # 矩形角度

        # 确保宽度总是大于高度
        if width < height:
            width, height = height, width
            angle += 90 # 对应角度调整
        
        aspect_ratio = width / float(height)

        # 简单的长宽比和面积筛选，先找出潜在的候选，角度校正后再精细筛选
        if min_aspect_ratio < aspect_ratio < max_aspect_ratio:
            candidates.append((contour, rect, area, aspect_ratio, angle))

    final_boxes = []

    if enable_angle_correction and candidates: # 如果启用了角度校正且有候选区域
        # 尝试对每个候选区域进行角度校正
        for contour, rect, area, aspect_ratio, angle in candidates:
            # 只对有一定倾斜角度且高度足够的矩形进行校正，避免对小的噪声区域进行复杂处理
            if abs(angle) > angle_correction_threshold and rect[1][1] >= min_rect_height_for_angle_correction:
                center, (w, h), angle = rect
                if w < h: # 再次确认宽度大于高度，并调整角度到-90到0之间
                    angle += 90

                # 计算旋转矩阵
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                
                # 获取旋转后的新图像尺寸
                # rotated_w = int(h * abs(np.sin(np.radians(angle))) + w * abs(np.cos(np.radians(angle))))
                # rotated_h = int(h * abs(np.cos(np.radians(angle))) + w * abs(np.sin(np.radians(angle))))
                # M[0, 2] += (rotated_w / 2) - center[0]
                # M[1, 2] += (rotated_h / 2) - center[1]

                # 旋转图像，保持原始图像尺寸
                rotated_gray = cv2.warpAffine(gray_image, M, (gray_image.shape[1], gray_image.shape[0]), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                if debug_mode:
                    display_image(rotated_gray, title=f"Debug: Rotated Gray - {os.path.basename(image_path)}", color_mode='gray')

                # 在旋转后的图像上再次进行处理流程
                if apply_clahe_preprocess:
                    rotated_processed = apply_clahe(rotated_gray, clip_limit=clahe_clip_limit, tile_grid_size=clahe_tile_grid_size)
                    if debug_mode:
                        display_image(rotated_processed, title=f"Debug: Rotated CLAHE - {os.path.basename(image_path)}", color_mode='gray')
                else:
                    rotated_processed = rotated_gray

                rotated_blurred = gaussian_blur(rotated_processed, kernel_size=gaussian_kernel_size)
                if debug_mode:
                    display_image(rotated_blurred, title=f"Debug: Rotated Blurred - {os.path.basename(image_path)}", color_mode='gray')

                rotated_edges = edge_detection(rotated_blurred, method='canny', low_threshold=canny_low_threshold, high_threshold=canny_high_threshold)
                if debug_mode:
                    display_image(rotated_edges, title=f"Debug: Rotated Canny - {os.path.basename(image_path)}", color_mode='gray')

                rotated_closed_edges = cv2.morphologyEx(rotated_edges, cv2.MORPH_CLOSE, kernel) # 使用相同的核
                if debug_mode:
                    display_image(rotated_closed_edges, title=f"Debug: Rotated Closed Edges - {os.path.basename(image_path)}", color_mode='gray')

                # 在旋转后的图像上查找轮廓
                rotated_contours, _ = cv2.findContours(rotated_closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # 筛选旋转后的轮廓
                for r_contour in rotated_contours:
                    r_area = cv2.contourArea(r_contour)
                    if r_area < min_contour_area or r_area > max_contour_area:
                        continue

                    r_rect = cv2.minAreaRect(r_contour)
                    r_width = r_rect[1][0]
                    r_height = r_rect[1][1]

                    if r_width < r_height:
                        r_width, r_height = r_height, r_width
                    
                    r_aspect_ratio = r_width / float(r_height)

                    # 在旋转校正后，长宽比和面积筛选应该更严格
                    if min_aspect_ratio < r_aspect_ratio < max_aspect_ratio and min_contour_area < r_area < max_contour_area:
                        # 将旋转后的矩形顶点反向变换回原始图像坐标
                        r_box = cv2.boxPoints(r_rect)
                        r_box = r_box.astype(np.int32)
                        
                        # 构建逆旋转矩阵
                        M_inv = cv2.getRotationMatrix2D(center, -angle, 1.0)
                        # M_inv[0, 2] += center[0] - (rotated_w / 2)
                        # M_inv[1, 2] += center[1] - (rotated_h / 2)

                        # 变换回原始坐标
                        transformed_box = cv2.transform(np.array([r_box]), M_inv)[0]
                        transformed_box = transformed_box.astype(np.int32)
                        final_boxes.append(transformed_box)
            else:
                # 如果不进行角度校正，直接使用原始轮廓的最小外接矩形
                box = cv2.boxPoints(rect)
                box = box.astype(np.int32)
                final_boxes.append(box)
    else: # 如果没有启用角度校正，或者没有候选区域，则使用原始筛选的轮廓
        for contour, rect, area, aspect_ratio, angle in candidates:
            box = cv2.boxPoints(rect)
            box = box.astype(np.int32)
            final_boxes.append(box)
    
    # 绘制最终检测到的车牌框
    for box in final_boxes:
        cv2.drawContours(result_image, [box], 0, (0, 255, 0), 2) # 绿色边框

    if save_results:
        # 构建输出路径
        base_name = os.path.basename(image_path)
        name, ext = os.path.splitext(base_name)
        output_path = os.path.join(output_dir, f"{name}_detected{ext}")
        save_image(result_image, output_path, color_mode='rgb')
        print(f"检测结果已保存到: {output_path}")

    return result_image

if __name__ == '__main__':
    # 示例用法
    import os
    input_images = [
        "data/input/vehicle1.png",
        "data/input/vehicle2.png",
        "data/input/vehicle3.png",
        "data/input/vehicle4.png",
    ]
    
    # 可以在这里调整参数进行测试
    params = {
        "gaussian_kernel_size": 5,
        "canny_low_threshold": 100,
        "canny_high_threshold": 200,
        "morph_kernel_width": 15,
        "morph_kernel_height": 3,
        "min_contour_area": 100,
        "max_contour_area": 30000,
        "min_aspect_ratio": 2.0,
        "max_aspect_ratio": 6.0,
        "apply_clahe_preprocess": False,
        "clahe_clip_limit": 2.0,
        "clahe_tile_grid_size": (8, 8),
        "enable_angle_correction": False,
        "angle_correction_threshold": 1.0,
        "min_rect_height_for_angle_correction": 10,
        "debug_mode": False,
        "apply_shadow_removal": False,
        "shadow_kernel_size": (30, 30),
    }

    for img_path in input_images:
        print(f"处理图像: {img_path}")
        result_img = locate_vehicle_number(img_path, save_results=True, **params)
        # 如果需要在脚本运行时显示图像，可以取消注释下面这行
        display_image(result_img, title=f"Detected - {os.path.basename(img_path)}")
