"""
图像处理作业主程序
可以运行不同的任务
"""
import os
import argparse

from src.task1_beautification import beautify_image
from src.task2_fruit_fusion import blend_images
import src.config as config

def main():
    """
    主函数，解析命令行参数并执行相应任务
    """
    parser = argparse.ArgumentParser(description='图像处理作业')
    parser.add_argument('--task', type=str, choices=['beauty', 'fusion', 'all'], 
                        default='all', help='要执行的任务')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    if args.task == 'beauty' or args.task == 'all':
        print("\n=== 执行任务1: 美女图像美化 ===")
        if not os.path.exists(config.BEAUTY_IMAGE_PATH):
            print(f"错误: 美女图像不存在 {config.BEAUTY_IMAGE_PATH}")
            print(f"请将美女图像放置在 {config.BEAUTY_IMAGE_PATH} 路径下")
        else:
            beautify_image()
            print(f"美化处理完成，结果已保存至: {config.BEAUTY_OUTPUT_PATH}")
    
    if args.task == 'fusion' or args.task == 'all':
        print("\n=== 执行任务2: 苹果梨子融合 ===")
        if not os.path.exists(config.APPLE_IMAGE_PATH):
            print(f"错误: 苹果图像不存在 {config.APPLE_IMAGE_PATH}")
            print(f"请将苹果图像放置在 {config.APPLE_IMAGE_PATH} 路径下")
        elif not os.path.exists(config.PEAR_IMAGE_PATH):
            print(f"错误: 梨子图像不存在 {config.PEAR_IMAGE_PATH}")
            print(f"请将梨子图像放置在 {config.PEAR_IMAGE_PATH} 路径下")
        else:
            blend_images()
            print(f"融合处理完成，结果已保存至: {config.FUSION_OUTPUT_PATH}")
    
    print("\n所有任务执行完毕！")

if __name__ == "__main__":
    main()